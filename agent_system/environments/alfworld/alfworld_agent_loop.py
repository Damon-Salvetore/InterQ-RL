from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional
from uuid import uuid4

from agent_system.environments.alfworld.envs.tool_executor import INVALID_TOOL_CALL_ACTION, AlfworldToolExecutor
from agent_system.environments.alfworld.prompts import ALFWORLD_TOOL_SCHEMAS
from agent_system.environments.alfworld.utils import (
    build_alfworld_messages,
    build_invalid_tool_call_observation,
    extract_task_text,
    format_admissible_commands,
)

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_TOOL_CALL_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def _recover_tool_calls_from_text(text: str) -> list[FunctionCall]:
    recovered: list[FunctionCall] = []
    for raw in _TOOL_CALL_BLOCK.findall(text):
        try:
            payload = json.loads(raw.strip())
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        name = payload.get("name")
        arguments = payload.get("arguments")
        if not isinstance(name, str):
            continue
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except Exception:
                continue
        if not isinstance(arguments, dict):
            continue
        recovered.append(FunctionCall(name=name, arguments=json.dumps(arguments, ensure_ascii=False)))
    return recovered


def _parse_env_step(tool_calls: list[FunctionCall]) -> tuple[Optional[str], Optional[str]]:
    if not tool_calls:
        return None, "missing env_step tool call"
    tool_call = tool_calls[0]
    if tool_call.name != "env_step":
        return None, f"expected env_step tool, got {tool_call.name!r}"
    try:
        tool_args = json.loads(tool_call.arguments)
    except Exception as e:
        return None, f"failed to parse env_step arguments: {e}"
    if not isinstance(tool_args, dict):
        return None, f"env_step arguments not a dict (got {type(tool_args).__name__})"
    command = str(tool_args.get("command", "")).strip()
    if not command:
        return None, "missing command argument"
    return command, None


@register("alfworld_agent")
class AlfworldAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.tool_parser = ToolParser.get_tool_parser(
            self.rollout_config.multi_turn.format,
            self.tokenizer,
        )
        self.tool_schemas = ALFWORLD_TOOL_SCHEMAS
        self.max_steps = int(kwargs.get("max_steps", 30))
        self.max_episode_steps = int(kwargs.get("max_episode_steps", 50))
        self.data_root = str(kwargs.get("data_root", "data/alfworld"))

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        extra_info = kwargs.get("extra_info") or {}
        if isinstance(extra_info, str):
            try:
                extra_info = json.loads(extra_info)
            except json.JSONDecodeError:
                extra_info = {}
        game_relative_path = extra_info.get("game_relative_path")
        if not game_relative_path:
            raise ValueError("ALFWorld sample is missing extra_info.game_relative_path")
        task_id = extra_info.get("task_id")
        goal_text = extra_info.get("goal_text")

        metrics: dict[str, Any] = {}

        executor = AlfworldToolExecutor(
            max_episode_steps=self.max_episode_steps,
            data_root=self.data_root,
        )
        obs, reset_info = executor.reset_with_info(game_relative_path=game_relative_path, task_id=task_id)
        task_text = extract_task_text(obs, goal_text)
        admissible = reset_info.get("admissible_commands")
        admissible = admissible if isinstance(admissible, list) else []

        init_messages = build_alfworld_messages(
            task_text=task_text,
            observation=obs,
            history_actions=[],
            admissible_commands=admissible,
        )
        prompt_ids = await self.apply_chat_template(init_messages, tools=self.tool_schemas)

        all_ids: list[int] = list(prompt_ids)
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        request_id = str(task_id) if task_id else uuid4().hex
        num_turns = 0
        done = False
        success = False
        prev_obs = obs
        history_actions: list[str] = []
        min_global_steps: int | None = None
        max_global_steps: int | None = None

        sp = dict(sampling_params)
        if self.tool_parser.stop_token_ids:
            sp["stop_token_ids"] = list(set((sp.get("stop_token_ids") or []) + self.tool_parser.stop_token_ids))

        while num_turns < self.max_steps and not done:
            if len(response_mask) >= self.response_length:
                break

            with simple_timer("generate_sequences", metrics):
                output = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=all_ids,
                    sampling_params=sp,
                )
            gen_ids = list(output.token_ids)
            gen_lp = list(output.log_probs) if output.log_probs else [0.0] * len(gen_ids)
            gs = (getattr(output, "extra_fields", None) or {}).get("global_steps")
            if gs is not None:
                min_global_steps = gs if min_global_steps is None else min(min_global_steps, gs)
                max_global_steps = gs if max_global_steps is None else max(max_global_steps, gs)
            all_ids += gen_ids
            response_mask += [1] * len(gen_ids)
            response_logprobs += gen_lp
            num_turns += 1

            _, tool_calls = await self.tool_parser.extract_tool_calls(gen_ids)
            if not tool_calls:
                tool_calls = _recover_tool_calls_from_text(self.tokenizer.decode(gen_ids, skip_special_tokens=True))
            command, invalid_reason = _parse_env_step(tool_calls)

            if invalid_reason is None:
                result = executor.step(command)
                new_obs = str(result["observation"])
                done = bool(result["done"])
                info = result.get("info", {}) or {}
                if "success" in info:
                    success = bool(info["success"])
                elif "won" in info:
                    success = bool(info["won"])
                new_adm = info.get("admissible_commands")
                new_adm = new_adm if isinstance(new_adm, list) else []
                history_actions = result.get("history_actions", history_actions)
                tool_text = f"{new_obs}\n\n### Admissible Commands\n{format_admissible_commands(new_adm)}"
                prev_obs = new_obs
            else:
                tool_text = build_invalid_tool_call_observation(prev_obs, invalid_reason)
                history_actions.append(f"{INVALID_TOOL_CALL_ACTION}: {invalid_reason}")

            tool_msg = [{"role": "tool", "content": tool_text}]
            obs_ids = await self.apply_chat_template(tool_msg, remove_system_prompt=True)
            obs_ids = self.turn_separator + obs_ids
            if len(response_mask) + len(obs_ids) >= self.response_length:
                all_ids += obs_ids
                response_mask += [0] * len(obs_ids)
                response_logprobs += [0.0] * len(obs_ids)
                break
            all_ids += obs_ids
            response_mask += [0] * len(obs_ids)
            response_logprobs += [0.0] * len(obs_ids)

            if done:
                break

        n_resp = len(response_mask)
        response_ids = all_ids[-n_resp:] if n_resp else []
        final_prompt_ids = all_ids[: len(all_ids) - n_resp]

        response_ids = response_ids[: self.response_length]
        response_mask = response_mask[: self.response_length]
        response_logprobs = response_logprobs[: self.response_length]

        if not any(m == 1 for m in response_mask):
            pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
            response_ids.append(pad_id)
            response_mask.append(1)
            response_logprobs.append(0.0)

        reward_score = 1.0 if success else 0.0

        return AgentLoopOutput(
            prompt_ids=final_prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            reward_score=reward_score,
            num_turns=num_turns,
            metrics=AgentLoopMetrics(
                generate_sequences=float(metrics.get("generate_sequences", 0.0)),
                tool_calls=0.0,
                num_preempted=-1,
            ),
            extra_fields={
                "success": success,
                "num_turns": num_turns,
                "game_relative_path": game_relative_path,
                "global_steps": max_global_steps if max_global_steps is not None else 0,
                "min_global_steps": min_global_steps if min_global_steps is not None else 0,
                "max_global_steps": max_global_steps if max_global_steps is not None else 0,
                "reward_extra_info": {
                    "success": success,
                    "num_steps": num_turns,
                    "task_id": str(task_id or ""),
                    "split": str(extra_info.get("split", "")),
                    "task_family": str(extra_info.get("task_family", "")),
                },
            },
        )
