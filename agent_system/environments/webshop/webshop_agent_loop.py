from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Optional
from uuid import uuid4

from agent_system.environments.webshop.envs.client import WebShopEnvClient
from agent_system.environments.webshop.prompts import WEBSHOP_TOOL_SCHEMAS
from agent_system.environments.webshop.utils import (
    build_env_error_observation,
    build_invalid_tool_call_observation,
    build_webshop_messages,
    format_available_actions,
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

_MISSING = object()


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
    if len(tool_calls) > 1:
        return None, f"expected exactly one env_step tool call, got {len(tool_calls)}"
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


@register("webshop_agent")
class WebshopAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.tool_parser = ToolParser.get_tool_parser(
            self.rollout_config.multi_turn.format,
            self.tokenizer,
        )
        self.tool_schemas = WEBSHOP_TOOL_SCHEMAS
        self.max_steps = int(kwargs.get("max_steps", 15))
        self.env_timeout = float(kwargs.get("env_timeout", 30.0))
        self.invalid_penalty = float(kwargs.get("invalid_tool_call_penalty", 0.05))
        self.reset_retries = int(kwargs.get("reset_retries", 3))
        self.step_retries = int(kwargs.get("step_retries", 2))
        self.retry_backoff = float(kwargs.get("retry_backoff", 0.5))
        self.client = WebShopEnvClient(timeout=self.env_timeout)

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        extra_info = kwargs.get("extra_info") or {}
        if isinstance(extra_info, str):
            try:
                extra_info = json.loads(extra_info)
            except json.JSONDecodeError:
                extra_info = {}
        goal_index = extra_info.get("goal_index")
        if goal_index is None:
            raise ValueError("WebShop sample is missing extra_info.goal_index")
        goal_index = int(goal_index)
        instruction = str(extra_info.get("instruction") or "")
        split = extra_info.get("split", "")
        asin = extra_info.get("asin")

        metrics: dict[str, Any] = {}

        reset_payload = None
        reset_err: Optional[Exception] = None
        for attempt in range(self.reset_retries):
            try:
                reset_payload = await self.client.reset(goal_index)
                break
            except Exception as e:
                reset_err = e
                if attempt + 1 < self.reset_retries:
                    await asyncio.sleep(self.retry_backoff * (attempt + 1))
        if reset_payload is None:
            raise RuntimeError(
                f"WebShop env reset failed after {self.reset_retries} attempts "
                f"(goal_index={goal_index}): {reset_err}"
            )
        obs = str(reset_payload["observation"])
        env_state = reset_payload["env_state"]
        reset_info = reset_payload.get("info") or {}
        available_actions = reset_info.get("available_actions")
        available_actions = available_actions if isinstance(available_actions, list) else []
        if not instruction:
            instruction = str(reset_info.get("instruction") or "")

        init_messages = build_webshop_messages(
            instruction=instruction,
            observation=obs,
            available_actions=available_actions,
        )
        prompt_ids = await self.apply_chat_template(init_messages, tools=self.tool_schemas)

        all_ids: list[int] = list(prompt_ids)
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        request_id = str(goal_index) if goal_index is not None else uuid4().hex
        num_turns = 0
        num_invalid = 0
        num_env_fail = 0
        num_no_op = 0
        length_exceeded = False
        done = False
        success = False
        final_env_reward = 0.0
        final_task_score = 0.0
        prev_obs = obs
        min_global_steps: int | None = None
        max_global_steps: int | None = None

        sp = dict(sampling_params)
        if self.tool_parser.stop_token_ids:
            sp["stop_token_ids"] = list(set((sp.get("stop_token_ids") or []) + self.tool_parser.stop_token_ids))

        while num_turns < self.max_steps and not done:
            if len(response_mask) >= self.response_length:
                length_exceeded = True
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
                result = None
                env_err: Optional[Exception] = None
                for attempt in range(self.step_retries):
                    try:
                        result = await self.client.step(goal_index, env_state, command)
                        env_err = None
                        break
                    except Exception as e:
                        env_err = e
                        if attempt + 1 < self.step_retries:
                            await asyncio.sleep(self.retry_backoff * (attempt + 1))
                if result is not None:
                    new_obs = str(result["observation"])
                    env_state = result["env_state"]
                    env_reward = float(result["reward"])
                    done = bool(result["done"])
                    info = result.get("info", {}) or {}
                    if "success" in info and info["success"] is not None:
                        success = bool(info["success"])
                    elif env_reward >= 0.999:
                        success = True
                    new_adm = info.get("available_actions", _MISSING)
                    if new_adm is not _MISSING:
                        available_actions = new_adm if isinstance(new_adm, list) else []

                    if info.get("error"):
                        num_no_op += 1
                    tool_text = f"{new_obs}\n\n### Available Actions\n{format_available_actions(available_actions)}"
                    prev_obs = new_obs
                    if done:
                        final_env_reward = env_reward
                        final_task_score = float(info.get("task_score", env_reward))
                else:
                    num_env_fail += 1
                    tool_text = build_env_error_observation(f"env step failed after retries: {env_err}")
            else:
                num_invalid += 1
                tool_text = build_invalid_tool_call_observation(invalid_reason)

            tool_msg = [{"role": "tool", "content": tool_text}]
            obs_ids = await self.apply_chat_template(tool_msg, remove_system_prompt=True)
            obs_ids = self.turn_separator + obs_ids
            if len(response_mask) + len(obs_ids) >= self.response_length:
                length_exceeded = True
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

        reward_score = (float(final_task_score) if done else 0.0) - self.invalid_penalty * num_invalid

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
                "task_score": final_task_score,
                "num_turns": num_turns,
                "num_invalid": num_invalid,
                "num_env_fail": num_env_fail,
                "goal_index": goal_index,
                "global_steps": max_global_steps if max_global_steps is not None else 0,
                "min_global_steps": min_global_steps if min_global_steps is not None else 0,
                "max_global_steps": max_global_steps if max_global_steps is not None else 0,
                "reward_extra_info": {
                    "success": success,
                    "task_score": final_task_score,
                    "final_reward": final_env_reward,
                    "num_steps": num_turns,
                    "num_env_fail": num_env_fail,
                    "num_invalid": num_invalid,
                    "num_no_op": num_no_op,
                    "done": bool(done),
                    "length_exceeded": bool(length_exceeded),
                    "goal_index": goal_index,
                    "split": str(split),
                    "asin": str(asin or ""),
                },
            },
        )
