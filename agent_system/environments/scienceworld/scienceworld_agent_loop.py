from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from typing import Any
from uuid import uuid4

from agent_system.environments.scienceworld.action_aligner import align_scienceworld_action
from agent_system.environments.scienceworld.envs import ScienceWorldEnvPool, ScienceWorldEpisodeSpec
from agent_system.environments.scienceworld.prompts import (
    build_scienceworld_messages,
    build_scienceworld_observation,
)

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_TOOL_CALL_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
PARSER_REJECTION_MARKER = "No known action matches that input."
_SHARED_POOLS: dict[tuple[int, int, int], ScienceWorldEnvPool] = {}


def build_scienceworld_deterministic_identity(
    *,
    task_name: str,
    variation_id: int,
    split: str,
    global_steps: int,
    session_id: int,
    validate: bool,
    base_seed: int,
) -> tuple[str, int]:
    payload = json.dumps(
        {
            "task_name": str(task_name),
            "variation_id": int(variation_id),
            "split": str(split),
            "global_steps": int(global_steps),
            "session_id": int(session_id),
            "validate": bool(validate),
            "base_seed": int(base_seed),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest = hashlib.blake2b(payload, digest_size=16).digest()
    request_id = f"scienceworld-{digest.hex()}"
    seed = int.from_bytes(digest[:8], "big") % 2_147_483_647
    return request_id, seed


def recover_scienceworld_command(text: str) -> str | None:
    for raw in _TOOL_CALL_BLOCK.findall(text or ""):
        try:
            payload = json.loads(raw.strip())
            if payload.get("name") != "env_step":
                continue
            arguments = payload.get("arguments")
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                continue
            command = str(arguments.get("command", "")).strip()
            if command:
                return command
        except Exception:
            continue
    return None


_ACTION_ADAPTER_MODES = ("template_safe", "direct")


def resolve_action_adapter_mode(configured_mode: str | None) -> str:
    mode = os.getenv("SCIENCEWORLD_ACTION_ADAPTER_MODE") or configured_mode
    if mode is None:
        raise ValueError(
            "ScienceWorld action_adapter_mode is unset; pin it in the agent config"
        )
    mode = str(mode).strip()
    if mode not in _ACTION_ADAPTER_MODES:
        raise ValueError(
            "ScienceWorld action_adapter_mode must be template_safe or direct, "
            f"got {mode!r}"
        )
    logger.info(f"ScienceWorld action protocol resolved to {mode}")
    return mode


def _shared_pool(*, size: int, env_step_limit: int) -> ScienceWorldEnvPool:
    loop = asyncio.get_running_loop()
    key = (id(loop), int(size), int(env_step_limit))
    pool = _SHARED_POOLS.get(key)
    if pool is None:
        pool = ScienceWorldEnvPool(
            size=size,
            env_step_limit=env_step_limit,
        )
        _SHARED_POOLS[key] = pool
    return pool


@register("scienceworld_agent")
class ScienceWorldAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        env_pool = kwargs.pop("env_pool", None)
        super().__init__(*args, **kwargs)
        self.prompt_length = int(self.rollout_config.prompt_length)
        self.response_length = int(self.rollout_config.response_length)
        self.max_steps = int(kwargs.get("max_steps", 100))
        self.env_step_limit = int(kwargs.get("env_step_limit", 100))
        self.env_pool_size = int(kwargs.get("env_pool_size", 1))
        self.max_generation_tokens = int(kwargs.get("max_generation_tokens", 128))
        self.max_action_candidates = int(
            kwargs.get("max_action_candidates", 1)
        )
        self.action_adapter_mode = resolve_action_adapter_mode(
            kwargs.get("action_adapter_mode")
        )
        self._env_pool = env_pool

    def _pool(self) -> ScienceWorldEnvPool:
        return self._env_pool or _shared_pool(
            size=self.env_pool_size,
            env_step_limit=self.env_step_limit,
        )

    @rollout_trace_op
    async def run(
        self,
        sampling_params: dict[str, Any],
        **kwargs,
    ) -> AgentLoopOutput:
        extra_info = kwargs.get("extra_info") or {}
        if isinstance(extra_info, str):
            try:
                extra_info = json.loads(extra_info)
            except json.JSONDecodeError:
                extra_info = {}
        task_name = str(extra_info.get("task_name", "")).strip()
        if not task_name:
            raise ValueError("ScienceWorld sample is missing extra_info.task_name")
        variation_id = int(extra_info.get("variation_id", -1))
        if variation_id < 0:
            raise ValueError("ScienceWorld sample has invalid variation_id")
        spec = ScienceWorldEpisodeSpec(
            task_name=task_name,
            variation_id=variation_id,
            simplification=str(extra_info.get("simplification", "")),
        )

        sp = dict(sampling_params)
        if self.max_generation_tokens > 0:
            sp["max_tokens"] = min(
                int(sp.get("max_tokens", self.max_generation_tokens)),
                self.max_generation_tokens,
            )
        metrics: dict[str, Any] = {}
        min_global_steps: int | None = None
        max_global_steps: int | None = None
        validate = (
            bool(kwargs.get("validate", False))
            or str(extra_info.get("split", "")) != "train"
        )
        configured_deterministic_rollout = os.getenv(
            "SCIENCEWORLD_DETERMINISTIC_ROLLOUT",
            "0",
        ).strip().lower() in {"1", "true", "yes", "on"}
        deterministic_rollout = (
            validate or configured_deterministic_rollout
        )
        deterministic_seed: int | None = None
        if deterministic_rollout:
            sampling_seed = sp.get("seed")
            base_seed = int(
                os.getenv(
                    "SCIENCEWORLD_DETERMINISTIC_BASE_SEED",
                    str(0 if sampling_seed is None else sampling_seed),
                )
            )
            request_id, deterministic_seed = (
                build_scienceworld_deterministic_identity(
                    task_name=task_name,
                    variation_id=variation_id,
                    split=str(extra_info.get("split", "")),
                    global_steps=(
                        0 if validate else int(kwargs.get("global_steps", -1))
                    ),
                    session_id=(
                        0 if validate else int(kwargs.get("session_id", 0))
                    ),
                    validate=validate,
                    base_seed=base_seed,
                )
            )
        else:
            request_id = str(extra_info.get("id") or uuid4().hex)
        metrics["deterministic_rollout"] = float(deterministic_rollout)
        all_ids: list[int] = []
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        num_turns = 0
        parser_rejected_steps = 0
        exact_action_matches = 0
        aligned_action_steps = 0
        template_fallback_steps = 0
        safe_fallback_steps = 0
        direct_action_steps = 0
        invalid_empty_steps = 0
        selected_candidate_rank_sum = 0
        selected_candidate_rank_count = 0
        proposed_candidate_count_sum = 0
        token_f1_sum = 0.0
        char_similarity_sum = 0.0
        context_truncated = False

        async with self._pool().lease() as environment:
            current = await asyncio.to_thread(environment.start_episode, spec)
            messages = build_scienceworld_messages(
                task_description=current.task_description or task_name,
                observation=current.observation,
                valid_actions=current.valid_actions,
                action_adapter_mode=self.action_adapter_mode,
            )
            prompt_ids = await self.apply_chat_template(
                messages,
            )
            if len(prompt_ids) > self.prompt_length:
                raise ValueError(
                    f"ScienceWorld prompt length {len(prompt_ids)} exceeds "
                    f"configured limit {self.prompt_length}"
                )
            all_ids = list(prompt_ids)

            while num_turns < self.max_steps and not current.done:
                if len(response_mask) >= self.response_length:
                    context_truncated = True
                    break
                generation_params = sp
                if deterministic_seed is not None:
                    generation_params = dict(sp)
                    generation_params["seed"] = (
                        deterministic_seed + num_turns
                    ) % 2_147_483_647
                with simple_timer("generate_sequences", metrics):
                    output = await self.server_manager.generate(
                        request_id=request_id,
                        prompt_ids=all_ids,
                        sampling_params=generation_params,
                    )
                generated_ids = list(output.token_ids)
                generated_logprobs = (
                    list(output.log_probs)
                    if output.log_probs
                    else [0.0] * len(generated_ids)
                )
                remaining_response = self.response_length - len(response_mask)
                generation_was_truncated = len(generated_ids) > remaining_response
                if generation_was_truncated:
                    generated_ids = generated_ids[:remaining_response]
                    generated_logprobs = generated_logprobs[:remaining_response]
                all_ids.extend(generated_ids)
                response_mask.extend([1] * len(generated_ids))
                response_logprobs.extend(generated_logprobs)
                num_turns += 1
                if generation_was_truncated:
                    context_truncated = True
                    break

                global_step = (
                    getattr(output, "extra_fields", None) or {}
                ).get("global_steps")
                if global_step is not None:
                    min_global_steps = (
                        global_step
                        if min_global_steps is None
                        else min(min_global_steps, global_step)
                    )
                    max_global_steps = (
                        global_step
                        if max_global_steps is None
                        else max(max_global_steps, global_step)
                    )

                generated_text = self.tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                ).strip()
                command = (
                    recover_scienceworld_command(generated_text)
                    or generated_text
                    or None
                )
                if self.action_adapter_mode == "direct":
                    direct_action_steps += 1
                    executed_action = (command or "").strip()
                    is_exact = executed_action in current.valid_actions
                    exact_action_matches += int(is_exact)
                    proposed_candidate_count_sum += int(bool(executed_action))
                    if executed_action:
                        selected_candidate_rank_count += 1
                        token_f1_sum += float(is_exact)
                        char_similarity_sum += float(is_exact)
                        current = await asyncio.to_thread(
                            environment.step,
                            executed_action,
                        )
                        if PARSER_REJECTION_MARKER in current.observation:
                            parser_rejected_steps += 1
                        observation_text = current.observation
                    else:
                        invalid_empty_steps += 1
                        observation_text = (
                            "Invalid Action.\n\n" + current.observation
                        )
                        executed_action = "<empty>"
                    tool_text = build_scienceworld_observation(
                        observation=observation_text,
                        executed_action=executed_action,
                        valid_actions=current.valid_actions,
                    )
                else:
                    alignment = align_scienceworld_action(
                        command or "",
                        current.valid_actions,
                        max_candidates=self.max_action_candidates,
                    )
                    exact_action_matches += int(alignment.exact_match)
                    aligned_action_steps += int(not alignment.exact_match)
                    template_fallback_steps += int(
                        alignment.match_mode == "template_fallback"
                    )
                    safe_fallback_steps += int(
                        alignment.match_mode == "safe_fallback"
                    )
                    proposed_candidate_count_sum += len(alignment.candidates)
                    if alignment.selected_rank is not None:
                        selected_candidate_rank_sum += alignment.selected_rank
                        selected_candidate_rank_count += 1
                    token_f1_sum += alignment.token_f1
                    char_similarity_sum += alignment.char_similarity
                    current = await asyncio.to_thread(
                        environment.step,
                        alignment.executed_action,
                    )
                    if PARSER_REJECTION_MARKER in current.observation:
                        parser_rejected_steps += 1
                    tool_text = build_scienceworld_observation(
                        observation=current.observation,
                        executed_action=alignment.executed_action,
                        valid_actions=current.valid_actions,
                    )

                observation_ids = await self.apply_chat_template(
                    [{"role": "user", "content": tool_text}],
                    remove_system_prompt=True,
                )
                observation_ids = self.turn_separator + observation_ids
                if (
                    len(response_mask) + len(observation_ids)
                    > self.response_length
                ):
                    context_truncated = True
                    break
                all_ids.extend(observation_ids)
                response_mask.extend([0] * len(observation_ids))
                response_logprobs.extend([0.0] * len(observation_ids))

            raw_final_score = float(current.score)
            final_score = max(0.0, raw_final_score)
            episode_terminated = bool(current.done)

        if not any(response_mask):
            pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
            all_ids.append(pad_id)
            response_mask.append(1)
            response_logprobs.append(0.0)

        response_ids = all_ids[-len(response_mask) :]
        final_prompt_ids = all_ids[: len(all_ids) - len(response_mask)]
        data_source = str(kwargs.get("data_source", f"scienceworld/{task_name}"))
        success = final_score >= 100.0
        return AgentLoopOutput(
            prompt_ids=final_prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            reward_score=final_score / 100.0,
            num_turns=num_turns,
            metrics=AgentLoopMetrics(
                generate_sequences=float(metrics.get("generate_sequences", 0.0)),
                tool_calls=0.0,
                num_preempted=-1,
            ),
            extra_fields={
                "global_steps": max_global_steps or 0,
                "min_global_steps": min_global_steps or 0,
                "max_global_steps": max_global_steps or 0,
                "reward_extra_info": {
                    "raw_final_score": raw_final_score,
                    "final_score": final_score,
                    "success": success,
                    "completed": success,
                    "terminated": episode_terminated,
                    "num_steps": num_turns,
                    "task_name": task_name,
                    "variation_id": variation_id,
                    "split": str(extra_info.get("split", "")),
                    "data_source": data_source,
                    "action_adapter_mode": self.action_adapter_mode,
                    "direct_action_steps": direct_action_steps,
                    "invalid_empty_steps": invalid_empty_steps,
                    "parser_rejected_steps": parser_rejected_steps,
                    "parser_rejected_rate": (
                        parser_rejected_steps / num_turns if num_turns else 0.0
                    ),
                    "exact_action_matches": exact_action_matches,
                    "exact_action_match_rate": (
                        exact_action_matches / num_turns if num_turns else 0.0
                    ),
                    "aligned_action_steps": aligned_action_steps,
                    "template_fallback_steps": template_fallback_steps,
                    "safe_fallback_steps": safe_fallback_steps,
                    "proposed_candidate_count_mean": (
                        proposed_candidate_count_sum / num_turns
                        if num_turns
                        else 0.0
                    ),
                    "selected_candidate_rank_mean": (
                        selected_candidate_rank_sum
                        / selected_candidate_rank_count
                        if selected_candidate_rank_count
                        else -1.0
                    ),
                    "alignment_token_f1_mean": (
                        token_f1_sum / num_turns if num_turns else 0.0
                    ),
                    "alignment_char_similarity_mean": (
                        char_similarity_sum / num_turns if num_turns else 0.0
                    ),
                    "context_truncated": context_truncated,
                },
            },
        )


__all__ = ["ScienceWorldAgentLoop", "recover_scienceworld_command"]
