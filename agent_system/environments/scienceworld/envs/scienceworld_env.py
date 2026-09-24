from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScienceWorldEpisodeSpec:
    task_name: str
    variation_id: int
    simplification: str = ""

    def __post_init__(self) -> None:
        if not str(self.task_name).strip():
            raise ValueError("task_name must be non-empty")
        if int(self.variation_id) < 0:
            raise ValueError("variation_id must be non-negative")


@dataclass(frozen=True)
class ScienceWorldTransition:
    observation: str
    reward_delta: float
    score: float
    done: bool
    valid_actions: list[str]
    task_description: str
    task_name: str
    variation_id: int
    moves: int

    @property
    def normalized_score(self) -> float:
        return self.score / 100.0


EnvFactory = Callable[[int], Any]


def _default_env_factory(env_step_limit: int):
    try:
        from scienceworld import ScienceWorldEnv
    except ImportError as exc:
        raise ImportError(
            "ScienceWorld is optional. Install scienceworld==1.2.3 and provide Java 21."
        ) from exc
    return ScienceWorldEnv(envStepLimit=env_step_limit)


def _normalize_valid_actions(value: Any, fallback: Iterable[str]) -> list[str]:
    source = value if isinstance(value, (list, tuple)) else fallback
    actions = []
    seen = set()
    for item in source:
        action = str(item).strip()
        if not action or action in seen:
            continue
        seen.add(action)
        actions.append(action)
    if not actions:
        raise RuntimeError("ScienceWorld returned no valid actions")
    return actions


class ScienceWorldEnvAdapter:
    def __init__(
        self,
        *,
        env_step_limit: int = 100,
        env_factory: EnvFactory | None = None,
    ) -> None:
        if int(env_step_limit) <= 0:
            raise ValueError("env_step_limit must be positive")
        self.env_step_limit = int(env_step_limit)
        self._env = (env_factory or _default_env_factory)(self.env_step_limit)
        self._closed = False
        self._episode: ScienceWorldEpisodeSpec | None = None

    @property
    def raw_env(self):
        return self._env

    @property
    def gateway_port(self) -> int | None:
        gateway = getattr(self._env, "_gateway", None)
        client = getattr(gateway, "_gateway_client", None)
        port = getattr(client, "port", None)
        return int(port) if port is not None else None

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("ScienceWorld environment is closed")

    def _fallback_valid_actions(self) -> list[str]:
        getter = getattr(self._env, "get_valid_action_object_combinations", None)
        if getter is None:
            return []
        return list(getter())

    @staticmethod
    def _validated_score(value: Any) -> float:
        score = float(value)
        if not -100.0 <= score <= 100.0:
            raise ValueError(f"ScienceWorld score must be in [-100, 100], got {score}")
        return score

    def _transition(
        self,
        *,
        observation: Any,
        reward_delta: Any,
        done: Any,
        info: dict[str, Any] | None,
    ) -> ScienceWorldTransition:
        if self._episode is None:
            raise RuntimeError("No ScienceWorld episode is loaded")
        info = dict(info or {})
        score = self._validated_score(info.get("score", 0.0))
        valid_actions = _normalize_valid_actions(
            info.get("valid"),
            self._fallback_valid_actions(),
        )
        return ScienceWorldTransition(
            observation=str(observation),
            reward_delta=float(reward_delta),
            score=score,
            done=bool(done),
            valid_actions=valid_actions,
            task_description=str(info.get("taskDesc", "")).strip(),
            task_name=str(info.get("taskName", self._episode.task_name)),
            variation_id=int(info.get("variationIdx", self._episode.variation_id)),
            moves=int(info.get("moves", 0)),
        )

    def start_episode(self, spec: ScienceWorldEpisodeSpec) -> ScienceWorldTransition:
        self._ensure_open()
        self._env.load(
            spec.task_name,
            variationIdx=int(spec.variation_id),
            simplificationStr=str(spec.simplification),
        )
        self._episode = spec
        reset_output = self._env.reset()
        if not isinstance(reset_output, tuple) or len(reset_output) != 2:
            raise RuntimeError(f"Unsupported ScienceWorld reset output: {reset_output!r}")
        observation, info = reset_output
        return self._transition(
            observation=observation,
            reward_delta=0.0,
            done=False,
            info=info,
        )

    def step(self, action: str) -> ScienceWorldTransition:
        self._ensure_open()
        action = str(action).strip()
        if not action:
            raise ValueError("ScienceWorld action must be non-empty")
        step_output = self._env.step(action)
        if not isinstance(step_output, tuple) or len(step_output) != 4:
            raise RuntimeError(f"Unsupported ScienceWorld step output: {step_output!r}")
        observation, reward_delta, done, info = step_output
        return self._transition(
            observation=observation,
            reward_delta=reward_delta,
            done=done,
            info=info,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self._env, "close", None)
        if close is not None:
            close()
