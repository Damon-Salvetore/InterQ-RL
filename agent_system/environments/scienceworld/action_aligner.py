from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_RANK_PREFIX = re.compile(
    r"^\s*(?:(?:[-*•]+)|(?:\d+\s*[\.\):\-]))\s*"
)
_STRUCTURAL_TOKENS = {
    "at",
    "from",
    "in",
    "into",
    "of",
    "on",
    "the",
    "to",
    "with",
}


@dataclass(frozen=True)
class ActionAlignment:
    proposal: str
    candidates: tuple[str, ...]
    executed_action: str
    exact_match: bool
    match_mode: str
    selected_rank: int | None
    token_f1: float
    char_similarity: float


def _normalize_text(text: str) -> str:
    tokens = [
        token
        for token in _TOKEN_PATTERN.findall(str(text).lower())
        if token != "the"
    ]
    return " ".join(tokens)


def _parse_candidates(proposal: str, max_candidates: int) -> tuple[str, ...]:
    candidates: list[str] = []
    seen: set[str] = set()
    for raw_line in str(proposal).splitlines():
        candidate = _RANK_PREFIX.sub("", raw_line).strip(" `\"'")
        normalized = _normalize_text(candidate)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        candidates.append(candidate)
        if len(candidates) >= max_candidates:
            break
    return tuple(candidates)


@dataclass(frozen=True)
class _ActionShape:
    family: str
    content: str


def _action_shape(text: str) -> _ActionShape | None:
    normalized = _normalize_text(text)
    patterns = (
        ("open_door_to", r"^open door to (.+)$"),
        ("close_door_to", r"^close door to (.+)$"),
        ("pick_up", r"^pick up (.+)$"),
        ("focus_on", r"^focus on (.+)$"),
        ("look_at", r"^look at (.+)$"),
        ("look_in", r"^look (?:in|inside) (.+)$"),
        ("pour_into", r"^pour (.+) into (.+)$"),
        ("use_on", r"^use (.+) on (.+)$"),
        ("connect_to", r"^connect (.+) to (.+)$"),
        ("dunk_into", r"^dunk (.+) into (.+)$"),
        ("move_object_to", r"^move (.+) to (.+)$"),
        ("disconnect", r"^disconnect (.+)$"),
        ("activate", r"^activate (.+)$"),
        ("deactivate", r"^deactivate (.+)$"),
        ("open", r"^open (.+)$"),
        ("close", r"^close (.+)$"),
        ("mix", r"^mix (.+)$"),
        ("read", r"^read (.+)$"),
        ("eat", r"^eat (.+)$"),
        ("flush", r"^flush (.+)$"),
    )

    navigation_match = re.match(r"^(?:go|move) to (.+)$", normalized)
    if navigation_match:
        return _ActionShape("open_door_to", navigation_match.group(1))
    if normalized == "look around":
        return _ActionShape("look_around", "")
    if normalized in {"wait", "wait1"}:
        return _ActionShape("wait", "")
    if normalized in {"inventory", "task"}:
        return _ActionShape(normalized, "")

    for family, pattern in patterns:
        match = re.match(pattern, normalized)
        if match:
            content_tokens = [
                token
                for group in match.groups()
                for token in group.split()
                if token not in _STRUCTURAL_TOKENS
            ]
            return _ActionShape(family, " ".join(content_tokens))
    return None


def _token_f1(left: str, right: str) -> float:
    left_tokens = Counter(_TOKEN_PATTERN.findall(left))
    right_tokens = Counter(_TOKEN_PATTERN.findall(right))
    left_count = sum(left_tokens.values())
    right_count = sum(right_tokens.values())
    if left_count == 0 or right_count == 0:
        return 0.0
    overlap = sum((left_tokens & right_tokens).values())
    return (2.0 * overlap) / (left_count + right_count)


def align_scienceworld_action(
    proposal: str,
    valid_actions: Iterable[str],
    max_candidates: int = 1,
) -> ActionAlignment:
    if int(max_candidates) <= 0:
        raise ValueError("max_candidates must be positive")
    candidates = {
        str(action).strip()
        for action in valid_actions
        if str(action).strip()
    }
    if not candidates:
        raise ValueError("ScienceWorld valid actions must be non-empty")

    ranked_candidates = sorted(
        ((_normalize_text(action), action) for action in candidates),
        key=lambda item: (item[0], item[1]),
    )
    proposals = _parse_candidates(proposal, int(max_candidates))

    def safe_fallback() -> ActionAlignment:
        fallback = next(
            (
                action
                for target in ("look around", "wait", "wait1")
                for normalized, action in ranked_candidates
                if normalized == target
            ),
            ranked_candidates[0][1],
        )
        return ActionAlignment(
            proposal=str(proposal).strip(),
            candidates=proposals,
            executed_action=fallback,
            exact_match=False,
            match_mode="safe_fallback",
            selected_rank=None,
            token_f1=0.0,
            char_similarity=0.0,
        )

    if not proposals:
        return safe_fallback()

    exact_actions: dict[str, str] = {}
    for normalized, action in ranked_candidates:
        exact_actions.setdefault(normalized, action)
    for rank, candidate in enumerate(proposals):
        normalized = _normalize_text(candidate)
        if normalized in exact_actions:
            return ActionAlignment(
                proposal=str(proposal).strip(),
                candidates=proposals,
                executed_action=exact_actions[normalized],
                exact_match=True,
                match_mode="exact",
                selected_rank=rank,
                token_f1=1.0,
                char_similarity=1.0,
            )

    shaped_actions = [
        (_action_shape(action), normalized, action)
        for normalized, action in ranked_candidates
    ]
    for rank, candidate in enumerate(proposals):
        proposal_shape = _action_shape(candidate)
        if proposal_shape is None or not proposal_shape.content:
            continue
        best: tuple[float, float, str] | None = None
        best_action: str | None = None
        for action_shape, normalized, action in shaped_actions:
            if (
                action_shape is None
                or action_shape.family != proposal_shape.family
                or not action_shape.content
            ):
                continue
            token_f1 = _token_f1(
                proposal_shape.content,
                action_shape.content,
            )
            if token_f1 <= 0.0:
                continue
            char_similarity = SequenceMatcher(
                None,
                proposal_shape.content,
                action_shape.content,
            ).ratio()
            score = (token_f1, char_similarity, normalized)
            if best is None or score[:2] > best[:2]:
                best = score
                best_action = action
        if best is not None and best_action is not None:
            return ActionAlignment(
                proposal=str(proposal).strip(),
                candidates=proposals,
                executed_action=best_action,
                exact_match=False,
                match_mode="template_fallback",
                selected_rank=rank,
                token_f1=best[0],
                char_similarity=best[1],
            )

    return safe_fallback()


__all__ = ["ActionAlignment", "align_scienceworld_action"]
