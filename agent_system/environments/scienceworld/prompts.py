from __future__ import annotations

from collections.abc import Iterable


SCIENCEWORLD_TEMPLATE_SAFE_SYSTEM_PROMPT = (
    "You are an agent in ScienceWorld. Complete the science task by emitting one "
    "short environment command each turn. Output only that command, with no "
    "numbering, reasoning, JSON, or prose. A deterministic adapter executes an "
    "exact official action or a guarded same-template match. Common forms "
    "include: look around; open OBJECT; open door to ROOM; pick up OBJECT; move "
    "OBJECT to CONTAINER; focus on OBJECT; look at OBJECT; look in CONTAINER; use "
    "TOOL on OBJECT; activate OBJECT; deactivate OBJECT; mix CONTAINER; pour "
    "OBJECT into CONTAINER; connect OBJECT to OBJECT; disconnect OBJECT; wait."
)

SCIENCEWORLD_DIRECT_SYSTEM_PROMPT = (
    "You are an agent in ScienceWorld. Complete the science task by emitting one "
    "short environment command each turn. Output only that command, with no "
    "numbering, reasoning, JSON, or prose. Your command is sent directly to "
    "ScienceWorld; invalid commands receive the environment's own error feedback. "
    "Common forms include: look around; open OBJECT; open door to ROOM; pick up "
    "OBJECT; move OBJECT to CONTAINER; focus on OBJECT; look at OBJECT; look in "
    "CONTAINER; use TOOL on OBJECT; activate OBJECT; deactivate OBJECT; mix "
    "CONTAINER; pour OBJECT into CONTAINER; connect OBJECT to OBJECT; disconnect "
    "OBJECT; wait."
)


def build_scienceworld_messages(
    *,
    task_description: str,
    observation: str,
    valid_actions: Iterable[str],
    action_adapter_mode: str = "template_safe",
) -> list[dict[str, str]]:
    valid_action_count = sum(
        1 for action in valid_actions if str(action).strip()
    )
    if valid_action_count <= 0:
        raise ValueError("ScienceWorld valid actions must be non-empty")
    system_prompt = (
        SCIENCEWORLD_DIRECT_SYSTEM_PROMPT
        if action_adapter_mode == "direct"
        else SCIENCEWORLD_TEMPLATE_SAFE_SYSTEM_PROMPT
    )
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Task:\n{task_description}\n\n"
                f"Observation:\n{observation}"
            ),
        },
    ]


def build_scienceworld_observation(
    *,
    observation: str,
    executed_action: str,
    valid_actions: Iterable[str],
) -> str:
    valid_action_count = sum(
        1 for action in valid_actions if str(action).strip()
    )
    if valid_action_count <= 0:
        raise ValueError("ScienceWorld valid actions must be non-empty")
    return (
        f"Executed action: {executed_action}\n\n"
        f"Observation:\n{observation}"
    )
