from __future__ import annotations

from agent_system.environments.webshop.prompts import WEBSHOP_SYSTEM_PROMPT, WEBSHOP_USER_PROMPT


def format_available_actions(actions: list[str] | None) -> str:
    if not isinstance(actions, list) or not actions:
        return "None"
    return "\n".join(f"- {action}" for action in actions)


def build_webshop_messages(
    *,
    instruction: str,
    observation: str,
    available_actions: list[str] | None,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": WEBSHOP_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": WEBSHOP_USER_PROMPT.format(
                instruction=instruction,
                observation=observation,
                available_actions=format_available_actions(available_actions),
            ),
        },
    ]


def build_invalid_tool_call_observation(reason: str) -> str:
    return (
        "Invalid tool call. You must call the `env_step` tool exactly once with JSON arguments "
        'like {"command": "search[wireless headphones]"} or {"command": "click[Buy Now]"}. '
        f"Reason: {reason}. The environment state did not change."
    )


def build_env_error_observation(reason: str) -> str:
    return (
        "The WebShop environment could not execute your last action due to a temporary server "
        f"error (this is not caused by your action): {reason}. "
        "The environment state did not change; you may retry the same action."
    )
