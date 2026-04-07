"""
graders/multiturn_grader.py
Grade multi_turn episodes (2 sequential tool calls).

Score:
  Turn 1 correct         0.50
  Turn 2 correct         0.50
  Both in correct order  +0.0 (already captured above)
  ──────────────────────────────
  max                    1.00
"""
from __future__ import annotations
from typing import Dict, List

from graders import _norm, name_match, arg_present

# Expected sequence for the canonical multi-turn episode
EXPECTED_SEQUENCE = [
    {"name": "create_calendar_event", "required_args": ["title", "datetime"]},
    {"name": "send_email",            "required_args": ["to", "subject"]},
]


def _score_action(action: Dict, expected: Dict) -> float:
    score = 0.0
    if name_match(action.get("name", ""), expected["name"]):
        score += 0.5
    args = action.get("arguments", {})
    required = expected.get("required_args", [])
    if required:
        present = sum(1 for k in required if arg_present(args, k))
        score += 0.5 * (present / len(required))
    return score


def grade(episode_result: Dict) -> float:
    cum_r = episode_result.get("cumulative_reward", 0.0)
    base_score = _norm(cum_r / 2, min_r=-0.5, max_r=1.0)   # 2 GT calls max

    actions: List[Dict] = episode_result.get("agent_actions", [])
    if not actions:
        return round(base_score, 4)

    structural = 0.0
    for i, expected in enumerate(EXPECTED_SEQUENCE):
        if i < len(actions):
            structural += _score_action(actions[i], expected) * 0.5  # each turn = 0.5 weight

    final = 0.60 * structural + 0.40 * base_score
    return round(min(1.0, max(0.0, final)), 4)


if __name__ == "__main__":
    good = {
        "cumulative_reward": 2.0, "steps": 2,
        "agent_actions": [
            {"name": "create_calendar_event", "arguments": {"title": "Dentist", "datetime": "2024-08-23T10:00:00"}},
            {"name": "send_email",            "arguments": {"to": "clinic@dr.com", "subject": "Appointment"}},
        ],
    }
    partial = {
        "cumulative_reward": 1.0, "steps": 2,
        "agent_actions": [
            {"name": "create_calendar_event", "arguments": {"title": "Dentist", "datetime": "2024-08-23T10:00:00"}},
            {"name": "make_call", "arguments": {"phone_number": "1234"}},
        ],
    }
    print(f"good    → {grade(good):.4f}   (expected ~1.0)")
    print(f"partial → {grade(partial):.4f} (expected ~0.5-0.7)")
    assert grade(good)    >= 0.9
    assert grade(partial) >= 0.4
    print("multiturn_grader OK")
