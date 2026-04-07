"""
graders/maps_grader.py
Grade map_navigation episodes.

score breakdown:
  0.50  correct tool name (show_map)
  0.50  query arg present & non-empty
  ──────────────────────────────────
  1.00  max
"""
from __future__ import annotations
from typing import Dict, List

from graders import _norm, name_match, arg_present

EXPECTED_TOOL = "show_map"


def grade(episode_result: Dict) -> float:
    cum_r = episode_result.get("cumulative_reward", 0.0)
    steps = max(episode_result.get("steps", 1), 1)
    base_score = _norm(cum_r / steps, min_r=-0.5, max_r=1.0)

    actions: List[Dict] = episode_result.get("agent_actions", [])
    if not actions:
        return round(base_score, 4)

    first = actions[0]
    structural = 0.0

    if name_match(first.get("name", ""), EXPECTED_TOOL):
        structural += 0.50
    if arg_present(first.get("arguments", {}), "query"):
        structural += 0.50

    final = 0.60 * structural + 0.40 * base_score
    return round(min(1.0, max(0.0, final)), 4)


if __name__ == "__main__":
    good = {"cumulative_reward": 1.0, "steps": 1, "agent_actions": [
        {"name": "show_map", "arguments": {"query": "coffee shop near me"}}]}
    bad  = {"cumulative_reward": 0.0, "steps": 1, "agent_actions": [
        {"name": "send_email", "arguments": {"to": "a@b.com", "subject": "x"}}]}
    print(f"good → {grade(good):.4f}")
    print(f"bad  → {grade(bad):.4f}")
    assert grade(good) >= 0.9
    assert grade(bad)  <= 0.3
    print("maps_grader OK")
