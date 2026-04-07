"""
graders/email_grader.py
Grade email_communication episodes.

score breakdown:
  0.35  correct tool (send_email)
  0.35  'to' field present
  0.30  'subject' field present
"""
from __future__ import annotations
from typing import Dict, List

from graders import _norm, name_match, arg_present

EXPECTED_TOOL = "send_email"


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
        structural += 0.35
    args = first.get("arguments", {})
    if arg_present(args, "to"):
        structural += 0.35
    if arg_present(args, "subject"):
        structural += 0.30

    final = 0.60 * structural + 0.40 * base_score
    return round(min(1.0, max(0.0, final)), 4)


if __name__ == "__main__":
    good = {"cumulative_reward": 1.0, "steps": 1, "agent_actions": [
        {"name": "send_email", "arguments": {"to": "a@b.com", "subject": "hi", "body": "hello"}}]}
    bad  = {"cumulative_reward": -0.25, "steps": 1, "agent_actions": [
        {"name": "make_call", "arguments": {"phone_number": "1234"}}]}
    print(f"good → {grade(good):.4f}")
    print(f"bad  → {grade(bad):.4f}")
    assert grade(good) >= 0.9
    assert grade(bad)  <= 0.2
    print("email_grader OK")
