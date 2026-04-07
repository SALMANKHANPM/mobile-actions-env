"""
graders/media_grader.py
Grade media_control episodes (play_music or set_alarm).
"""
from __future__ import annotations
from typing import Dict, List

from graders import _norm, name_match, arg_present

VALID_TOOLS = {"play_music", "set_alarm"}


def grade(episode_result: Dict) -> float:
    cum_r = episode_result.get("cumulative_reward", 0.0)
    steps = max(episode_result.get("steps", 1), 1)
    base_score = _norm(cum_r / steps, min_r=-0.5, max_r=1.0)

    actions: List[Dict] = episode_result.get("agent_actions", [])
    if not actions:
        return round(base_score, 4)

    first = actions[0]
    pred_name = first.get("name", "").strip().lower()
    structural = 0.0

    if pred_name in VALID_TOOLS:
        structural += 0.50
    args = first.get("arguments", {})
    # play_music needs 'query'; set_alarm needs 'time'
    if pred_name == "play_music" and arg_present(args, "query"):
        structural += 0.50
    elif pred_name == "set_alarm" and arg_present(args, "time"):
        structural += 0.50

    final = 0.60 * structural + 0.40 * base_score
    return round(min(0.999, max(0.001, final)), 4)


if __name__ == "__main__":
    good = {"cumulative_reward": 1.0, "steps": 1, "agent_actions": [
        {"name": "play_music", "arguments": {"query": "lo-fi beats"}}]}
    good2 = {"cumulative_reward": 1.0, "steps": 1, "agent_actions": [
        {"name": "set_alarm", "arguments": {"time": "06:30", "label": "wake up"}}]}
    bad = {"cumulative_reward": 0.0, "steps": 1, "agent_actions": [
        {"name": "send_email", "arguments": {"to": "x@y.com", "subject": "z"}}]}
    print(f"play_music → {grade(good):.4f}")
    print(f"set_alarm  → {grade(good2):.4f}")
    print(f"bad        → {grade(bad):.4f}")
    assert grade(good)  >= 0.9
    assert grade(good2) >= 0.9
    assert grade(bad)   <= 0.3
    print("media_grader OK")
