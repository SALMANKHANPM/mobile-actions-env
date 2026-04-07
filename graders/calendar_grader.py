"""
graders/calendar_grader.py
Grade calendar_scheduling episodes.

score breakdown:
  0.40  correct tool name (create_calendar_event)
  0.35  title field present & non-empty
  0.25  datetime field present & ISO-8601-like
  ──────────────────────────────────────────────
  1.00  max
"""
from __future__ import annotations
import re
from typing import Dict, List

from graders import _norm, name_match, arg_present

ISO_RE = re.compile(r"\d{4}-\d{2}-\d{2}(T\d{2}:\d{2})?")

EXPECTED_TOOL = "create_calendar_event"


def grade(episode_result: Dict) -> float:
    """
    episode_result keys expected:
      cumulative_reward : float
      steps             : int
      reasons           : list[str]
      rewards           : list[float]
      task              : str
    Plus optional:
      agent_actions     : list[dict]  ({"name": str, "arguments": dict})
    """
    # ── primary signal: reward-based score ────────────────────────────────
    cum_r = episode_result.get("cumulative_reward", 0.0)
    steps = max(episode_result.get("steps", 1), 1)

    # normalise to [0, 1]
    base_score = _norm(cum_r / steps, min_r=-0.5, max_r=1.0)

    # ── structural checks on agent_actions (if provided) ──────────────────
    actions: List[Dict] = episode_result.get("agent_actions", [])
    if not actions:
        # fall back to reward only
        return round(base_score, 4)

    first = actions[0]
    structural = 0.0

    if name_match(first.get("name", ""), EXPECTED_TOOL):
        structural += 0.40
    args = first.get("arguments", {})
    if arg_present(args, "title"):
        structural += 0.35
    dt = args.get("datetime") or ""
    if ISO_RE.search(str(dt)):
        structural += 0.25

    # blend: 60% structural, 40% reward-based
    final = 0.60 * structural + 0.40 * base_score
    return round(min(0.999, max(0.001, final)), 4)


if __name__ == "__main__":
    # quick smoke test
    result_good = {
        "cumulative_reward": 1.0, "steps": 1, "reasons": ["exact_match"],
        "rewards": [1.0], "task": "calendar_scheduling",
        "agent_actions": [{"name": "create_calendar_event",
                           "arguments": {"title": "Meeting", "datetime": "2024-08-20T14:00:00"}}],
    }
    result_bad = {
        "cumulative_reward": -0.25, "steps": 1, "reasons": ["hallucinated_tool"],
        "rewards": [-0.25], "task": "calendar_scheduling",
        "agent_actions": [{"name": "web_search", "arguments": {"query": "meeting"}}],
    }
    print(f"good → {grade(result_good):.4f}  (expected ~1.0)")
    print(f"bad  → {grade(result_bad):.4f}   (expected ~0.0)")
    assert grade(result_good) >= 0.9
    assert grade(result_bad)  <= 0.2
    print("calendar_grader OK")
