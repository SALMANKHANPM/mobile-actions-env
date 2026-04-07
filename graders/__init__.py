"""
graders/__init__.py
Shared grading utilities for all task graders.
Each grader exposes:
    grade(episode_result: dict) -> float   in [0.0, 1.0]
"""
from __future__ import annotations
from typing import Dict, List


def _norm(val, min_r=-0.5, max_r=1.0):
    """Normalise val to strictly (0, 1) — never 0.0 or 1.0 exactly."""
    raw = (val - min_r) / (max_r - min_r)
    return round(max(0.001, min(0.999, raw)), 4)


def name_match(pred: str, gt: str) -> bool:
    return pred.strip().lower() == gt.strip().lower()


def arg_present(args: dict, key: str) -> bool:
    return bool(args.get(key) not in (None, "", []))


def soft_arg_match(pred_val, gt_val) -> float:
    """Returns 1.0 exact, 0.5 partial string overlap, 0.0 no match."""
    if pred_val is None or gt_val is None:
        return 0.0
    p, g = str(pred_val).lower().strip(), str(gt_val).lower().strip()
    if p == g:
        return 1.0
    # substring partial credit
    if p in g or g in p:
        return 0.5
    return 0.0
