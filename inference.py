"""
inference.py — OpenEnv Mobile-Actions Inference Script
=======================================================
MANDATORY env vars:
  API_BASE_URL   LLM API endpoint  (default: https://router.huggingface.co/v1)
  MODEL_NAME     Model identifier  (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN       HuggingFace / API key
  OPENENV_URL    Running env server (default: http://localhost:7860)

HF TOKEN PERMISSIONS NOTE:
  When using https://router.huggingface.co/v1 (Inference Providers router) your
  token MUST have the "Make calls to Inference Providers" scope enabled.
  Go to https://huggingface.co/settings/tokens → create/edit token →
  tick "Make calls to Inference Providers" → save, then re-run.

Usage:
  python inference.py                             # all 5 tasks, 2 episodes each
  python inference.py --task calendar_scheduling --episodes 3
  python inference.py --task map_navigation --dry-run   # random-agent baseline

STDOUT FORMAT (per episode):
  [START] task=<task_name> env=mobile-actions-env model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────────
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY:      str = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
MODEL_NAME:   str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL:      str = os.getenv("OPENENV_URL", "http://localhost:7860")

BENCHMARK    = "mobile-actions-env"
MAX_STEPS    = 8
TEMPERATURE  = 0.0
MAX_TOKENS   = 512

# A score at or above this threshold counts as a successful episode
SUCCESS_SCORE_THRESHOLD = 0.5

TASKS = [
    "calendar_scheduling",
    "map_navigation",
    "email_communication",
    "media_control",
    "multi_turn",
]

FALLBACK_ACTION: Dict = {"name": "noop", "arguments": {}}

# Set to True the first time a 403 is detected so we abort early.
_AUTH_FAILED: bool = False

SYSTEM_PROMPT = textwrap.dedent("""
    You are a mobile assistant that controls a phone on behalf of the user.
    When the user makes a request, respond with EXACTLY ONE tool call.
    Do NOT add any explanations or commentary — only the tool call.
    Choose the most appropriate tool from those listed.
    Fill all required parameters; use null for optional ones you cannot infer.
""").strip()


# ── Mandatory stdout helpers ───────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── OpenEnv HTTP client ────────────────────────────────────────────────────

class OpenEnvClient:
    def __init__(self, base_url: str = ENV_URL):
        self.base  = base_url.rstrip("/")
        self._http = httpx.Client(timeout=30.0)

    def reset(self, task: Optional[str] = None, seed: Optional[int] = None) -> Dict:
        payload: Dict[str, Any] = {}
        if task:
            payload["task"] = task
        if seed is not None:
            payload["seed"] = seed
        r = self._http.post(f"{self.base}/reset", json=payload)
        r.raise_for_status()
        return r.json()

    def step(self, name: str, arguments: Dict) -> Dict:
        payload = {"action": {"name": name, "arguments": arguments}}
        r = self._http.post(f"{self.base}/step", json=payload)
        r.raise_for_status()
        return r.json()

    def state(self) -> Dict:
        r = self._http.get(f"{self.base}/state")
        r.raise_for_status()
        return r.json()

    def close(self) -> None:
        self._http.close()


# ── LLM agent ─────────────────────────────────────────────────────────────

class LLMAgent:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model  = model

    def act(self, observation: Dict) -> Dict:
        global _AUTH_FAILED
        if _AUTH_FAILED:
            # Don't retry after a confirmed auth failure.
            return FALLBACK_ACTION

        messages = observation.get("messages", [])
        tools    = observation.get("available_tools", [])

        # Convert 'developer' role → 'system' for OpenAI compatibility;
        # also serialise prior tool_calls into the expected wire format.
        oai_msgs: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role", "")
            if role == "developer":
                oai_msgs.append({"role": "system", "content": m.get("content", "")})
            elif role == "tool":
                oai_msgs.append({
                    "role":         "tool",
                    "content":      m.get("content", ""),
                    "tool_call_id": m.get("tool_call_id", "call_0"),
                })
            else:
                msg: Dict[str, Any] = {"role": role, "content": m.get("content")}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [
                        {
                            "id":   f"call_{i}",
                            "type": "function",
                            "function": {
                                "name":      tc["function"]["name"],
                                "arguments": json.dumps(tc["function"].get("arguments", {})),
                            },
                        }
                        for i, tc in enumerate(m["tool_calls"])
                    ]
                oai_msgs.append(msg)

        # Inject system prompt at front if missing
        if not oai_msgs or oai_msgs[0]["role"] != "system":
            oai_msgs.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        try:
            completion = self.client.chat.completions.create(
                model       = self.model,
                messages    = oai_msgs,
                tools       = tools if tools else None,
                tool_choice = "required" if tools else None,
                temperature = TEMPERATURE,
                max_tokens  = MAX_TOKENS,
            )
            msg = completion.choices[0].message
            if msg.tool_calls:
                tc = msg.tool_calls[0]
                return {
                    "name":      tc.function.name,
                    "arguments": json.loads(tc.function.arguments or "{}"),
                }
            # Fallback: model returned text — try to parse JSON
            text = (msg.content or "").strip()
            if text.startswith("{"):
                parsed = json.loads(text)
                if "name" in parsed:
                    return parsed
        except Exception as exc:
            _handle_llm_error(exc)

        return FALLBACK_ACTION


# ── LLM error handler ─────────────────────────────────────────────────────

def _handle_llm_error(exc: Exception) -> None:
    """Print a human-friendly error message and set the auth-failure flag on 403."""
    global _AUTH_FAILED
    msg = str(exc)
    if "403" in msg or "insufficient permissions" in msg.lower() or "authentication" in msg.lower():
        _AUTH_FAILED = True
        print(
            "  [LLM error] 403 Forbidden — your HF token lacks 'Inference Providers' permission.",
            flush=True,
        )
        print(
            "  Fix: go to https://huggingface.co/settings/tokens, edit your token,",
            flush=True,
        )
        print(
            "       tick 'Make calls to Inference Providers', save and re-export HF_TOKEN.",
            flush=True,
        )
    elif "401" in msg or "unauthorized" in msg.lower():
        _AUTH_FAILED = True
        print(f"  [LLM error] 401 Unauthorized — check that your HF_TOKEN is correct.", flush=True)
    else:
        print(f"  [LLM error] {exc}", flush=True)


# ── Random-agent baseline --------------------------------------------------

class RandomAgent:
    """Picks a random tool with null args — no LLM calls required."""

    def act(self, observation: Dict) -> Dict:
        tools = observation.get("available_tools", [])
        if not tools:
            return FALLBACK_ACTION
        tool  = random.choice(tools)["function"]
        props = tool.get("parameters", {}).get("properties", {})
        return {"name": tool["name"], "arguments": {k: None for k in props}}


# ── Single episode runner ──────────────────────────────────────────────────

def run_episode(
    env_client: OpenEnvClient,
    agent,
    task: str,
    episode_seed: int,
    verbose: bool = True,
) -> Dict:
    """
    Runs one episode and emits the mandatory [START] / [STEP]* / [END] lines.
    Returns a summary dict for post-run aggregation.
    """
    obs       = env_client.reset(task=task, seed=episode_seed)
    rewards:  List[float] = []
    reasons:  List[str]   = []
    steps_taken = 0
    score       = 0.0
    success     = False
    t0          = time.perf_counter()

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    if verbose:
        tools_listed = [t["function"]["name"] for t in obs.get("available_tools", [])]
        last_user    = next(
            (m["content"] for m in reversed(obs.get("messages", [])) if m["role"] == "user"),
            "",
        )
        print(f"  seed={episode_seed}  tools={tools_listed}", flush=True)
        print(f"  user ▶ {last_user[:100]}", flush=True)

    try:
        for step in range(1, MAX_STEPS + 1):
            state = env_client.state()
            if state["done"]:
                break

            action = agent.act(obs)
            action_str = f"{action['name']}({json.dumps(action['arguments'])})"

            if verbose:
                print(f"    step {step}: {action_str[:80]}", flush=True)

            result = env_client.step(action["name"], action["arguments"])
            obs    = result["observation"]
            reward = result.get("reward", 0.0)
            done   = result.get("done", False)
            info   = result.get("info", {})

            # last_action_error from info if present
            error: Optional[str] = info.get("last_action_error") or info.get("error") or None

            rewards.append(reward)
            reasons.append(info.get("reason", ""))
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if verbose:
                print(f"           → reward={reward:+.2f}  reason={info.get('reason','')}  done={done}", flush=True)

            if done:
                break

        # Score = mean reward normalised to [0, 1] assuming reward ∈ [-0.5, 1.0]
        # Using the same _norm logic as the graders but applied to cumulative mean.
        cum_reward = sum(rewards)
        mean_r     = cum_reward / max(steps_taken, 1)
        score      = max(0.0, min(1.0, (mean_r - (-0.5)) / (1.0 - (-0.5))))
        success    = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "task":             task,
        "seed":             episode_seed,
        "steps":            steps_taken,
        "cumulative_reward": round(sum(rewards), 4),
        "mean_reward":      round(sum(rewards) / max(steps_taken, 1), 4),
        "score":            round(score, 4),
        "success":          success,
        "rewards":          rewards,
        "reasons":          reasons,
        "elapsed_ms":       elapsed,
    }


# ── Full benchmark suite ───────────────────────────────────────────────────

def run_suite(
    env_client: OpenEnvClient,
    agent,
    tasks: List[str],
    episodes_per_task: int,
    verbose: bool = True,
) -> Dict:
    all_results: List[Dict] = []
    for task in tasks:
        print(f"\n── Task: {task} ──", flush=True)
        for ep in range(episodes_per_task):
            seed   = ep * 7 + 13
            result = run_episode(env_client, agent, task, episode_seed=seed, verbose=verbose)
            all_results.append(result)

    cum_rewards = [r["cumulative_reward"] for r in all_results]
    scores      = [r["score"] for r in all_results]
    summary = {
        "total_episodes":   len(all_results),
        "mean_score":       round(sum(scores) / len(scores), 4) if scores else 0.0,
        "mean_cum_reward":  round(sum(cum_rewards) / len(cum_rewards), 4) if cum_rewards else 0.0,
        "max_cum_reward":   max(cum_rewards) if cum_rewards else 0.0,
        "min_cum_reward":   min(cum_rewards) if cum_rewards else 0.0,
        "results":          all_results,
    }
    return summary


# ── Entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="OpenEnv Mobile-Actions inference script")
    parser.add_argument("--task",     choices=TASKS + ["all"], default="all")
    parser.add_argument("--episodes", type=int, default=2, help="Episodes per task")
    parser.add_argument("--env-url",  default=ENV_URL, help="Base URL of the OpenEnv server")
    parser.add_argument("--dry-run",  action="store_true", help="Use random agent (no LLM calls)")
    parser.add_argument("--verbose",  action="store_true", default=True)
    args = parser.parse_args()

    tasks = TASKS if args.task == "all" else [args.task]
    env_url = args.env_url

    print("=" * 60, flush=True)
    print("  OpenEnv Mobile-Actions — Inference", flush=True)
    print("=" * 60, flush=True)
    print(f"  Model    : {MODEL_NAME}", flush=True)
    print(f"  Env URL  : {env_url}", flush=True)
    print(f"  Tasks    : {tasks}", flush=True)
    print(f"  Episodes : {args.episodes} per task", flush=True)
    print(f"  Dry-run  : {args.dry_run}", flush=True)
    print("=" * 60, flush=True)

    env_client = OpenEnvClient(env_url)

    # Verify env is reachable
    try:
        health = httpx.get(f"{env_url}/health", timeout=10)
        health.raise_for_status()
        print(f"\n  ✓ Env healthy: {health.json()}", flush=True)
    except Exception as exc:
        print(f"\n  ✗ Env not reachable at {env_url}: {exc}", flush=True)
        print("    Start the server with: uvicorn server:app --port 7860", flush=True)
        sys.exit(1)

    if args.dry_run:
        print("\n  [DRY RUN] Using random agent — no LLM calls.\n", flush=True)
        agent = RandomAgent()
    else:
        if not API_KEY:
            print("  ✗ HF_TOKEN / API_KEY not set. Use --dry-run or set the env var.", flush=True)
            sys.exit(1)
        llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        agent      = LLMAgent(llm_client, MODEL_NAME)

        # ── Pre-flight: verify the LLM API key works before running episodes ──
        print("  Checking LLM API connectivity...", flush=True)
        try:
            llm_client.chat.completions.create(
                model      = MODEL_NAME,
                messages   = [{"role": "user", "content": "ping"}],
                max_tokens = 5,
            )
            print("  ✓ LLM API reachable\n", flush=True)
        except Exception as exc:
            exc_str = str(exc)
            if "403" in exc_str or "insufficient permissions" in exc_str.lower():
                print(
                    "  ✗ HF token error (403): 'Inference Providers' permission not enabled.",
                    flush=True,
                )
                print(
                    "    → Go to https://huggingface.co/settings/tokens",
                    flush=True,
                )
                print(
                    "    → Edit your token and tick 'Make calls to Inference Providers'",
                    flush=True,
                )
                print(
                    "    → Save, re-copy the token, and re-run with the new HF_TOKEN.",
                    flush=True,
                )
            elif "401" in exc_str:
                print("  ✗ HF token error (401): invalid or expired token.", flush=True)
            else:
                print(f"  ✗ LLM API error: {exc}", flush=True)
            sys.exit(1)

    summary = run_suite(env_client, agent, tasks, args.episodes, verbose=args.verbose)

    print("\n" + "=" * 60, flush=True)
    print("  SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"  Episodes       : {summary['total_episodes']}", flush=True)
    print(f"  Mean score     : {summary['mean_score']:.4f}", flush=True)
    print(f"  Mean cum reward: {summary['mean_cum_reward']:+.4f}", flush=True)
    print(f"  Max cum reward : {summary['max_cum_reward']:+.4f}", flush=True)
    print(f"  Min cum reward : {summary['min_cum_reward']:+.4f}", flush=True)
    print("=" * 60, flush=True)

    output_path = "inference_results.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved → {output_path}", flush=True)

    env_client.close()


if __name__ == "__main__":
    main()
