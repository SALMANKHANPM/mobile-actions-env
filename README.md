---
title: Mobile Actions Env
emoji: 📱
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
tags:
  - openenv
---

# mobile-actions-env

> An **OpenEnv**-compatible reinforcement-learning environment for LLM tool-calling on mobile devices — built on [google/mobile-actions](https://huggingface.co/datasets/google/mobile-actions).

---

## Description & Motivation

Modern LLMs are increasingly deployed as mobile assistants that must select and invoke the right API (calendar, maps, email, music) from natural-language user requests. However, there is no standardised RL training loop for this setting.

**mobile-actions-env** closes that gap: it wraps the Google Mobile-Actions dataset inside an OpenEnv-compliant HTTP server so any agent framework can interact through the canonical `reset()` / `step()` / `state()` loop, receiving dense graded rewards per tool call.

Key design goals:
- **Graded, dense rewards** — not just binary success/fail; partial credit for partially correct tool calls encourages learning even from imperfect actions.
- **Multiple task types** — five distinct mobile-action categories, each with different tool schemas.
- **Multi-turn support** — the `multi_turn` task requires chaining two sequential tool calls correctly.
- **Zero external dependencies at runtime** — all episodes are embedded in the server; no HF download required to start.

---

## Action Space

Every action is a JSON object with exactly two fields:

```json
{ "name": "<tool_name>", "arguments": { "<key>": "<value>", ... } }
```

| Field | Type | Description |
|---|---|---|
| `name` | string | Name of the tool to invoke |
| `arguments` | object | Key-value arguments (types vary per tool schema) |

Available tools depend on the active episode and are returned in `available_tools` on each `reset()` / `step()`.

**Example tools across tasks:**

| Tool | Required Args | Task |
|---|---|---|
| `create_calendar_event` | `title`, `datetime` | calendar_scheduling |
| `show_map` | `query` | map_navigation |
| `send_email` | `to`, `subject` | email_communication |
| `play_music` | `query` | media_control |
| `set_alarm` | `time` | media_control |

---

## Observation Space

Each `reset()` and `step()` response includes an **ObservationResponse**:

```json
{
  "messages": [
    { "role": "developer", "content": "You are a mobile assistant." },
    { "role": "user",      "content": "Book a dentist appointment for Friday 23rd at 10am." }
  ],
  "available_tools": [
    {
      "type": "function",
      "function": {
        "name": "create_calendar_event",
        "description": "Creates a calendar event.",
        "parameters": {
          "type": "OBJECT",
          "properties": {
            "title":    { "type": "STRING" },
            "datetime": { "type": "STRING" }
          },
          "required": ["title", "datetime"]
        }
      }
    }
  ],
  "turn_index": 0,
  "metadata": "calendar",
  "episode_id": 1
}
```

| Field | Type | Description |
|---|---|---|
| `messages` | array | Full conversation history (OpenAI message format) |
| `available_tools` | array | Tool schemas the agent may call this turn |
| `turn_index` | int | Current turn within the episode |
| `metadata` | string | Task category tag |
| `episode_id` | int | Index of the current episode |

---

## Reward Semantics

Rewards are issued per step in the range `[-0.5, 1.0]`:

| Outcome | Reward |
|---|---|
| Exact tool name + all required arg values match | `+1.00` |
| Exact tool name + all required args present (values differ) | `+0.60` |
| Exact tool name + partial required args present | `+0.30 × fraction` |
| Wrong tool name (but in the action space) | `0.00` |
| Hallucinated tool (not in action space) | `−0.25` |
| Schema violation (bad type / missing required param) | `−0.50` |

---

## Tasks

### 1. `calendar_scheduling` — 🟢 Easy
Agent receives a natural-language scheduling request and must call `create_calendar_event` with correct `title` and `datetime` fields.  
**Max steps:** 5 · **Episodes:** 2

### 2. `map_navigation` — 🟢 Easy
Agent receives a location-lookup request and must call `show_map` with a relevant `query`.  
**Max steps:** 5 · **Episodes:** 2

### 3. `email_communication` — 🟡 Medium
Agent must call `send_email` with correct recipient (`to`) and `subject`. Body is optional but checked for value matching.  
**Max steps:** 5 · **Episodes:** 2

### 4. `media_control` — 🟡 Medium
Agent must choose between `play_music` (needs `query`) or `set_alarm` (needs `time`) based on the user request.  
**Max steps:** 5 · **Episodes:** 2

### 5. `multi_turn` — 🔴 Hard
Agent handles a two-turn conversation requiring two sequential tool calls in order: `create_calendar_event` then `send_email`. Both turns must be correct.  
**Max steps:** 10 · **Episodes:** 1

---

## Baseline Scores

Measured using `python inference.py --dry-run` (random agent) and `HF_TOKEN=... python inference.py` (Qwen2.5-72B-Instruct via HF Inference Providers):

| Agent | Mean Score | Notes |
|---|---|---|
| Random agent | 0.33 | Always picks `null` for all args → `partial_args` or `wrong_tool` (reward 0.00) |
| Qwen/Qwen2.5-72B-Instruct | ~0.87 | Tool name correct in almost all cases; value matching varies |

Score formula per episode:
```
score = clamp((mean_step_reward - (-0.5)) / (1.0 - (-0.5)), 0, 1)
```

---

## Project Structure

```
.
├── openenv.yaml           # OpenEnv spec manifest
├── Dockerfile             # Container definition (docker build + run)
├── requirements.txt       # Python dependencies
├── server.py              # FastAPI: /reset  /step  /state  /health  /tasks
├── inference.py           # Agent inference script (OpenAI client)
├── test_server.py         # Integration test suite (26 tests, all pass)
├── validate-submission.sh # Pre-submission validator
└── graders/
    ├── __init__.py        # Shared utilities (_norm, name_match, arg_present)
    ├── calendar_grader.py
    ├── maps_grader.py
    ├── email_grader.py
    ├── media_grader.py
    └── multiturn_grader.py
```

---

## Setup & Usage

### Local (no Docker)

```bash
pip install -r requirements.txt

# Start the environment server
uvicorn server:app --host 0.0.0.0 --port 7860 --reload

# Verify it's healthy
curl http://localhost:7860/health
```

### Docker

```bash
# Build
docker build -t openenv-mobile-actions .

# Run
docker run -p 7860:7860 openenv-mobile-actions

# Run with LLM credentials
docker run -p 7860:7860 \
  -e HF_TOKEN=$HF_TOKEN \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  openenv-mobile-actions
```

### Run the test suite

```bash
# Server must be running on :7860
pytest test_server.py -v   # 26 tests
```

### Run the inference agent

```bash
# Dry-run (random agent, no API key needed)
python inference.py --dry-run

# Real LLM — all tasks
export HF_TOKEN=hf_...
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py

# Single task, 3 episodes
python inference.py --task calendar_scheduling --episodes 3
```

### Pre-submission validation

```bash
# Requires your deployed HF Space URL
./validate-submission.sh https://your-space.hf.space .
```

---

## API Reference

### `POST /reset`
Start a new episode. Optionally filter by task or pin to a specific episode.

**Request:**
```json
{ "task": "calendar_scheduling", "seed": 42, "episode_index": null }
```
**Response:** `ObservationResponse` (see Observation Space above)

---

### `POST /step`
Submit a tool-call action. Returns the next observation, reward, done flag, and debug info.

**Request:**
```json
{ "action": { "name": "create_calendar_event", "arguments": { "title": "Meeting", "datetime": "2024-08-20T14:00:00" } } }
```
**Response:**
```json
{
  "observation": { "messages": [...], "available_tools": [...], "turn_index": 1, ... },
  "reward": 0.6,
  "done": true,
  "info": { "reason": "args_present_partial_values", "pred_name": "create_calendar_event", "gt_name": "create_calendar_event", "value_fraction": 0.5, "cumulative_reward": 0.6 }
}
```

---

### `GET /state`
Read-only snapshot of the current episode state.

**Response:**
```json
{
  "episode_id": 1, "task": "calendar_scheduling", "turn_index": 0,
  "max_turns": 10, "done": false, "cumulative_reward": 0.0,
  "available_tools": ["create_calendar_event"], "pending_gt_calls": 1,
  "metadata": "calendar", "elapsed_ms": 12.3
}
```

---

### `GET /health`
Returns `{"status": "ok", "version": "1.0.0"}`.

### `POST /reset/{task}`
Shorthand to reset directly to a specific task: `POST /reset/map_navigation`

### `GET /tasks`
Returns all task IDs with episode counts.

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | For LLM agent | — | HuggingFace API key (needs "Inference Providers" scope) |
| `API_BASE_URL` | For LLM agent | `https://router.huggingface.co/v1` | OpenAI-compatible LLM endpoint |
| `MODEL_NAME` | For LLM agent | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `OPENENV_URL` | Optional | `http://localhost:7860` | Override env server URL |
