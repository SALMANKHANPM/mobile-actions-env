"""
server.py — OpenEnv-compliant FastAPI server for mobile-actions-env.

Endpoints
  POST /reset        → ObservationResponse
  POST /step         → StepResponse
  GET  /state        → StateResponse
  GET  /health       → {"status": "ok"}
  GET  /tasks        → list of task descriptors
  POST /reset/{task} → ObservationResponse  (task-specific reset)
"""

from __future__ import annotations

import copy
import json
import os
import random
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Episode bank (embedded — no HF download needed at startup) ─────────────
# Full dataset loaded lazily when HF_TOKEN is present; fallback to built-ins.
BUILTIN_EPISODES: List[Dict] = [
    # ── Task: calendar_scheduling ──────────────────────────────────────────
    {
        "task": "calendar_scheduling",
        "metadata": "calendar",
        "tools": [
            {"function": {"name": "create_calendar_event", "description": "Creates a calendar event.",
                          "parameters": {"type": "OBJECT", "properties": {
                              "title":    {"type": "STRING", "description": "Event title"},
                              "datetime": {"type": "STRING", "description": "ISO-8601 datetime"},
                              "email":    {"type": "STRING", "description": "Attendee email (optional)"},
                          }, "required": ["title", "datetime"]}}},
            {"function": {"name": "send_email", "description": "Sends an email.",
                          "parameters": {"type": "OBJECT", "properties": {
                              "to":      {"type": "STRING"},
                              "subject": {"type": "STRING"},
                              "body":    {"type": "STRING"},
                          }, "required": ["to", "subject"]}}},
        ],
        "messages": [
            {"role": "developer", "content": "You are a mobile assistant. Call tools as required. Do not narrate."},
            {"role": "user",      "content": "Schedule a meeting with Ms. Chen about the project proposal for Tuesday August 20th at 2:00 PM.", "tool_calls": None},
            {"role": "assistant", "content": None, "tool_calls": [{"function": {
                "name": "create_calendar_event",
                "arguments": {"title": "Meeting with Ms. Chen about the project proposal", "datetime": "2024-08-20T14:00:00", "email": None},
            }}]},
        ],
    },
    {
        "task": "calendar_scheduling",
        "metadata": "calendar",
        "tools": [
            {"function": {"name": "create_calendar_event", "description": "Creates a calendar event.",
                          "parameters": {"type": "OBJECT", "properties": {
                              "title":    {"type": "STRING"},
                              "datetime": {"type": "STRING"},
                          }, "required": ["title", "datetime"]}}},
        ],
        "messages": [
            {"role": "developer", "content": "You are a mobile assistant."},
            {"role": "user",      "content": "Book a dentist appointment for Friday 23rd at 10am.", "tool_calls": None},
            {"role": "assistant", "content": None, "tool_calls": [{"function": {
                "name": "create_calendar_event",
                "arguments": {"title": "Dentist appointment", "datetime": "2024-08-23T10:00:00"},
            }}]},
        ],
    },
    # ── Task: map_navigation ───────────────────────────────────────────────
    {
        "task": "map_navigation",
        "metadata": "maps",
        "tools": [
            {"function": {"name": "show_map", "description": "Shows a location on the map.",
                          "parameters": {"type": "OBJECT", "properties": {
                              "query": {"type": "STRING", "description": "Location or search query"},
                          }, "required": ["query"]}}},
            {"function": {"name": "search_contacts", "description": "Searches contacts.",
                          "parameters": {"type": "OBJECT", "properties": {"query": {"type": "STRING"}}, "required": ["query"]}}},
        ],
        "messages": [
            {"role": "developer", "content": "You are a mobile assistant."},
            {"role": "user",      "content": "Show me the nearest coffee shop.", "tool_calls": None},
            {"role": "assistant", "content": None, "tool_calls": [{"function": {
                "name": "show_map",
                "arguments": {"query": "coffee shop near me"},
            }}]},
        ],
    },
    {
        "task": "map_navigation",
        "metadata": "maps",
        "tools": [
            {"function": {"name": "show_map", "description": "Shows a location.",
                          "parameters": {"type": "OBJECT", "properties": {
                              "query": {"type": "STRING"},
                          }, "required": ["query"]}}},
        ],
        "messages": [
            {"role": "developer", "content": "You are a mobile assistant."},
            {"role": "user",      "content": "Navigate to Hyderabad Airport.", "tool_calls": None},
            {"role": "assistant", "content": None, "tool_calls": [{"function": {
                "name": "show_map",
                "arguments": {"query": "Rajiv Gandhi International Airport Hyderabad"},
            }}]},
        ],
    },
    # ── Task: email_communication ──────────────────────────────────────────
    {
        "task": "email_communication",
        "metadata": "email",
        "tools": [
            {"function": {"name": "send_email", "description": "Sends an email.",
                          "parameters": {"type": "OBJECT", "properties": {
                              "to":      {"type": "STRING"},
                              "subject": {"type": "STRING"},
                              "body":    {"type": "STRING"},
                          }, "required": ["to", "subject"]}}},
            {"function": {"name": "make_call", "description": "Makes a phone call.",
                          "parameters": {"type": "OBJECT", "properties": {
                              "phone_number": {"type": "STRING"},
                              "name":         {"type": "STRING"},
                          }, "required": ["phone_number"]}}},
        ],
        "messages": [
            {"role": "developer", "content": "You are a mobile assistant."},
            {"role": "user",      "content": "Email priya@example.com about the 3PM meeting tomorrow.", "tool_calls": None},
            {"role": "assistant", "content": None, "tool_calls": [{"function": {
                "name": "send_email",
                "arguments": {"to": "priya@example.com", "subject": "3PM meeting tomorrow", "body": "Hi Priya, just a reminder about our 3PM meeting tomorrow."},
            }}]},
        ],
    },
    {
        "task": "email_communication",
        "metadata": "email",
        "tools": [
            {"function": {"name": "send_email", "description": "Sends an email.",
                          "parameters": {"type": "OBJECT", "properties": {
                              "to":      {"type": "STRING"},
                              "subject": {"type": "STRING"},
                              "body":    {"type": "STRING"},
                          }, "required": ["to", "subject"]}}},
        ],
        "messages": [
            {"role": "developer", "content": "You are a mobile assistant."},
            {"role": "user",      "content": "Send an email to raj@corp.com saying the report is ready.", "tool_calls": None},
            {"role": "assistant", "content": None, "tool_calls": [{"function": {
                "name": "send_email",
                "arguments": {"to": "raj@corp.com", "subject": "Report ready", "body": "Hi, the report is ready."},
            }}]},
        ],
    },
    # ── Task: media_control ────────────────────────────────────────────────
    {
        "task": "media_control",
        "metadata": "media",
        "tools": [
            {"function": {"name": "play_music", "description": "Plays music.",
                          "parameters": {"type": "OBJECT", "properties": {
                              "query":  {"type": "STRING"},
                              "artist": {"type": "STRING"},
                          }, "required": ["query"]}}},
            {"function": {"name": "set_alarm", "description": "Sets an alarm.",
                          "parameters": {"type": "OBJECT", "properties": {
                              "time":  {"type": "STRING"},
                              "label": {"type": "STRING"},
                          }, "required": ["time"]}}},
        ],
        "messages": [
            {"role": "developer", "content": "You are a mobile assistant."},
            {"role": "user",      "content": "Play some lo-fi music to help me focus.", "tool_calls": None},
            {"role": "assistant", "content": None, "tool_calls": [{"function": {
                "name": "play_music",
                "arguments": {"query": "lo-fi focus music", "artist": None},
            }}]},
        ],
    },
    {
        "task": "media_control",
        "metadata": "media",
        "tools": [
            {"function": {"name": "set_alarm", "description": "Sets an alarm.",
                          "parameters": {"type": "OBJECT", "properties": {
                              "time":  {"type": "STRING"},
                              "label": {"type": "STRING"},
                          }, "required": ["time"]}}},
            {"function": {"name": "play_music", "description": "Plays music.",
                          "parameters": {"type": "OBJECT", "properties": {"query": {"type": "STRING"}}, "required": ["query"]}}},
        ],
        "messages": [
            {"role": "developer", "content": "You are a mobile assistant."},
            {"role": "user",      "content": "Wake me up at 6:30 AM tomorrow.", "tool_calls": None},
            {"role": "assistant", "content": None, "tool_calls": [{"function": {
                "name": "set_alarm",
                "arguments": {"time": "06:30", "label": "Wake up"},
            }}]},
        ],
    },
    # ── Task: multi_turn ───────────────────────────────────────────────────
    {
        "task": "multi_turn",
        "metadata": "multi-turn",
        "tools": [
            {"function": {"name": "create_calendar_event", "description": "Creates a calendar event.",
                          "parameters": {"type": "OBJECT", "properties": {
                              "title":    {"type": "STRING"},
                              "datetime": {"type": "STRING"},
                          }, "required": ["title", "datetime"]}}},
            {"function": {"name": "send_email", "description": "Sends an email.",
                          "parameters": {"type": "OBJECT", "properties": {
                              "to":      {"type": "STRING"},
                              "subject": {"type": "STRING"},
                              "body":    {"type": "STRING"},
                          }, "required": ["to", "subject"]}}},
        ],
        "messages": [
            {"role": "developer", "content": "You are a mobile assistant."},
            {"role": "user",      "content": "Book a dentist appointment for Friday at 10am.", "tool_calls": None},
            {"role": "assistant", "content": None, "tool_calls": [{"function": {
                "name": "create_calendar_event",
                "arguments": {"title": "Dentist appointment", "datetime": "2024-08-23T10:00:00"},
            }}]},
            {"role": "user",      "content": "Also remind Dr. Mehta's clinic by email.", "tool_calls": None},
            {"role": "assistant", "content": None, "tool_calls": [{"function": {
                "name": "send_email",
                "arguments": {"to": "clinic@drmehta.com", "subject": "Appointment reminder Friday 10am", "body": "Hi, confirming my appointment on Friday at 10am."},
            }}]},
        ],
    },
]

TASK_IDS = ["calendar_scheduling", "map_navigation", "email_communication", "media_control", "multi_turn"]


# ── Pydantic models ────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task: Optional[str] = None          # filter to a specific task type
    episode_index: Optional[int] = None  # pin to specific episode
    seed: Optional[int] = None


class ActionModel(BaseModel):
    name: str
    arguments: Dict[str, Any] = {}


class StepRequest(BaseModel):
    action: ActionModel


class ObservationResponse(BaseModel):
    messages: List[Dict[str, Any]]
    available_tools: List[Dict[str, Any]]
    turn_index: int
    metadata: Optional[str]
    episode_id: int


class StepResponse(BaseModel):
    observation: ObservationResponse
    reward: float
    done: bool
    info: Dict[str, Any]


class StateResponse(BaseModel):
    episode_id: Optional[int]
    task: Optional[str]
    turn_index: int
    max_turns: int
    done: bool
    cumulative_reward: float
    available_tools: List[str]
    pending_gt_calls: int
    metadata: Optional[str]
    elapsed_ms: float


# ── Reward engine ──────────────────────────────────────────────────────────

class RewardEngine:
    EXACT_MATCH       =  1.00
    ARGS_PRESENT      =  0.60
    PARTIAL_ARGS      =  0.30
    WRONG_TOOL        =  0.00
    HALLUCINATED_TOOL = -0.25
    SCHEMA_VIOLATION  = -0.50

    def compute(self, action: Dict, gt_call: Dict, tool_names: List[str], tool_schemas: Dict) -> tuple:
        pred_name = action.get("name", "")
        pred_args = action.get("arguments", {}) or {}
        gt_name   = gt_call.get("function", {}).get("name", "")
        gt_args   = gt_call.get("function", {}).get("arguments", {}) or {}

        info: Dict[str, Any] = {"pred_name": pred_name, "gt_name": gt_name}

        if pred_name not in tool_names:
            info["reason"] = "hallucinated_tool"
            return self.HALLUCINATED_TOOL, info

        schema   = tool_schemas.get(pred_name, {})
        required = schema.get("required", [])
        props    = schema.get("properties", {})

        # basic type check for schema violation
        for k, v in pred_args.items():
            if k not in props:
                continue
            expected = props[k].get("type", "").upper()
            if expected == "STRING" and v is not None and not isinstance(v, str):
                info["reason"] = "schema_violation"
                info["errors"] = [f"'{k}' must be STRING"]
                return self.SCHEMA_VIOLATION, info

        if pred_name != gt_name:
            info["reason"] = "wrong_tool"
            return self.WRONG_TOOL, info

        if not required:
            info["reason"] = "exact_match_no_required_args"
            return self.EXACT_MATCH, info

        present = [k for k in required if pred_args.get(k) not in (None, "")]
        frac_present = len(present) / len(required)

        if frac_present < 1.0:
            info["reason"] = "partial_args"
            info["fraction_present"] = round(frac_present, 3)
            return round(self.PARTIAL_ARGS * frac_present, 4), info

        value_matches = sum(
            1 for k in required
            if str(pred_args.get(k, "")).lower().strip() == str(gt_args.get(k, "")).lower().strip()
        )
        vf = value_matches / len(required)
        reward = self.ARGS_PRESENT + (self.EXACT_MATCH - self.ARGS_PRESENT) * vf
        info["reason"]         = "exact_match" if vf == 1.0 else "args_present_partial_values"
        info["value_fraction"] = round(vf, 3)
        return round(reward, 4), info


# ── Tool simulator ──────────────────────────────────────────────────────────

def simulate_tool_result(name: str, args: dict) -> str:
    sims = {
        "send_email":            lambda a: f"Email sent to {a.get('to','?')} ✓",
        "create_calendar_event": lambda a: f"Event '{a.get('title','?')}' at {a.get('datetime','?')} created ✓",
        "show_map":              lambda a: f"Map showing: {a.get('query','?')} ✓",
        "search_contacts":       lambda a: f"Found contacts for '{a.get('query','?')}' ✓",
        "make_call":             lambda a: f"Calling {a.get('phone_number') or a.get('name','?')} ✓",
        "send_message":          lambda a: f"Message sent to {a.get('to','?')} ✓",
        "set_alarm":             lambda a: f"Alarm set for {a.get('time','?')} ✓",
        "play_music":            lambda a: f"Playing: {a.get('query','?')} ✓",
        "web_search":            lambda a: f"Results for: {a.get('query','?')} ✓",
    }
    fn = sims.get(name, lambda a: f"{name} executed ✓")
    return fn(args)


# ── Environment state (singleton per server process) ───────────────────────

class EnvState:
    def __init__(self):
        self.reset_to_idle()

    def reset_to_idle(self):
        self.episode_id: Optional[int]   = None
        self.task: Optional[str]         = None
        self.turn_idx: int               = 0
        self.max_turns: int              = 10
        self.done: bool                  = True
        self.history: List[dict]         = []
        self.tools_raw: List[dict]       = []
        self.tool_names: List[str]       = []
        self.tool_schemas: Dict          = {}
        self.gt_calls: List[dict]        = []
        self.user_turns: List[dict]      = []
        self.cum_reward: float           = 0.0
        self.meta: Optional[str]         = None
        self.start_ts: float             = 0.0


_state  = EnvState()
_reward = RewardEngine()

app = FastAPI(title="mobile-actions-env", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Helpers ────────────────────────────────────────────────────────────────

def _pick_episode(task: Optional[str], episode_index: Optional[int], seed: Optional[int]) -> tuple:
    pool = BUILTIN_EPISODES
    if task:
        pool = [e for e in pool if e.get("task") == task]
        if not pool:
            raise HTTPException(400, f"Unknown task '{task}'. Valid: {TASK_IDS}")

    # try to load from HF if token present (optional enhancement)
    rng = random.Random(seed) if seed is not None else random
    if episode_index is not None:
        idx = episode_index % len(pool)
    else:
        idx = rng.randrange(len(pool))
    return pool[idx], idx


def _build_observation() -> ObservationResponse:
    raw_tools = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": schema.get("description", ""),
                "parameters": {"type": "OBJECT", "properties": schema["properties"], "required": schema["required"]},
            },
        }
        for name, schema in _state.tool_schemas.items()
    ]
    return ObservationResponse(
        messages        = copy.deepcopy(_state.history),
        available_tools = raw_tools,
        turn_index      = _state.turn_idx,
        metadata        = _state.meta,
        episode_id      = _state.episode_id or 0,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/metadata")
def metadata():
    """OpenEnv spec: returns environment name and description."""
    return {
        "name":        "mobile-actions-env",
        "version":     "1.0.0",
        "description": (
            "OpenEnv environment for LLM tool-calling on mobile. "
            "Agents learn to call the right mobile API (calendar, maps, email, "
            "music, contacts) from natural-language user requests through "
            "graded reward signals."
        ),
        "tasks":       TASK_IDS,
        "tags":        ["openenv", "tool-calling", "mobile", "llm"],
    }


@app.get("/schema")
def schema():
    """OpenEnv spec: returns action, observation, and state schemas."""
    return {
        "action": {
            "type": "object",
            "required": ["name", "arguments"],
            "properties": {
                "name":      {"type": "string", "description": "Tool name to invoke"},
                "arguments": {"type": "object", "description": "Key-value arguments for the tool"},
            },
        },
        "observation": {
            "type": "object",
            "properties": {
                "messages":        {"type": "array",   "description": "Conversation history (OpenAI message format)"},
                "available_tools": {"type": "array",   "description": "Tool schemas the agent may call"},
                "turn_index":      {"type": "integer", "description": "Current turn within episode"},
                "metadata":        {"type": "string",  "description": "Task category tag"},
                "episode_id":      {"type": "integer", "description": "Episode index"},
            },
        },
        "state": {
            "type": "object",
            "properties": {
                "episode_id":        {"type": "integer"},
                "task":              {"type": "string"},
                "turn_index":        {"type": "integer"},
                "max_turns":         {"type": "integer"},
                "done":              {"type": "boolean"},
                "cumulative_reward": {"type": "number"},
                "available_tools":   {"type": "array", "items": {"type": "string"}},
                "pending_gt_calls":  {"type": "integer"},
                "metadata":          {"type": "string"},
                "elapsed_ms":        {"type": "number"},
            },
        },
    }


@app.post("/mcp")
async def mcp(request: Request):
    """Minimal JSON-RPC 2.0 endpoint required by OpenEnv spec (MCP transport)."""
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    method = payload.get("method", "")
    req_id = payload.get("id", 1)

    if method == "initialize":
        result = {
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": "mobile-actions-env", "version": "1.0.0"},
            "capabilities": {"tools": {}},
        }
    elif method == "tools/list":
        result = {
            "tools": [
                {"name": t, "description": f"Mobile action: {t}", "inputSchema": {"type": "object"}}
                for t in ["create_calendar_event", "show_map", "send_email", "play_music", "set_alarm"]
            ]
        }
    else:
        result = {"message": f"method '{method}' not implemented"}

    return {"jsonrpc": "2.0", "id": req_id, "result": result}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {"id": t, "episodes": len([e for e in BUILTIN_EPISODES if e["task"] == t])}
            for t in TASK_IDS
        ]
    }


@app.post("/reset", response_model=ObservationResponse)
@app.post("/reset/{task}", response_model=ObservationResponse)
def reset(req: ResetRequest = ResetRequest(), task: Optional[str] = None):
    task_filter = task or req.task
    episode, idx = _pick_episode(task_filter, req.episode_index, req.seed)

    _state.reset_to_idle()
    _state.episode_id = idx
    _state.task       = episode.get("task")
    _state.meta       = episode.get("metadata")
    _state.tools_raw  = episode.get("tools", [])
    _state.done       = False
    _state.start_ts   = time.perf_counter()

    # parse tool schemas
    for t in _state.tools_raw:
        fn   = t["function"]
        name = fn["name"]
        params = fn.get("parameters", {})
        _state.tool_names.append(name)
        _state.tool_schemas[name] = {
            "description": fn.get("description", ""),
            "properties":  params.get("properties", {}),
            "required":    params.get("required", []),
        }

    # parse messages
    raw_msgs = episode.get("messages", [])
    for m in raw_msgs:
        if m["role"] == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                _state.gt_calls.append(tc)
        if m["role"] == "user":
            _state.user_turns.append(m)

    _state.max_turns = max(10, len(_state.gt_calls) * 2)

    # seed history
    dev_msg = next((m for m in raw_msgs if m["role"] == "developer"), None)
    if dev_msg:
        _state.history.append({"role": "developer", "content": dev_msg["content"]})
    if _state.user_turns:
        _state.history.append({"role": "user", "content": _state.user_turns[0]["content"]})

    return _build_observation()


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    if _state.done:
        raise HTTPException(400, "Episode is done. Call /reset first.")

    action  = {"name": req.action.name, "arguments": req.action.arguments}
    gt_call = _state.gt_calls[_state.turn_idx] if _state.turn_idx < len(_state.gt_calls) else {}

    reward, info = _reward.compute(action, gt_call, _state.tool_names, _state.tool_schemas)
    _state.cum_reward += reward

    # append to history
    _state.history.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [{"function": {"name": req.action.name, "arguments": req.action.arguments}}],
    })
    _state.history.append({
        "role": "tool",
        "content": simulate_tool_result(req.action.name, req.action.arguments),
        "tool_call_id": f"call_{_state.turn_idx}",
    })

    _state.turn_idx += 1
    no_more_gt = _state.turn_idx >= len(_state.gt_calls)
    max_hit    = _state.turn_idx >= _state.max_turns
    _state.done = no_more_gt or max_hit

    # next user turn
    if not _state.done and _state.turn_idx < len(_state.user_turns):
        _state.history.append({
            "role": "user",
            "content": _state.user_turns[_state.turn_idx]["content"],
        })

    elapsed_ms = (time.perf_counter() - _state.start_ts) * 1000
    info.update({
        "turn_index":        _state.turn_idx,
        "cumulative_reward": round(_state.cum_reward, 4),
        "elapsed_ms":        round(elapsed_ms, 2),
        "gt_call":           gt_call,
    })

    return StepResponse(
        observation = _build_observation(),
        reward      = reward,
        done        = _state.done,
        info        = info,
    )


@app.get("/state", response_model=StateResponse)
def state():
    elapsed_ms = (time.perf_counter() - _state.start_ts) * 1000 if _state.start_ts else 0.0
    return StateResponse(
        episode_id        = _state.episode_id,
        task              = _state.task,
        turn_index        = _state.turn_idx,
        max_turns         = _state.max_turns,
        done              = _state.done,
        cumulative_reward = round(_state.cum_reward, 4),
        available_tools   = list(_state.tool_names),
        pending_gt_calls  = max(0, len(_state.gt_calls) - _state.turn_idx),
        metadata          = _state.meta,
        elapsed_ms        = round(elapsed_ms, 2),
    )


def main():
    """Entry point for the openenv-core server runner."""
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        workers=1,
        log_level="info",
    )


if __name__ == "__main__":
    main()

