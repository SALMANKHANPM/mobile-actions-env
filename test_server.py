"""
tests/test_server.py
Full integration test suite — runs against a live server.
Usage:
    # Start server first:
    uvicorn server.app:app --port 7860 &
    # Then:
    python -m pytest tests/ -v
"""
import pytest
import httpx

BASE = "http://localhost:7860"
client = httpx.Client(base_url=BASE, timeout=15)

TASKS = [
    "calendar_scheduling",
    "map_navigation",
    "email_communication",
    "media_control",
    "multi_turn",
]

TASK_ACTIONS = {
    "calendar_scheduling": {"name": "create_calendar_event",
                            "arguments": {"title": "Test meeting", "datetime": "2024-08-20T14:00:00"}},
    "map_navigation":      {"name": "show_map", "arguments": {"query": "coffee shop"}},
    "email_communication": {"name": "send_email", "arguments": {"to": "a@b.com", "subject": "test"}},
    "media_control":       {"name": "play_music", "arguments": {"query": "lo-fi beats"}},
    "multi_turn":          {"name": "create_calendar_event", "arguments": {"title": "Meeting", "datetime": "2024-08-23T10:00:00"}},
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_tasks_endpoint():
    r = client.get("/tasks")
    assert r.status_code == 200
    data = r.json()
    assert "tasks" in data
    ids = [t["id"] for t in data["tasks"]]
    for task in TASKS:
        assert task in ids


@pytest.mark.parametrize("task", TASKS)
def test_reset_returns_observation(task):
    r = client.post("/reset", json={"task": task, "seed": 0})
    assert r.status_code == 200, r.text
    obs = r.json()
    assert "messages" in obs
    assert "available_tools" in obs
    assert "turn_index" in obs
    assert obs["turn_index"] == 0
    assert len(obs["available_tools"]) > 0


@pytest.mark.parametrize("task", TASKS)
def test_step_returns_valid_structure(task):
    client.post("/reset", json={"task": task, "seed": 1})
    action = TASK_ACTIONS[task]
    r = client.post("/step", json={"action": action})
    assert r.status_code == 200, r.text
    result = r.json()
    assert "reward" in result
    assert "done" in result
    assert "info" in result
    assert "observation" in result
    reward = result["reward"]
    assert -1.0 <= reward <= 1.0, f"Reward out of range: {reward}"


@pytest.mark.parametrize("task", TASKS)
def test_state_returns_snapshot(task):
    client.post("/reset", json={"task": task, "seed": 2})
    r = client.get("/state")
    assert r.status_code == 200, r.text
    state = r.json()
    assert "episode_id" in state
    assert "turn_index" in state
    assert "done" in state
    assert "cumulative_reward" in state
    assert state["turn_index"] == 0
    assert state["done"] is False


def test_step_before_reset_returns_400():
    # Force done state by exhausting an episode
    obs = client.post("/reset", json={"task": "map_navigation", "seed": 99}).json()
    # step until done
    for _ in range(10):
        r = client.post("/step", json={"action": {"name": "show_map", "arguments": {"query": "x"}}})
        if r.json().get("done"):
            break
    r2 = client.post("/step", json={"action": {"name": "show_map", "arguments": {"query": "x"}}})
    assert r2.status_code == 400


def test_hallucinated_tool_gives_negative_reward():
    client.post("/reset", json={"task": "calendar_scheduling", "seed": 0})
    r = client.post("/step", json={"action": {"name": "fake_tool_xyz", "arguments": {}}})
    assert r.status_code == 200
    assert r.json()["reward"] < 0.0


def test_correct_tool_gives_positive_reward():
    client.post("/reset", json={"task": "map_navigation", "seed": 0})
    r = client.post("/step", json={"action": {"name": "show_map", "arguments": {"query": "coffee shop near me"}}})
    assert r.status_code == 200
    assert r.json()["reward"] > 0.0


@pytest.mark.parametrize("task", TASKS)
def test_full_episode_runs_to_done(task):
    client.post("/reset", json={"task": task, "seed": 42})
    done = False
    for _ in range(20):
        r = client.get("/state")
        if r.json()["done"]:
            done = True
            break
        action = TASK_ACTIONS[task]
        result = client.post("/step", json={"action": action}).json()
        if result["done"]:
            done = True
            break
    assert done, f"Episode never reached done for task={task}"


def test_rewards_in_range():
    """All rewards across 3 full episodes must be in [-0.5, 1.0]."""
    for i, task in enumerate(TASKS[:3]):
        client.post("/reset", json={"task": task, "seed": i})
        for _ in range(5):
            st = client.get("/state").json()
            if st["done"]:
                break
            r = client.post("/step", json={"action": TASK_ACTIONS[task]}).json()
            reward = r["reward"]
            assert -0.5 <= reward <= 1.0, f"reward {reward} out of [-0.5, 1.0]"


if __name__ == "__main__":
    # Run without pytest for quick manual check
    tests = [
        test_health,
        test_tasks_endpoint,
        test_hallucinated_tool_gives_negative_reward,
        test_correct_tool_gives_positive_reward,
    ]
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:
            print(f"  FAIL  {fn.__name__}  —  {e}")
