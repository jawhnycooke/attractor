"""Tests for subagent management tools."""

import asyncio
from types import SimpleNamespace

import pytest

from attractor.agent.tools.subagent import (
    close_agent,
    send_input,
    spawn_agent,
    wait_agent,
)


def _make_env() -> SimpleNamespace:
    """Create a minimal environment with a subagents dict."""
    return SimpleNamespace(subagents={})


class TestSpawnAgent:
    @pytest.mark.asyncio
    async def test_spawn_returns_unique_agent_id(self) -> None:
        env = _make_env()
        r1 = await spawn_agent({"task": "task one"}, env)
        r2 = await spawn_agent({"task": "task two"}, env)

        assert not r1.is_error
        assert not r2.is_error

        # Extract agent IDs from output
        ids = list(env.subagents.keys())
        assert len(ids) == 2
        assert ids[0] != ids[1]

    @pytest.mark.asyncio
    async def test_spawn_missing_task_returns_error(self) -> None:
        env = _make_env()
        result = await spawn_agent({"task": ""}, env)
        assert result.is_error
        assert "required" in result.output.lower()

        result2 = await spawn_agent({}, env)
        assert result2.is_error


class TestWaitAgent:
    @pytest.mark.asyncio
    async def test_wait_agent_nonexistent_returns_error(self) -> None:
        env = _make_env()
        result = await wait_agent({"agent_id": "bogus_id"}, env)
        assert result.is_error
        assert "no subagent" in result.output.lower()


class TestCloseAgent:
    @pytest.mark.asyncio
    async def test_close_agent_removes_from_registry(self) -> None:
        env = _make_env()

        # Spawn an agent
        await spawn_agent({"task": "test task"}, env)
        agent_id = list(env.subagents.keys())[0]

        # Close it
        close_result = await close_agent({"agent_id": agent_id}, env)
        assert not close_result.is_error

        # Now waiting should fail
        wait_result = await wait_agent({"agent_id": agent_id}, env)
        assert wait_result.is_error

    @pytest.mark.asyncio
    async def test_close_agent_cancels_running_task(self) -> None:
        env = _make_env()

        # Spawn an agent
        await spawn_agent({"task": "long task"}, env)
        agent_id = list(env.subagents.keys())[0]

        # Attach a fake running task
        async def long_running():
            await asyncio.sleep(100)
            return "done"

        task = asyncio.create_task(long_running())
        env.subagents[agent_id].task = task

        # Close should cancel the task
        result = await close_agent({"agent_id": agent_id}, env)
        assert not result.is_error

        # Yield to the event loop so the cancellation propagates
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert task.cancelled()


class TestSendInput:
    @pytest.mark.asyncio
    async def test_send_input_to_closed_agent_fails(self) -> None:
        env = _make_env()

        # Spawn and close
        await spawn_agent({"task": "temp"}, env)
        agent_id = list(env.subagents.keys())[0]
        env.subagents[agent_id].closed = True

        result = await send_input({"agent_id": agent_id, "message": "hello"}, env)
        assert result.is_error
        assert "closed" in result.output.lower()


class TestSessionScopedIsolation:
    @pytest.mark.asyncio
    async def test_session_scoped_isolation(self) -> None:
        """Regression for B3: different environments should have isolated
        subagent registries."""
        env1 = _make_env()
        env2 = _make_env()

        await spawn_agent({"task": "task on env1"}, env1)
        await spawn_agent({"task": "task on env2"}, env2)

        # Each environment has its own subagent
        assert len(env1.subagents) == 1
        assert len(env2.subagents) == 1

        # The IDs should be different
        id1 = list(env1.subagents.keys())[0]
        id2 = list(env2.subagents.keys())[0]
        assert id1 != id2

        # Closing on env1 should not affect env2
        await close_agent({"agent_id": id1}, env1)
        assert len(env1.subagents) == 0
        assert len(env2.subagents) == 1
