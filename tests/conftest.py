"""Shared fixtures for smoke tests requiring real API keys."""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

# Load API keys from .env.local (project root)
load_dotenv(".env.local")

SMOKE_MODEL = "claude-sonnet-4-6"


@pytest.fixture(scope="session")
def requires_anthropic_key() -> None:
    """Skip the test if ANTHROPIC_API_KEY is missing or placeholder."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key or key == "your-key-here":
        pytest.skip("ANTHROPIC_API_KEY not set — skipping smoke test")


@pytest.fixture()
def llm_client(requires_anthropic_key):  # noqa: ARG001
    """Return an LLMClient configured from environment variables."""
    from attractor.llm.client import LLMClient

    return LLMClient.from_env()
