"""Shared fixtures for smoke tests requiring real API keys."""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

# Load API keys from .env.local (project root)
load_dotenv(".env.local")

SMOKE_MODEL = "claude-sonnet-4-6"


def _has_key(env_var: str) -> bool:
    """Return True if the environment variable is set and non-placeholder."""
    val = os.environ.get(env_var, "")
    return bool(val) and val != "your-key-here"


@pytest.fixture(scope="session")
def requires_anthropic_key() -> None:
    """Skip the test if ANTHROPIC_API_KEY is missing or placeholder."""
    if not _has_key("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set — skipping smoke test")


@pytest.fixture(scope="session")
def requires_openai_key() -> None:
    """Skip the test if OPENAI_API_KEY is missing or placeholder."""
    if not _has_key("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set — skipping smoke test")


@pytest.fixture(scope="session")
def requires_gemini_key() -> None:
    """Skip the test if GOOGLE_API_KEY is missing or placeholder."""
    if not _has_key("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY not set — skipping smoke test")


@pytest.fixture()
def llm_client(requires_anthropic_key):  # noqa: ARG001
    """Return an LLMClient configured from environment variables."""
    from attractor.llm.client import LLMClient

    return LLMClient.from_env()
