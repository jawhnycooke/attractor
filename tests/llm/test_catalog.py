"""Tests for attractor.llm.catalog — model catalog lookups."""

from __future__ import annotations

from attractor.llm.catalog import MODELS, get_latest_model, get_model_info, list_models


class TestGetModelInfo:
    def test_lookup_by_model_id(self) -> None:
        info = get_model_info("claude-opus-4-6")
        assert info is not None
        assert info.model_id == "claude-opus-4-6"
        assert info.provider == "anthropic"

    def test_lookup_by_alias(self) -> None:
        info = get_model_info("opus")
        assert info is not None
        assert info.model_id == "claude-opus-4-6"

    def test_lookup_another_alias(self) -> None:
        info = get_model_info("gemini-flash")
        assert info is not None
        assert info.model_id == "gemini-3-flash-preview"
        assert info.provider == "gemini"

    def test_unknown_model_returns_none(self) -> None:
        result = get_model_info("nonexistent-model-xyz")
        assert result is None

    def test_empty_string_returns_none(self) -> None:
        result = get_model_info("")
        assert result is None


class TestListModels:
    def test_list_all_models(self) -> None:
        models = list_models()
        assert len(models) == len(MODELS)
        assert models is not MODELS  # should be a copy

    def test_filter_by_provider_anthropic(self) -> None:
        models = list_models(provider="anthropic")
        assert len(models) >= 1
        assert all(m.provider == "anthropic" for m in models)

    def test_filter_by_provider_openai(self) -> None:
        models = list_models(provider="openai")
        assert len(models) >= 1
        assert all(m.provider == "openai" for m in models)

    def test_filter_by_provider_gemini(self) -> None:
        models = list_models(provider="gemini")
        assert len(models) >= 1
        assert all(m.provider == "gemini" for m in models)

    def test_unknown_provider_returns_empty(self) -> None:
        models = list_models(provider="unknown-provider")
        assert models == []


class TestGetLatestModel:
    def test_get_latest_anthropic(self) -> None:
        model = get_latest_model("anthropic")
        assert model is not None
        assert model.provider == "anthropic"

    def test_get_latest_openai(self) -> None:
        model = get_latest_model("openai")
        assert model is not None
        assert model.provider == "openai"

    def test_get_latest_gemini(self) -> None:
        model = get_latest_model("gemini")
        assert model is not None
        assert model.provider == "gemini"

    def test_get_latest_with_reasoning_capability(self) -> None:
        model = get_latest_model("anthropic", capability="reasoning")
        assert model is not None
        assert model.supports_reasoning is True

    def test_get_latest_with_vision_capability(self) -> None:
        model = get_latest_model("openai", capability="vision")
        assert model is not None
        assert model.supports_vision is True

    def test_get_latest_with_tools_capability(self) -> None:
        model = get_latest_model("gemini", capability="tools")
        assert model is not None
        assert model.supports_tools is True

    def test_unknown_provider_returns_none(self) -> None:
        result = get_latest_model("nonexistent-provider")
        assert result is None
