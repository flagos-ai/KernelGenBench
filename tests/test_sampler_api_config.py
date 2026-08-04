from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from generator.sampler import utils


def test_openai_compatible_endpoint_uses_runtime_env_and_chat(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "runtime-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example/v1")

    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="kernel"))]
    )
    client_factory = MagicMock(return_value=client)
    monkeypatch.setattr(utils, "OpenAI", client_factory)

    result = utils.query_server(
        "generate a kernel",
        server_type="openai",
        model_name="custom-model",
        base_url="https://cli.example/v1",
    )

    assert result == "kernel"
    client_factory.assert_called_once_with(
        api_key="runtime-key",
        base_url="https://cli.example/v1",
    )
    request = client.chat.completions.create.call_args.kwargs
    assert request["model"] == "custom-model"
    assert request["messages"] == [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "generate a kernel"},
    ]


def test_anthropic_compatible_endpoint_uses_runtime_env(monkeypatch):
    import anthropic

    monkeypatch.setenv("ANTHROPIC_API_KEY", "runtime-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://env.example")

    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(text="kernel")]
    )
    client_factory = MagicMock(return_value=client)
    monkeypatch.setattr(anthropic, "Anthropic", client_factory)

    result = utils.query_server(
        "generate a kernel",
        server_type="anthropic",
        model_name="custom-model",
        base_url="https://cli.example",
    )

    assert result == "kernel"
    client_factory.assert_called_once_with(
        api_key="runtime-key",
        base_url="https://cli.example",
    )


def test_provider_name_is_not_an_api_format():
    with pytest.raises(ValueError, match="OpenAI-compatible"):
        utils.query_server("prompt", server_type="some-provider")
