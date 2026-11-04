import json
import logging

import pytest

from ollama_tokeniser.proxy import (
    ProxyConfig,
    TokenizingProxy,
    filter_tag_payload,
    route_is_allowed,
    sanitise_log_text,
    sanitise_response_header,
    whitelisted_route_path,
)


def test_proxy_config_loads_model_mapping(tmp_path):
    path = tmp_path / "tokenizers.json"
    path.write_text(
        json.dumps({"model_tokenizers": {"model:tag": "org/tokenizer"}, "context_size": 8192})
    )

    config = ProxyConfig.from_file(path)

    assert config.model_tokenizers == {"model:tag": "org/tokenizer"}
    assert config.context_size == 8192
    assert config.local_files_only is True


def test_proxy_config_loads_builtin_catalogue(tmp_path):
    path = tmp_path / "tokenizers.json"
    path.write_text(json.dumps({"catalogue": "builtin"}))

    config = ProxyConfig.from_file(path)

    assert config.model_tokenizers["qwen2.5-coder"] == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    assert "llava" in config.unsupported_models


def test_proxy_config_requires_mapping(tmp_path):
    path = tmp_path / "tokenizers.json"
    path.write_text('{"model_tokenizers": {}}')

    with pytest.raises(ValueError, match="enable a catalogue"):
        ProxyConfig.from_file(path)


def test_tag_filter_hides_models_without_tokenizer_mapping():
    body = json.dumps({"models": [{"name": "safe:latest"}, {"name": "unmapped:latest"}]}).encode()

    filtered = json.loads(filter_tag_payload(body, {"safe:latest": "org/safe"}))

    assert filtered == {"models": [{"name": "safe:latest"}]}


def test_tag_filter_accepts_catalogue_family_for_any_tag():
    body = json.dumps({"models": [{"name": "qwen2.5-coder:32b"}]}).encode()

    filtered = json.loads(filter_tag_payload(body, {"qwen2.5-coder": "org/tokenizer"}))

    assert filtered == {"models": [{"name": "qwen2.5-coder:32b"}]}


@pytest.mark.parametrize(
    ("method", "path"),
    [("DELETE", "/api/delete"), ("POST", "/api/pull"), ("POST", "/api/push")],
)
def test_mutating_ollama_routes_are_blocked(method, path):
    assert route_is_allowed(method, path) is False


def test_required_vscode_routes_are_allowed():
    assert route_is_allowed("GET", "/api/tags") is True
    assert route_is_allowed("POST", "/api/show") is True
    assert route_is_allowed("POST", "/api/chat") is True


def test_whitelisted_route_path_strips_query():
    assert whitelisted_route_path("POST", "/api/chat?stream=true") == "/api/chat"


def test_whitelisted_route_path_rejects_unknown_route():
    with pytest.raises(ValueError, match="not allowed"):
        whitelisted_route_path("POST", "/api/pull")


def test_sanitise_log_text_escapes_newlines():
    assert sanitise_log_text("line1\r\nline2\nline3") == "line1\\r\\nline2\\nline3"


def test_tokenizer_loading_log_sanitises_model_on_cache_miss(monkeypatch, caplog):
    model = "qwen2.5-coder:latest\nFORGED LOG ENTRY"
    tokenizer_name = "org/tokenizer\nFORGED TOKENIZER ENTRY"
    config = ProxyConfig(model_tokenizers={"qwen2.5-coder": tokenizer_name})
    monkeypatch.setattr("ollama_tokeniser.proxy.load_tokenizer", lambda *args, **kwargs: object())
    server = TokenizingProxy(("127.0.0.1", 0), "http://127.0.0.1:11434", config)

    try:
        with caplog.at_level(logging.INFO, logger="ollama-tokeniser-proxy"):
            server.tokenizer_for(model)
    finally:
        server.server_close()

    assert "org/tokenizer\\nFORGED TOKENIZER ENTRY" in caplog.text
    assert "qwen2.5-coder:latest\\nFORGED LOG ENTRY" in caplog.text
    assert "\nFORGED LOG ENTRY" not in caplog.text
    assert "\nFORGED TOKENIZER ENTRY" not in caplog.text


def test_sanitise_response_header_drops_invalid_name():
    assert sanitise_response_header("X-Test\r\nInjected", "value") is None


def test_sanitise_response_header_removes_newlines_from_value():
    assert sanitise_response_header("X-Test", "ok\r\nInjected: bad") == (
        "X-Test",
        "okInjected: bad",
    )


def test_proxy_config_rejects_invalid_budget():
    with pytest.raises(ValueError, match="token reserves"):
        ProxyConfig(
            model_tokenizers={"model": "tokenizer"},
            context_size=100,
            reserve_tokens=100,
        )
