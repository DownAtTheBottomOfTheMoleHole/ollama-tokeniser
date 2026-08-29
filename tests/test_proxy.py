import json

import pytest

from ollama_tokeniser.proxy import ProxyConfig, filter_tag_payload, route_is_allowed


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


def test_proxy_config_rejects_invalid_budget():
    with pytest.raises(ValueError, match="token reserves"):
        ProxyConfig(
            model_tokenizers={"model": "tokenizer"},
            context_size=100,
            reserve_tokens=100,
        )
