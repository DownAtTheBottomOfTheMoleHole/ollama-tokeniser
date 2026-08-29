import sys
from types import SimpleNamespace

from ollama_tokeniser.ollama_client import generate


class CharacterTokenizer:
    def encode(self, text, *, add_special_tokens=False):
        return [ord(character) for character in text]

    def decode(self, token_ids, *, skip_special_tokens=True):
        return "".join(chr(token_id) for token_id in token_ids)


def test_generate_sends_truncated_prompt(monkeypatch):
    calls = {}

    class Client:
        def __init__(self, host=None):
            calls["host"] = host

        def generate(self, **kwargs):
            calls.update(kwargs)
            return {"response": "ok"}

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=Client))

    response = generate(
        "0123456789",
        model="qwen2.5:7b",
        tokenizer=CharacterTokenizer(),
        context_size=7,
        reserve_tokens=2,
        template_tokens=0,
        temperature=0.2,
    )

    assert calls["prompt"] == "56789"
    assert calls["stream"] is False
    assert calls["options"] == {"num_ctx": 7, "num_predict": 2, "temperature": 0.2}
    assert response["truncation"]["removed_tokens"] == 5
