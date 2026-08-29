import pytest

from ollama_tokeniser.core import truncate_prompt


class CharacterTokenizer:
    """Predictable test tokenizer: each Unicode character is one token."""

    def encode(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(character) for character in text]

    def decode(self, token_ids, *, skip_special_tokens=True):
        assert skip_special_tokens is True
        return "".join(chr(token_id) for token_id in token_ids)


@pytest.fixture
def tokenizer():
    return CharacterTokenizer()


def test_short_prompt_is_unchanged(tokenizer):
    result = truncate_prompt(
        "hello", tokenizer, context_size=10, reserve_tokens=2, template_tokens=1
    )

    assert result.text == "hello"
    assert result.original_tokens == 5
    assert result.kept_tokens == 5
    assert result.truncated is False


def test_tail_strategy_keeps_recent_content(tokenizer):
    result = truncate_prompt(
        "0123456789",
        tokenizer,
        context_size=8,
        reserve_tokens=2,
        template_tokens=1,
        strategy="tail",
    )

    assert result.text == "56789"
    assert result.removed_tokens == 5
    assert result.truncated is True


def test_head_strategy_keeps_opening_content(tokenizer):
    result = truncate_prompt(
        "0123456789",
        tokenizer,
        context_size=8,
        reserve_tokens=2,
        template_tokens=1,
        strategy="head",
    )

    assert result.text == "01234"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"context_size": 0}, "context_size"),
        ({"reserve_tokens": -1}, "reserve_tokens"),
        ({"template_tokens": -1}, "template_tokens"),
        (
            {"context_size": 4, "reserve_tokens": 3, "template_tokens": 1},
            "reserve_tokens plus template_tokens",
        ),
        ({"strategy": "middle"}, "strategy"),
    ],
)
def test_invalid_budgets_are_rejected(tokenizer, kwargs, message):
    with pytest.raises(ValueError, match=message):
        truncate_prompt("hello", tokenizer, **kwargs)
