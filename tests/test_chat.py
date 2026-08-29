from ollama_tokeniser.chat import count_chat_tokens, fit_chat_messages


class CharacterChatTokenizer:
    def encode(self, text, *, add_special_tokens=False):
        return list(map(ord, text))

    def decode(self, token_ids, *, skip_special_tokens=True):
        return "".join(map(chr, token_ids))

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        rendered = "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"
        return self.encode(rendered)


class MappingChatTokenizer(CharacterChatTokenizer):
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        ids = super().apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt
        )
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def test_chat_count_uses_input_ids_from_transformers_mapping_result():
    tokenizer = MappingChatTokenizer()
    messages = [{"role": "user", "content": "hello"}]

    assert count_chat_tokens(messages, tokenizer) == len(
        tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)[
            "input_ids"
        ]
    )


def test_chat_drops_old_turns_before_truncating_latest_message():
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "new question"},
    ]

    result = fit_chat_messages(
        messages,
        CharacterChatTokenizer(),
        context_size=60,
        reserve_tokens=5,
        template_tokens=0,
    )

    assert result.messages[0] == messages[0]
    assert result.messages[-1] == messages[-1]
    assert result.dropped_messages == 2
    assert result.kept_tokens <= result.max_input_tokens


def test_chat_truncates_latest_message_at_token_boundary():
    result = fit_chat_messages(
        [{"role": "user", "content": "0123456789"}],
        CharacterChatTokenizer(),
        context_size=30,
        reserve_tokens=5,
        template_tokens=0,
        strategy="tail",
    )

    assert result.messages[0]["content"].endswith("789")
    assert result.kept_tokens <= 25
    assert result.truncated is True
