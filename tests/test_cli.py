from ollama_tokeniser.cli import _parser


def test_truncate_command_accepts_prompt_argument():
    args = _parser().parse_args(["truncate", "--tokenizer", "example/tokenizer", "hello"])

    assert args.command == "truncate"
    assert args.prompt == "hello"


def test_generate_command_accepts_model():
    args = _parser().parse_args(
        ["generate", "--tokenizer", "example/tokenizer", "--model", "llama3.2", "hello"]
    )

    assert args.command == "generate"
    assert args.model == "llama3.2"
    assert args.prompt == "hello"
