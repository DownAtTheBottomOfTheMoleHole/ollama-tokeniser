import json
from importlib.resources import files

from ollama_tokeniser.catalogue import ModelCatalogue, model_family, tokenizer_for_model


def test_builtin_catalogue_covers_ollama_snapshot() -> None:
    resource = files("ollama_tokeniser").joinpath("data/ollama-model-catalogue.json")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    catalogue = ModelCatalogue.from_payload(payload)

    assert catalogue.model_count >= 239
    assert catalogue.generated_at == "2026-08-29"
    assert len(catalogue.tokenizers) + len(catalogue.unsupported) == catalogue.model_count


def test_catalogue_resolves_any_tag_for_a_mapped_family() -> None:
    mappings = {
        "qwen2.5-coder": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "qwen2.5-coder:custom": "local/tokenizer",
    }

    assert tokenizer_for_model("qwen2.5-coder:32b", mappings) == mappings["qwen2.5-coder"]
    assert tokenizer_for_model("qwen2.5-coder:custom", mappings) == "local/tokenizer"
    assert tokenizer_for_model("unknown:latest", mappings) is None


def test_model_family_handles_tags_and_digests() -> None:
    assert model_family("namespace/model:7b") == "namespace/model"
    assert model_family("model@sha256:abc") == "model"
