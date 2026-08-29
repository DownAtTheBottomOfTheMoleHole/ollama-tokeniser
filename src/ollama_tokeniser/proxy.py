"""Local Ollama reverse proxy that injects model-matched token truncation."""

from __future__ import annotations

import argparse
import http.client
import ipaddress
import json
import logging
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .chat import fit_chat_messages
from .core import load_tokenizer, truncate_prompt

LOG = logging.getLogger("ollama-tokeniser-proxy")
HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}
ALLOWED_ROUTES = {
    ("GET", "/api/tags"),
    ("GET", "/api/version"),
    ("POST", "/api/chat"),
    ("POST", "/api/generate"),
    ("POST", "/api/show"),
}


def route_is_allowed(method: str, path: str) -> bool:
    return (method, path.split("?", 1)[0]) in ALLOWED_ROUTES


def filter_tag_payload(body: bytes, mappings: dict[str, str]) -> bytes:
    """Hide models that cannot be safely handled by this proxy."""
    payload = json.loads(body)
    models = payload.get("models", [])
    payload["models"] = [
        model
        for model in models
        if str(model.get("name", model.get("model", ""))) in mappings
        or (
            ":" not in str(model.get("name", model.get("model", "")))
            and f"{model.get('name', model.get('model', ''))}:latest" in mappings
        )
    ]
    return json.dumps(payload, ensure_ascii=False).encode()


@dataclass(frozen=True, slots=True)
class ProxyConfig:
    model_tokenizers: dict[str, str]
    context_size: int = 32768
    reserve_tokens: int = 4096
    template_tokens: int = 128
    strategy: str = "tail"
    local_files_only: bool = True

    def __post_init__(self) -> None:
        if self.context_size <= 0:
            raise ValueError("context_size must be greater than zero")
        if self.reserve_tokens < 0 or self.template_tokens < 0:
            raise ValueError("token reserves cannot be negative")
        if self.reserve_tokens + self.template_tokens >= self.context_size:
            raise ValueError("token reserves must be smaller than context_size")
        if self.strategy not in ("head", "tail"):
            raise ValueError("strategy must be 'head' or 'tail'")
        if any(not model or not tokenizer for model, tokenizer in self.model_tokenizers.items()):
            raise ValueError("model_tokenizers entries cannot be empty")

    @classmethod
    def from_file(cls, path: Path) -> ProxyConfig:
        raw = json.loads(path.read_text(encoding="utf-8"))
        mappings = raw.get("model_tokenizers")
        if not isinstance(mappings, dict) or not mappings:
            raise ValueError("config must contain a non-empty model_tokenizers object")
        return cls(
            model_tokenizers={str(key): str(value) for key, value in mappings.items()},
            context_size=int(raw.get("context_size", 32768)),
            reserve_tokens=int(raw.get("reserve_tokens", 4096)),
            template_tokens=int(raw.get("template_tokens", 128)),
            strategy=str(raw.get("strategy", "tail")),
            local_files_only=bool(raw.get("local_files_only", True)),
        )


class TokenizingProxy(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], upstream: str, config: ProxyConfig):
        super().__init__(address, ProxyHandler)
        parsed = urlsplit(upstream)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("upstream must be an http(s) URL")
        self.upstream = parsed
        self.config = config
        self._tokenizers: dict[str, Any] = {}

    def tokenizer_for(self, model: str):
        tokenizer_name = self.config.model_tokenizers.get(model)
        if tokenizer_name is None and ":" not in model:
            tokenizer_name = self.config.model_tokenizers.get(f"{model}:latest")
        if tokenizer_name is None:
            raise KeyError(model)
        if model in self._tokenizers:
            return self._tokenizers[model]
        LOG.info("loading tokenizer %s for %s", tokenizer_name, model)
        tokenizer = load_tokenizer(tokenizer_name, local_files_only=self.config.local_files_only)
        if len(self._tokenizers) >= 8:
            self._tokenizers.pop(next(iter(self._tokenizers)))
        self._tokenizers[model] = tokenizer
        return tokenizer


class ProxyHandler(BaseHTTPRequestHandler):
    server: TokenizingProxy
    protocol_version = "HTTP/1.0"
    server_version = "ollama-tokeniser"
    sys_version = ""

    def do_GET(self) -> None:
        self._handle()

    def do_POST(self) -> None:
        self._handle()

    def do_DELETE(self) -> None:
        self._handle()

    def _handle(self) -> None:
        try:
            if not route_is_allowed(self.command, self.path):
                self._json_error(403, "Ollama API route is not allowed by this proxy")
                return
            body = self._read_body()
            truncation: dict[str, Any] | None = None
            if self.command == "POST" and self.path.split("?", 1)[0] in (
                "/api/chat",
                "/api/generate",
            ):
                body, truncation = self._truncate(body)
            self._forward(body, truncation)
        except KeyError as exc:
            self._json_error(
                422,
                f"no tokenizer mapping for Ollama model {exc.args[0]!r}; "
                "add it to model_tokenizers in the proxy config",
            )
        except (ValueError, json.JSONDecodeError) as exc:
            self._json_error(400, str(exc))
        except (OSError, http.client.HTTPException) as exc:
            LOG.exception("proxy request failed")
            self._json_error(502, f"Ollama upstream error: {exc}")
        except Exception as exc:  # keep client failures diagnosable at the HTTP boundary
            LOG.exception("tokenizer proxy failed")
            self._json_error(500, f"tokenizer proxy error: {exc}")

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 64 * 1024 * 1024:
            raise ValueError("request body exceeds 64 MiB")
        return self.rfile.read(length) if length else b""

    def _truncate(self, body: bytes) -> tuple[bytes, dict[str, Any]]:
        payload = json.loads(body)
        model = str(payload.get("model", ""))
        if not model:
            raise ValueError("request is missing model")
        tokenizer = self.server.tokenizer_for(model)
        options = payload.get("options") or {}
        context_size = int(options.get("num_ctx", self.server.config.context_size))
        reserve_tokens = int(options.get("num_predict", self.server.config.reserve_tokens))
        if reserve_tokens < 0:
            reserve_tokens = self.server.config.reserve_tokens

        if self.path.split("?", 1)[0] == "/api/chat":
            result = fit_chat_messages(
                payload.get("messages") or [],
                tokenizer,
                context_size=context_size,
                reserve_tokens=reserve_tokens,
                template_tokens=self.server.config.template_tokens,
                strategy=self.server.config.strategy,  # type: ignore[arg-type]
                tools=payload.get("tools") or None,
            )
            payload["messages"] = result.messages
            details = {
                "truncated": result.truncated,
                "original_tokens": result.original_tokens,
                "kept_tokens": result.kept_tokens,
                "dropped_messages": result.dropped_messages,
            }
        else:
            result = truncate_prompt(
                str(payload.get("prompt", "")),
                tokenizer,
                context_size=context_size,
                reserve_tokens=reserve_tokens,
                template_tokens=self.server.config.template_tokens,
                strategy=self.server.config.strategy,  # type: ignore[arg-type]
            )
            payload["prompt"] = result.text
            details = {
                "truncated": result.truncated,
                "original_tokens": result.original_tokens,
                "kept_tokens": result.kept_tokens,
                "removed_tokens": result.removed_tokens,
            }
        LOG.info("%s %s token accounting: %s", self.path, model, details)
        return json.dumps(payload, ensure_ascii=False).encode(), details

    def _forward(self, body: bytes, truncation: dict[str, Any] | None) -> None:
        upstream = self.server.upstream
        connection_type = (
            http.client.HTTPSConnection
            if upstream.scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connection_type(upstream.hostname, upstream.port, timeout=600)
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP and key.lower() not in ("host", "content-length")
        }
        if body:
            headers["Content-Length"] = str(len(body))
        connection.request(self.command, self.path, body=body or None, headers=headers)
        response = connection.getresponse()
        filtered_body: bytes | None = None
        if (
            self.command == "GET"
            and self.path.split("?", 1)[0] == "/api/tags"
            and response.status == 200
        ):
            filtered_body = filter_tag_payload(response.read(), self.server.config.model_tokenizers)
        self.send_response(response.status, response.reason)
        for key, value in response.getheaders():
            if key.lower() not in HOP_BY_HOP and key.lower() != "content-length":
                self.send_header(key, value)
        if truncation is not None:
            self.send_header("X-Ollama-Tokeniser", "applied")
            self.send_header("X-Ollama-Tokeniser-Truncated", str(truncation["truncated"]).lower())
        if filtered_body is not None:
            self.send_header("Content-Length", str(len(filtered_body)))
            self.send_header("X-Ollama-Tokeniser-Models", "filtered")
        self.end_headers()
        if filtered_body is not None:
            self.wfile.write(filtered_body)
        else:
            while chunk := response.read(64 * 1024):
                self.wfile.write(chunk)
                self.wfile.flush()
        connection.close()

    def _json_error(self, status: int, message: str) -> None:
        body = json.dumps({"error": message}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        LOG.info("%s - %s", self.client_address[0], fmt % args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--listen", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11435)
    parser.add_argument("--upstream", default="http://127.0.0.1:11434")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    try:
        listen_address = ipaddress.ip_address(args.listen)
    except ValueError:
        parser.error("--listen must be a loopback IP address")
    if not listen_address.is_loopback:
        parser.error("refusing a non-loopback --listen address")
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = ProxyConfig.from_file(args.config)
    server = TokenizingProxy((args.listen, args.port), args.upstream, config)
    LOG.info("listening on http://%s:%d -> %s", args.listen, args.port, args.upstream)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOG.info("stopping")
    finally:
        server.server_close()
    return 0


def cache_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download configured tokenizer files for subsequent offline use."
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = ProxyConfig.from_file(args.config)
    for model, tokenizer_name in config.model_tokenizers.items():
        print(f"Caching {tokenizer_name} for {model}...")
        load_tokenizer(tokenizer_name, local_files_only=False)
    print("Tokenizer cache is ready for offline proxy use.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
