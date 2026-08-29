# Security policy

## Supported versions

Security fixes are applied to the latest release on the `main` branch.

## Reporting a vulnerability

Do not report security vulnerabilities in a public issue. Use GitHub's private
vulnerability reporting or draft a private security advisory for this repository.

Include reproduction steps, affected versions, expected impact, and any suggested
mitigation. Please avoid including real credentials, private prompts, or sensitive
model data in the report.

## Security boundaries

The proxy is designed for local development. It binds to loopback, accepts only
the Ollama API routes required by VS Code Chat, limits request-body size, does not
log prompt content, disables remote tokenizer code, and uses cached tokenizer files
during normal operation.

Do not modify it to listen on a public or shared interface without adding proper
authentication, transport encryption, request isolation, and abuse controls.
