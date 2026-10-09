#!/usr/bin/env bash
# Full verification for language-forensics-service. No GPU/Triton image or
# weights required: the contract tests use a protocol-accurate ONNX-Runtime
# mock Triton and a tiny fixture model.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export PATH="$HOME/.cargo/bin:${PATH}"

echo "==> Backend fixtures + Triton config schema validation (real model_config.proto)"
uv run --extra onnx python tools/make_fixtures.py
uv run --with grpcio-tools --with protobuf --with onnx python tools/verify_repo.py

echo "==> Rust: fmt + clippy + tests (contract crate, every backend, gateway e2e)"
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace

echo "==> Rust: each backend builds on its own (feature isolation)"
cargo build -q -p gateway --no-default-features --features horizon-guard
cargo build -q -p gateway --no-default-features --features prompt-guard

echo "==> Rust release binary + Python OpenAPI contract test (real ONNX fixture)"
cargo build --release -p gateway
uv run --extra test python tests/test_detect_integration.py
uv run --extra test python tests/test_real_model.py   # skips if no real model exported

echo "All checks passed."
