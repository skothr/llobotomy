# Integration tests — manual / env-gated harness notes

The deep test tier has tests that require live external services
(FastAPI backend) or large model files (TinyLlama GGUF).  These tests
self-skip when their prerequisites aren't present, so `ctest` runs
cleanly on a fresh box.  This doc records the setup steps to run them
locally when you're working on the relevant paths.

## HFProxy integration tests

Three test binaries gated by `LLOB_INTEGRATION_TEST=1`:

- `test_hf_proxy_capture_integration` — Phase 2 capture flow against a
  live FastAPI backend.  Verifies attention + residual + QKV fetch.
- `test_hf_proxy_intervention_integration` — Phase 3 intervention.
  Loads baseline, ablates head (5, 3), re-captures, verifies the
  attention pattern changed.
- `test_hf_proxy_ws_integration` — Phase 4 WS streams.  Asserts the
  logit-lens trajectory + getOutputLogits round-trip.

### Setup

From the repo root:

```bash
# 1. Bring up the FastAPI backend with TinyLlama loaded.
cd testing/gui
./run.sh   # starts the dev stack — backend on :8000, frontend on :5173

# 2. (in another shell) build + run the integration tests.
cd testing/llm_engine_cpp/build-verify-on-llama
LLOB_INTEGRATION_TEST=1 \
LLOB_BACKEND_URL=http://127.0.0.1:8000 \
  ctest -L deep --output-on-failure
```

The tests fail fast if the backend isn't reachable or the session
can't be created.  `run.sh` waits for the backend to be ready before
returning to the prompt, so there's no race.

### What "skipped" looks like

Without `LLOB_INTEGRATION_TEST=1`, the test binaries exit 0
immediately with a message like:

```
test_hf_proxy_capture_integration: LLOB_INTEGRATION_TEST not set; skipping
```

ctest reports them as Passed (not Skipped) because the binary's exit
code is the contract.

## LlamaCpp deep tests

`test_llama_cpp_real` requires a GGUF file via
`LLOB_DEEP_LLAMA_GGUF_PATH`.  The canonical fixture is TinyLlama
converted from HF:

```bash
# Convert TinyLlama HF → GGUF f16 (~2.2 GB, 201 tensors).
cd /home/ai/ai-projects/llm
python lib/llama.cpp/convert_hf_to_gguf.py \
  testing/.cache/models/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/<HASH>/ \
  --outtype f16 \
  --outfile /tmp/tinyllama-1.1b-chat-f16.gguf

# Run the deep tier.
LLOB_DEEP_LLAMA_GGUF_PATH=/tmp/tinyllama-1.1b-chat-f16.gguf \
  ctest -L deep --output-on-failure
```

The deep test exercises: loadCheckpoint, capabilities transitions,
state_dict enumeration, prefill capture, head ablation
write-back, sibling-head invariance, token streaming.

## GGUF inspector deep test

`test_real_gguf` requires `LLOB_DEEP_GGUF_PATH` pointing at any GGUF
file.  The TinyLlama fixture above works.

## Future CI

`scripts/verify.sh --llama` is the canonical full-matrix gate.  When
CI runs, it should arrange for the TinyLlama fixture to exist (cached
or built on first run) so the deep tier exercises real model paths.
The FastAPI integration tier likely stays manual-only until the CI
runner has GPU + the Python backend stack available.
