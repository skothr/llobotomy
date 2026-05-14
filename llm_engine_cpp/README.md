# llm_engine — C++ backend-abstraction library for the llobotomy GUI

`testing/llm_engine_cpp/` is the C++23 substrate every interpretability
workspace in the llobotomy GUI talks to.  It defines a single uniform
`ModelView` data structure plus a `Model` interface, and ships four
concrete backends behind that interface:

| Backend | Source | What it does |
|---|---|---|
| `MockModel` | `mock_model.cpp` | Opt-in screenshot / demo data when `LLOB_BACKEND=mock` is selected.  Inherits `Model` like any other backend — never a silent fallback. |
| `HFProxyEngine` | `hf_proxy_engine.cpp` | HTTP/WS to the FastAPI backend at `testing/gui/backend/` (Python + HuggingFace transformers).  For any HF-shaped model. |
| `GgufInspectorEngine` | `gguf_inspector_engine.cpp` | Pure-native GGUF reader.  Loads tensor topology + state dict via the in-house `GgufParser` + mmap-backed `GgufSource`.  No inference. |
| `LlamaCppEngine` | `llama_cpp_engine.cpp` + `llama_cpp_capture.cpp` | Embedded llama.cpp.  Local CUDA inference + cb_eval-based activation capture + token streaming + head/component ablation + steering vectors that genuinely mutate the forward pass via GPU write-back. |

## Build

Standard CMake.  Two relevant options:

```bash
cmake -S . -B build \
  -DLLM_ENGINE_USE_MOCK_DATA=OFF \      # OFF for real backends; ON for screenshot mode
  -DLLM_ENGINE_BUILD_LLAMA_CPP=ON       # compile in the embedded llama.cpp backend
cmake --build build -j
```

Defaults work for most cases:
- `LLM_ENGINE_USE_MOCK_DATA=OFF` is correct for any real run.
- `LLM_ENGINE_BUILD_LLAMA_CPP=OFF` is the default; turn ON to link
  against the pre-built `lib/llama.cpp/build/bin/libllama.so`.

## Test

```bash
./scripts/verify.sh --llama   # 4-config matrix (smoke + deep)
```

17 smoke tests run every time.  5 deep tests env-gate on:

- `LLOB_DEEP_GGUF_PATH=/path/to/file.gguf` — for `test_real_gguf`.
- `LLOB_DEEP_LLAMA_GGUF_PATH=/path/to/file.gguf` — for `test_llama_cpp_real`.
- `LLOB_INTEGRATION_TEST=1` + live FastAPI backend — for the
  `test_hf_proxy_*_integration` tier.

See `docs/INTEGRATION_TESTS.md` for the full setup walk-through.

## Run the demo

`scripts/demo.sh` does the full end-to-end loop in one command —
convert the cached TinyLlama HF model to GGUF if needed, build the
engine with LlamaCpp enabled, build gui_cpp, launch the binary
pointed at the GGUF.

```bash
./scripts/demo.sh
```

Override via env vars: `TINYLLAMA_HF_DIR`, `DEMO_GGUF`, `LLOB_BACKEND`,
`ENGINE_BUILD_DIR`, `GUI_BUILD_DIR`.

## Documents to read next

- `docs/HOW_TO_ADD_A_BACKEND.md` — what to implement to plug a new
  backend in.
- `docs/SUPPORTED_ARCHITECTURES.md` — model families with capture
  coverage and their gotchas.
- `docs/INTEGRATION_TESTS.md` — env-gated test setup runbook.
- `docs/WAVE_D_LLAMACPP_PLAN.md` — original LlamaCppEngine plan; now
  a historical reference (the extensions all landed — state_dict,
  token streaming, head ablation, steering vectors, component
  ablation).
- `docs/WAVE_E_LIBTORCH_PLAN.md` — the deferred `LibtorchEngine` plan
  with a `2026-05-13` status section explaining why it stays deferred
  (LlamaCpp covers the use cases; libtorch only uniquely adds
  training-in-engine + sub-tensor surgery, neither of which is
  required today).

## Design notes

- **Honest-empty contract**: concrete backends inherit `Model`
  directly, never `MockModel`.  Unimplemented getters return the
  no-data sentinel (NaN / `{}` / `kNoInt`) — never silently fall
  through to mock data.  Locked in by
  `tests/test_no_mock_leak.cpp`.
- **Capabilities honest about load state**: `getCapabilities()` is
  conservative-false until `loadCheckpoint` succeeds.  A backend
  pre-asserting `has_topology=true` before any data exists is the
  same class of dishonesty as silent mock fallback.
- **Atomic shared_ptr publish**: per-step capture is published via
  `atomic<shared_ptr<const CaptureBundle>> current`.  UI reads
  lock-free; streaming decodes copy the bundle so consumers holding
  a previous pointer see immutable state.
- **Intervention via cb_eval write-back**: head + component ablation
  and steering vector application all use
  `ggml_backend_tensor_set` to write to the GPU buffer BEFORE
  downstream ops consume it.  These are real forward-pass
  mutations, not just visualisation overrides.
