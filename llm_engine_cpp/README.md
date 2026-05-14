# llm_engine — C++ backend-abstraction library for LLM interpretability tooling

`llm_engine` is a small, framework-agnostic C++23 library that gives
every interpretability tool the same view of an inspected model — its
weights, its live forward-pass activations, the interventions applied
to it, and the analyses derived from any of those — through one
coherent in-memory data structure: **`ModelView`**.

Each concrete backend (mock fake data, HTTP proxy to a Python runtime,
in-process GGUF inspection, embedded llama.cpp, etc.) populates the
same `ModelView` shape. Consumer code (`gui_cpp/`, future CLI tools)
holds a `Model&` and reads `view().tensors`, `view().current.load()`,
`view().surgery`, etc. — without knowing or caring which backend it's
talking to.

## At a glance

```
┌─────────────── consumers ───────────────┐
│ gui_cpp/ (Dear ImGui)   future tools…   │
└────────────────┬────────────────────────┘
                 │  llmengine::Model&   ←  view()
┌────────────────┴────────────────────────┐
│           llm_engine (this lib)         │
│                                         │
│   include/llm_engine/                   │
│     log.hpp            Severity / LogEntry
│     tensor_source.hpp  TensorSource ABC + DType
│     tensor_handle.hpp  TensorHandle + TensorRegistry
│     capture.hpp        CaptureBundle (one forward pass)
│     derived_cache.hpp  LRU-bounded analysis memoiser
│     model_view.hpp     ModelView + AttentionHeadRef + ComponentRef
│     model.hpp          Model + DTOs + Capabilities + LoadOptions
│     hf_proxy_engine.hpp                  │
│     gguf_inspector_engine.hpp            │  ← wave C
│                                         │
│   src/  (implementations of all of the above)
│                                         │
│   libs/  (private)                      │
│     cpp-httplib/    HTTP client (MIT)   │
│     nlohmann_json/  JSON       (MIT)    │
│                                         │
│   docs/                                 │
│     SUPPORTED_ARCHITECTURES.md          │
│                                         │
│   tests/                                │
│     test_tensor_handle / _derived_cache │
│     / _model_view / _mock_model_compat  │
└─────────────────────────────────────────┘
```

## The `ModelView` substrate

One struct per inspected model, holding everything an interpretability
tool wants to read or write:

| Field | What | Lifetime |
|---|---|---|
| `provenance`   | where the bytes came from (path, format, source label) | set once at load, frozen |
| `topology`     | n_layers, n_heads, dims, vocab, rope_theta, chat_template | set once at load, frozen |
| `tokenizer`    | vocab + encode/decode `std::function` slots | set once at load, frozen |
| `tensors`      | name→`TensorHandle`; lazy file-backed slices, no bytes loaded | set once at load, frozen |
| `captures`     | per-prompt `CaptureBundle` (attention/residual/QKV/logits) | engine writes, UI reads via atomic `current` |
| `surgery`      | structured ablations / steering / probes / weight deltas | UI mutates via setAblation/setSteering |
| `derived`      | memoised analyses (LRU, byte-bounded) | populated on demand from any thread |
| `capabilities` | bitset mirror of `getCapabilities()` (for path-API consumers) | set once at load |

Threading contract: static fields are read-only post-load; `current` /
`surgery` / `derived` are the only fields that mutate in steady state.
Full per-phase rules are in `include/llm_engine/model_view.hpp` (top-
of-file doc block).

## Random access from headers (no full load required)

`TensorSource` is the byte-supplier behind every `TensorHandle`:

```cpp
class TensorSource {
public:
    virtual void pread(std::size_t offset, std::size_t n_bytes, void* out) const = 0;
    virtual std::span<const std::byte> try_mmap() const { return {}; }
    virtual std::size_t size_bytes() const = 0;
    virtual bool loaded() const { return false; }
};
```

Concrete implementations:

| Source | Storage | `loaded()` |
|---|---|---|
| `InMemoryTensorSource` | owned byte buffer  | `true` |
| `Mulberry32Source`     | PRNG (mock)        | `false` |
| `GgufSource`           | mmap'd `.gguf` file (wave C) | `true` |
| `SafetensorsSource`    | mmap'd `.safetensors` (future) | `true` |
| `HfProxySource`        | HTTP range-GET (future) | `false` |

A 70B-parameter checkpoint becomes inspectable on a laptop without
130 GB resident — only the slices the user is currently looking at
are read.

## Concrete backends

- **`llmengine::MockModel`** — deterministic fake data backed by
  `Mulberry32Source` (each tensor's bytes are computed on demand from
  a PRNG seeded by its name). Same `TensorSource` ABI as real
  backends; consumers can't tell the difference. Enabled at compile
  time via `LLM_ENGINE_USE_MOCK_DATA=ON`. With the flag `OFF`, every
  per-DTO method returns the no-data sentinel — a release build can't
  silently ship mock numbers.
- **`llmengine::HFProxyEngine`** — HTTP+WS wrapper over the FastAPI
  backend at `testing/gui/backend/`. See
  `../gui_cpp/docs/HFPROXY_PLAN.md` for the per-method status.
- **`llmengine::GgufInspectorEngine`** — fully-native read-only
  inspector for `.gguf` files. Parses the header, enumerates tensors,
  serves slices via mmap'd `pread`. Supports the Llama / Qwen2 / Gemma /
  Mistral / Phi / Mixtral / GPT-2 / GPT-NeoX / Falcon / MPT / BLOOM /
  StarCoder / OPT families (see `docs/SUPPORTED_ARCHITECTURES.md`).

## `Model` interface

```cpp
struct Model {
    virtual const ModelView& view() const = 0;
    virtual CheckpointResult loadCheckpoint(std::string_view path, const LoadOptions& = {}) = 0;
    virtual void             unloadCheckpoint() = 0;
    virtual void             setActivePrompt(std::string_view) {}
    virtual void             setAblation   (const std::vector<std::string>& heads,
                                            const std::vector<std::string>& components) {}
    virtual void             setSteering   (const SteeringConfig&) {}
    virtual void             clearSteering () {}
    virtual Capabilities     getCapabilities() const { return {}; }
    virtual Progress         getProgress   () const { return {}; }
    virtual std::vector<LogEntry> drainEngineLogs() = 0;
    // + ~50 per-DTO getters from the v1 interface (kept for source compat)
};
```

A few discipline rules the interface enforces:

- **Per-frame samplers MUST be cheap.** UI threads call them ~60 times
  a second; backends cache aggressively (mostly via `view().current`).
- **Methods never throw on missing data.** They return the no-data
  sentinel (`kNoFloat`, `kNoInt`, empty vector, `""`) and the UI prints
  a placeholder. `noexcept(false)` is reserved for genuine IO failure.
- **Mutators default to no-op.** A backend that doesn't implement a
  hook isn't broken; the corresponding UI control just won't do
  anything.
- **`Capabilities` advertises what works.** The UI consults it before
  rendering controls that depend on optional features.

Full architectural contract (lifecycle, threading, per-area
requirements, error handling) is in
[`../gui_cpp/docs/ENGINE_API.md`](../gui_cpp/docs/ENGINE_API.md).

## Build & test

```bash
# Library + tests
cmake -S . -B build && cmake --build build -j

# Run smoke tests (fast, every commit)
ctest --test-dir build -L smoke

# Run deep tests (heavy data / behaviour validation; some need fixtures)
ctest --test-dir build -L deep

# Both modes must build green
cmake -S . -B build -DLLM_ENGINE_USE_MOCK_DATA=ON  && cmake --build build -j
cmake -S . -B build -DLLM_ENGINE_USE_MOCK_DATA=OFF && cmake --build build -j
```

Or as a sub-project (consumer side):

```cmake
add_subdirectory(/path/to/llm_engine_cpp ${CMAKE_BINARY_DIR}/llm_engine_cpp)
target_link_libraries(your_app PRIVATE llmengine::llm_engine)
```

`LLM_ENGINE_USE_MOCK_DATA` (default `ON`) controls MockModel's
behaviour. `LLM_ENGINE_BUILD_TESTS` (default `ON` when the engine is
the top-level project, off when consumed via `add_subdirectory`)
controls whether the test binaries get built.

## Adding a new backend

1. Subclass `Model` (or `MockModel` if you want the defaults to fall
   through to mock data while you're stubbing).
2. In `loadCheckpoint`, parse the source's header, populate
   `view.provenance` / `view.topology` / `view.tensors`. No weight
   bytes are loaded yet.
3. Implement a `TensorSource` subclass for your storage (file mmap,
   HTTP range, in-memory, …).
4. Wire `view.tensors` entries to share a single `shared_ptr<Source>`
   per file. Lifetime is automatic — last handle dropped → source
   dtor → file closed / unmap'd.
5. Set `view.capabilities` to reflect what's wired.
6. If your backend can run a forward pass: override `setActivePrompt`
   to populate `view.current` with a `CaptureBundle`.
7. Tests: at least one smoke test (hand-construct a minimal valid
   input, verify topology + a read_slice round-trip) and ideally a
   deep test against a real fixture (gated by env var so CI doesn't
   need model downloads).

## Status

| Wave | Backend / change | Status |
|---|---|---|
| substrate | `ModelView` + `TensorHandle` + `CaptureBundle` + `DerivedCache` + `Capabilities` + tri-state + `LoadOptions` + structured intervention refs | shipped |
| HFProxy P1 | `loadCheckpoint` / `getModelInfo` / heartbeat | shipped |
| HFProxy P2 | per-frame samplers (`getAttentionPattern`, `getResidualSummary`, `getQKVStats`); `CaptureBundle` population; Python `/capture` endpoint | shipped |
| HFProxy P3 | intervention (`setAblation` / `setSteering`) | not started |
| HFProxy P4 | WS streams (logit-lens, generate) | not started |
| Wave C | `GgufInspectorEngine` — fully-native GGUF inspector | shipped |
| Wave D | `LlamaCppEngine` — embedded inference + cb_eval attention capture; opt-in via `LLM_ENGINE_BUILD_LLAMA_CPP=ON` | shipped (first cut — attention only; residual / logits / per-arch / intervention deferred) |
| Wave E | `LibtorchEngine` — embedded PyTorch C++ | deferred (planned: `docs/WAVE_E_LIBTORCH_PLAN.md`) |
| Wave F | native_runtime — from-scratch forward pass | research-grade |

See `../gui_cpp/docs/HFPROXY_PLAN.md` and `docs/SUPPORTED_ARCHITECTURES.md`
for per-backend and per-architecture detail.

## Licensing

`llm_engine` itself is MIT (see `LICENSE`). Bundled third-party
headers keep their own licenses (`libs/cpp-httplib/LICENSE`,
`libs/nlohmann_json/LICENSE.MIT`) — both MIT.
