# llm_engine — C++ backend-abstraction library for LLM interpretability tooling

`llm_engine` is a small, framework-agnostic C++23 library that defines the
data surface every interpretability UI panel needs (architecture map, live
activations, attention patterns, residual flow, logit lens, weight stats,
training metrics, etc.) as a single abstract `Model` interface, plus a
handful of concrete implementations:

- **`llmengine::MockModel`** — deterministic fake data, enabled at compile
  time via `LLM_ENGINE_USE_MOCK_DATA=ON`.  Useful for UI development, demo
  builds, and screenshot fixtures.  When the flag is `OFF`, every method
  returns the no-data sentinel for its type — so a release build can't
  silently ship mock numbers.
- **`llmengine::HFProxyEngine`** — thin wrapper over the FastAPI backend
  at `testing/gui/backend/` (the same server that powers the existing
  React frontend).  See `../gui_cpp/docs/HFPROXY_PLAN.md` for the
  per-method status table.

The reference consumer is **[`testing/gui_cpp/`](../gui_cpp/)**, the
native Dear ImGui interpretability bench.  But the library has no UI
dependency — any host that wires up the `drainEngineLogs()` log fan-in
can use it.

## Architecture

```
┌─────────────── consumers ───────────────┐
│ gui_cpp/ (Dear ImGui)   future tools…   │
└────────────────┬────────────────────────┘
                 │  llmengine::Model&
┌────────────────┴────────────────────────┐
│           llm_engine (this lib)         │
│                                         │
│   include/llm_engine/                   │
│     log.hpp           Severity / LogEntry
│     model.hpp         Model + DTOs       │
│     hf_proxy_engine.hpp                  │
│                                         │
│   src/                                  │
│     log.cpp                             │
│     mock_model.cpp                      │
│     hf_proxy_engine.cpp                 │
│                                         │
│   libs/  (private)                      │
│     cpp-httplib/    HTTP client (MIT)   │
│     nlohmann_json/  JSON       (MIT)    │
└─────────────────────────────────────────┘
```

The `Model` interface enforces a few discipline rules:

- **Per-frame samplers MUST be cheap.** UI threads call them ~60 times
  a second; backends cache aggressively.
- **Methods never throw on missing data.** They return the no-data
  sentinel (`kNoFloat`, `kNoInt`, empty vector, `""`) and the UI prints
  a placeholder.  `noexcept(false)` is reserved for genuine IO failure.
- **Mutators default to no-op** on the base interface — a backend that
  doesn't implement a hook isn't broken; the corresponding UI control
  just won't do anything.

For the full architectural contract (lifecycle, threading, per-area
requirements, error handling), see
[`../gui_cpp/docs/ENGINE_API.md`](../gui_cpp/docs/ENGINE_API.md).

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

Or as a sub-project:

```cmake
add_subdirectory(/path/to/llm_engine_cpp ${CMAKE_BINARY_DIR}/llm_engine_cpp)
target_link_libraries(your_app PRIVATE llmengine::llm_engine)
```

`LLM_ENGINE_USE_MOCK_DATA` (default `ON`) controls MockModel's behaviour
as described above.

## Status

Phase 1 (shipped):
- `loadCheckpoint` / `unloadCheckpoint` round-trip via FastAPI.
- `drainEngineLogs` fan-in for HTTP errors.
- `MockModel` with the full DTO surface (~50 methods).

Phase 2 (in progress):
- `getModelInfo` cache from `/api/sessions/{n}/info`.
- `getCurrentTokens` via tokenize endpoint.
- Per-frame sampler thread + cache layer; first user is `getAttentionPattern`.

See `../gui_cpp/docs/HFPROXY_PLAN.md` for the per-method roadmap.

## Licensing

`llm_engine` itself is MIT (see `LICENSE`).  Bundled third-party headers
keep their own licenses (`libs/cpp-httplib/LICENSE`,
`libs/nlohmann_json/LICENSE.MIT`) — both MIT.
