# How to add a new backend to llm_engine

Worked example: walk through what's involved in implementing a new
`Model` subclass that pulls real data from a new storage format /
runtime. The substrate (`ModelView`, `TensorHandle`, `TensorSource`,
`CaptureBundle`, `DerivedCache`, `Capabilities`) does most of the work
— you supply two pieces of glue and the wire-up.

## The shape of a backend

Every backend has the same anatomy:

```
class YourEngine : public MockModel {       // inherit MockModel so unwired hooks fall through to mock data
    struct Impl;                            // PIMPL — keep heavy includes out of the header
    std::unique_ptr<Impl> m_impl;
public:
    YourEngine(... constructor args ...);
    ~YourEngine() override;

    // Wired hooks — start with these three, add more as needed
    CheckpointResult loadCheckpoint(std::string_view path,
                                    const LoadOptions& opts = {}) override;
    void             unloadCheckpoint() override;
    const ModelView& view() const override;
    Capabilities     getCapabilities() const override;
    std::vector<LogEntry> drainEngineLogs() override;
};
```

Inside `Impl` you hold:

- A `ModelView view` — the canonical data structure consumers read.
- A `std::mutex mu` — guards writes to the view's mutable parts.
- A FIFO of `LogEntry pending_logs` — drained by `drainEngineLogs`.
- Whatever your runtime needs (an HTTP client, an `int fd`, a
  `std::thread worker`, …).

## Step 1 — Implement a `TensorSource` for your storage

Subclass `llmengine::TensorSource`. Three required methods:

```cpp
class YourSource : public TensorSource {
public:
    // Read raw bytes from your storage layer.  Never throws —
    // out-of-range reads return without touching `out`.
    void pread(std::size_t offset, std::size_t n_bytes, void* out) const override;

    // Total size of the source's data region.
    std::size_t size_bytes() const override;

    // Override only if your bytes are already in addressable RAM
    // (mmap'd file, in-memory buffer).  Default false is correct for
    // HTTP / PRNG / anything that computes or fetches per call.
    bool loaded() const override { return /* true if mmap'd, else false */; }

    // Optional fast-path: return a span over your in-RAM bytes.  Lets
    // callers skip the pread() copy on hot paths.  Default returns
    // empty span; only override when you actually have an mmap.
    std::span<const std::byte> try_mmap() const override { return {}; }
};
```

The substrate's [`InMemoryTensorSource`](../include/llm_engine/tensor_source.hpp)
is a 30-LOC reference implementation if you want a starting template.
[`Mulberry32Source`](../src/tensor_source.cpp) is the same shape but
stateless (each `pread` re-runs the PRNG) and shows the "lazy,
computed-on-demand" pattern.

**One source per file (or per logical storage unit).** All
`TensorHandle`s for tensors in the same `.gguf` share a single
`shared_ptr<YourSource>` — the source's destructor (close fd / unmap /
release HTTP connection) runs automatically when the last handle is
dropped (typically at `view.clear()` in `unloadCheckpoint`).

## Step 2 — Parse the format header and populate `ModelView`

Inside `loadCheckpoint`:

```cpp
Model::CheckpointResult YourEngine::loadCheckpoint(std::string_view path,
                                                   const LoadOptions& opts) {
    auto parsed = parse_your_format(std::string(path), opts);
    if (!parsed) return {false, parsed.error()};

    std::lock_guard lk(m_impl->mu);

    // 1. Provenance — who this is, where it came from.
    m_impl->view.provenance = {
        .path         = std::string(path),
        .format       = "your-format",
        .content_hash = parsed->content_hash,  // "" if not computed
        .source_label = parsed->source_label,
    };

    // 2. Topology — fixed across the session.
    m_impl->view.topology = {
        .name        = parsed->model_name,
        .nLayers     = parsed->n_layers,
        .nHeads      = parsed->n_heads,
        // ... etc, sentinel (kNoInt / kNoFloat) when not known
    };

    // 3. Tokenizer — encode/decode are std::function slots.
    m_impl->view.tokenizer.bos_token     = parsed->bos_token;
    m_impl->view.tokenizer.eos_token     = parsed->eos_token;
    m_impl->view.tokenizer.chat_template = parsed->chat_template;
    m_impl->view.tokenizer.encode = [this](std::string_view s) {
        return your_encode_impl(s);
    };
    m_impl->view.tokenizer.decode = [this](TokenId id) {
        return your_decode_impl(id);
    };

    // 4. Tensors — one TensorHandle per tensor, all sharing the source.
    auto source = std::make_shared<YourSource>(std::move(parsed->raw));
    for (const auto& t : parsed->tensors) {
        TensorHandle h;
        h.source      = source;
        h.name        = canonicalise(t.raw_name, parsed->architecture);  // see SUPPORTED_ARCHITECTURES.md
        h.dtype       = t.dtype;
        h.shape       = t.shape;
        h.byte_offset = t.byte_offset;
        h.byte_length = t.byte_length;
        m_impl->view.tensors.insert(std::move(h));  // throws on non-canonical name
    }

    // 5. Capabilities — what bits this backend has wired.  Mirror
    //    written so path-API consumers can read `capabilities/has_X`.
    m_impl->view.capabilities = Capabilities{
        .has_topology   = true,
        .has_tokenizer  = m_impl->view.tokenizer.has_encode(),
        .has_state_dict = true,
        // ... etc
    };

    return {true, ""};
}
```

`unloadCheckpoint` is one line:

```cpp
void YourEngine::unloadCheckpoint() {
    std::lock_guard lk(m_impl->mu);
    m_impl->view.clear();   // resets every field including capabilities
}
```

## Step 3 — Optional: serve live forward-pass data

If your backend can run a forward pass, override `setActivePrompt`:

```cpp
void YourEngine::setActivePrompt(std::string_view prompt) {
    // The prompt is a borrow; copy it for async use.
    std::string p(prompt);

    // Push onto the worker thread's queue.  The worker:
    //   1. runs the forward pass under the engine's own lock
    //   2. captures activations / attention / residuals into a
    //      CaptureBundle (handles backed by InMemoryTensorSource)
    //   3. publishes via view.current.store(...)
    m_impl->worker.enqueue(p);
}
```

The worker thread does its own threading. UI reads via
`view().current.load()` lock-free. See `hf_proxy_engine.cpp` for the
full pattern (heartbeat + capture-poll FSM).

## Step 4 — Wire the factory

In `testing/gui_cpp/src/main.cpp`, the `LLOB_BACKEND` env var picks the
backend. Add a branch for your engine:

```cpp
else if (choice == std::string("your_name")) {
    backend = std::make_unique<llmengine::YourEngine>(/* ctor args from env */);
}
```

## Step 5 — Tests

Add two tests per backend:

**Smoke** (`tests/test_your_engine.cpp`, registered with the `smoke`
label): constructs the engine against a tiny in-test fixture or a
mocked source. Verifies (a) topology populates, (b) at least one
`view().tensors` entry round-trips through `read_slice`, (c)
`getCapabilities()` matches `view().capabilities`.

**Deep** (`tests/test_your_engine_real.cpp`, registered with the
`deep` label, gated by env var so CI doesn't need fixtures): loads a
real model fixture, exercises the same set against known-correct
values from a reference implementation (e.g. Python reads via
`testing/llm_surgeon/`).

Both register in `tests/CMakeLists.txt`:

```cmake
add_executable(test_your_engine test_your_engine.cpp)
target_link_libraries(test_your_engine PRIVATE llmengine::llm_engine)
add_test(NAME your_engine_smoke COMMAND test_your_engine)
set_tests_properties(your_engine_smoke PROPERTIES LABELS "smoke")
```

## Reference implementations

Look at these in order of complexity:

| Backend | Storage | What it shows |
|---|---|---|
| [`MockModel`](../src/mock_model.cpp) | `Mulberry32Source` (PRNG) | Substrate-only — no I/O, no async. Pure populator pattern. |
| [`HFProxyEngine`](../src/hf_proxy_engine.cpp) | HTTP (no `TensorSource` yet) | PIMPL + worker thread + log fan-in + mutex-guarded view writes. |
| [`GgufInspectorEngine`](../src/gguf_inspector_engine.cpp) | mmap'd file | Full read-only native backend; header parser + `GgufSource` + arch-aware tensor-name normalisation. |

## Discipline checklist before opening a PR

- Builds clean under `LLM_ENGINE_USE_MOCK_DATA=ON` AND `=OFF`.
- `-Wall -Wextra -Wpedantic` clean (the library target already has these on).
- Smoke tests pass on both build modes.
- Capabilities advertisement is honest — every `true` bit is actually wired.
- `view.clear()` is called from `unloadCheckpoint` (and nowhere else
  outside lifecycle — the substrate doesn't have "soft reset" semantics).
- New tensor names match the canonical regex `[A-Za-z0-9_.]+(/[A-Za-z0-9_.]+)*`.
  Architectures whose native names don't fit need a normaliser; see
  `SUPPORTED_ARCHITECTURES.md`.
- Threading: backend's worker thread joined in dtor; no detached threads
  outliving the engine.
- `drainEngineLogs` flushes the local FIFO and clears it; never
  emits to the global logger directly.
- Doc-block at the top of each new `.hpp` explaining "why this exists"
  in the substrate's vocabulary.
