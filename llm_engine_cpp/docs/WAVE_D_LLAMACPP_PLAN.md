# Wave D — `LlamaCppEngine` implementation plan

> Status: not started. Substrate (waves substrate/A/C) provides
> everything this backend plugs into. This doc is the implementation
> contract for the next session that picks it up.

## Goal

Embed `llama.cpp` as an in-process inference runtime and surface its
forward-pass activations / attention / residuals through the
`ModelView` substrate. The result: real model inference + real
activation capture on a laptop, with no Python in the loop. Same
`view().current.load()` UI path as the HFProxy backend.

## Why this matters

`LlamaCppEngine` is the bridge from "we can inspect weights" (Wave C —
GGUF inspector) to "we can inspect what the model is doing right now"
(Wave A — captures). It unlocks:

- Real-time inference on quantised checkpoints that don't fit in fp16
  RAM (Q4_K_M of a 70B model is ~40 GB → fits a single 48 GB GPU).
- Activation capture without a Python service running — single
  binary, embeddable on edge or in CI.
- Architecturally distinct from HFProxyEngine (no HTTP, no JSON,
  zero-copy access to llama.cpp's internal buffers).

## Where llama.cpp goes

Embedded as a git submodule under
`testing/llm_engine_cpp/libs/llama.cpp/`. Pin to a specific commit
(not a branch) so the integration is reproducible — `cb_eval` tensor
names drift across versions and we need a known-good reference.

CMake integration via `FetchContent_Declare` (preferred — keeps
submodule sprawl out of the main repo) or `add_subdirectory` if we
vendor it directly. Either way, the build is opt-in:

```cmake
option(LLM_ENGINE_BUILD_LLAMA_CPP "Build LlamaCppEngine" OFF)
if(LLM_ENGINE_BUILD_LLAMA_CPP)
  # FetchContent llama.cpp, pin to a tag, add target_link_libraries
endif()
```

`OFF` by default so the no-llama.cpp build path (which everyone else
uses) stays fast and dependency-free.

## File layout

New translation units to add:

| File | Purpose |
|---|---|
| `include/llm_engine/llama_cpp_engine.hpp` | public `Model` subclass interface |
| `src/llama_cpp_engine.cpp` | wires llama.cpp's `llama_model` + `llama_context` into a `ModelView` |
| `src/llama_cpp_source.cpp` | `LlamaCppSource : TensorSource` — wraps llama.cpp's already-mmap'd buffers |
| `src/llama_cpp_capture.cpp` | `cb_eval` callback that intercepts per-tensor compute → fills `CaptureBundle` |
| `tests/test_llama_cpp_engine_smoke.cpp` | smoke: load a tiny GGUF, single forward pass, verify capture populated |
| `tests/test_llama_cpp_engine_real.cpp` | deep (env-gated): TinyLlama Q4_0 → topology + a real attention matrix |

## `LlamaCppSource` — zero-copy from llama.cpp's buffers

llama.cpp already mmaps the GGUF file into a `llama_model`. Its
internal `llama_load_tensor` exposes per-tensor `void* data` pointers
into that mmap. Our `LlamaCppSource` wraps those:

```cpp
class LlamaCppSource final : public TensorSource {
public:
    LlamaCppSource(const llama_model* model,           // borrowed; outlives the source
                   const std::string&  tensor_name);

    void pread(std::size_t offset, std::size_t n_bytes, void* out) const override {
        const void* base = llama_get_model_tensor_data(m_model, m_name.c_str());
        std::memcpy(out, static_cast<const std::byte*>(base) + offset, n_bytes);
    }
    std::span<const std::byte> try_mmap() const override {
        // Fast-path: hand back a span over llama.cpp's own buffer.
        return { static_cast<const std::byte*>(
            llama_get_model_tensor_data(m_model, m_name.c_str())),
            llama_get_model_tensor_size(m_model, m_name.c_str()) };
    }
    bool loaded() const override { return true; }      // already in RAM
    // ...
};
```

One `LlamaCppSource` per tensor (because llama.cpp's API is
per-tensor, not whole-file). All sources share a single
`shared_ptr<llama_model*>` wrapper so when the last source goes away,
`llama_free_model` runs.

## `cb_eval` activation capture

llama.cpp exposes a per-tensor callback `cb_eval(struct ggml_tensor*
t, bool ask, void* user_data)` invoked during forward compute. We
register a callback that filters by tensor name and copies the
matching tensors into the active `CaptureBundle`.

Tensor names we care about (from llama.cpp's compute graph):

| `cb_eval` tensor name | Substrate slot |
|---|---|
| `kq-0`, `kq-1`, ... `kq-{L}`                | `attention[(L, *)]` — pre-softmax |
| `kq_soft_max-{L}`                           | `attention[(L, *)]` — post-softmax |
| `attn_norm-{L}`                             | (residual stream pre-attn LN — derived from inp + norm) |
| `attn_out-{L}` / `ffn_inp-{L}`              | `residual_post[L]` |
| `ffn_norm-{L}`                              | (mid-block norm) |
| `result_output` (last layer)                | `logits` |

Name conventions vary by llama.cpp version — pin a version and document
the mapping. A small `TensorNameToSlot` table (per-arch since llama.cpp's
compute graph differs slightly across families) maps the cb_eval name to
the substrate slot.

The callback runs synchronously inside the forward pass on the engine
thread. It must be FAST (copy + index into a map) — no allocation in
the hot path. We pre-allocate a `CaptureBundle` with empty `TensorHandle`
entries for the layers/heads the UI is likely to ask for, and the
callback fills the bytes via a member `InMemoryTensorSource` per entry.

## Threading

```
                       ┌──────────────┐
                       │  UI thread   │ — reads view().current.load()
                       └──────┬───────┘
                              │ setActivePrompt() pushes job
                              ▼
                       ┌──────────────┐
                       │ Engine thread│ — owns llama_context, runs llama_decode
                       │              │   cb_eval fills the in-progress
                       │              │   CaptureBundle on each tensor compute
                       │              │   On EOS / max_tokens, atomic-store
                       │              │   the bundle into view.current
                       └──────────────┘
```

Two locks: `m_model_mu` guards `llama_model*` (only at load/unload
boundaries), `m_ctx_mu` guards `llama_context*` + the in-progress
bundle. UI reads of `view.current` use the existing atomic
shared_ptr — no lock contention.

## Capabilities

```cpp
return Capabilities{
    .has_topology      = true,
    .has_tokenizer     = true,    // llama.cpp's tokenizer is wired
    .has_state_dict    = true,    // via LlamaCppSource per tensor
    .has_attention     = true,    // cb_eval captures
    .has_residual      = true,    // cb_eval captures
    .has_logit_lens    = true,    // we have unembed access; compute per call
    .has_token_stream  = true,    // llama_decode in a loop
    .has_captures      = true,
    .has_intervention  = false,   // future — needs custom forward
    .has_weight_deltas = false,
    .has_training      = false,   // llama.cpp doesn't train
};
```

## Architectures covered

Every architecture llama.cpp supports — at time of writing:

- Llama 1/2/3, Mistral, Mixtral, Qwen2.5, DeepSeek (`LLM_ARCH_LLAMA`)
- Qwen2 family (`LLM_ARCH_QWEN2`)
- Gemma 1/2 (`LLM_ARCH_GEMMA`)
- Phi-2 / Phi-3 (`LLM_ARCH_PHI2`, `LLM_ARCH_PHI3`)
- StarCoder (`LLM_ARCH_STARCODER`)
- Falcon (`LLM_ARCH_FALCON`)
- MPT (`LLM_ARCH_MPT`)
- BLOOM (`LLM_ARCH_BLOOM`)
- GPT-NeoX (`LLM_ARCH_GPTNEOX`)
- OPT (`LLM_ARCH_OPT`)
- Mamba (`LLM_ARCH_MAMBA`) — needs SSM-shaped captures (CaptureBundle extension)
- RWKV (`LLM_ARCH_RWKV6`) — needs recurrent-state captures

llama.cpp does the heavy arch-specific work; we just consume what its
compute graph emits.

## Verification

**Smoke** (CI): hand-construct a tiny GGUF (the GgufInspectorEngine's
test scaffolding already does this), load via LlamaCppEngine, run a
single decode pass against a fixed prompt, verify `view().current.load()
!= nullptr`, verify `attention[(0,0)]` is a non-empty causal-shaped
matrix, verify `logits` exists with the right vocab size.

**Deep** (env-gated, `LLOB_DEEP_LLAMA_GGUF_PATH=/path/to/tinyllama.gguf`):
load a real TinyLlama Q4_0, decode "The capital of France is", verify
`view().topology.nLayers == 22`, sample the top-1 token via
`getOutputLogits()`, expect "Paris" or a close cognate.

## Open questions (decide during implementation)

1. **Per-arch `cb_eval` name maps** — llama.cpp's tensor naming has
   minor variations across architectures (e.g. Gemma's RMSNorm trick,
   Falcon's parallel attn+MLP). The simplest design is a per-arch
   table; the cleanest is a regex that handles the family. Likely the
   table — same as our SUPPORTED_ARCHITECTURES.md GGUF tensor-name map.

2. **Pre-allocated vs. on-demand capture entries** — pre-allocating
   `attention[(L, H)]` for every (L, H) upfront wastes RAM (e.g. 32
   heads × 32 layers × seq² × sizeof(f32) bytes per capture). On-demand
   means the cb_eval callback checks a "did UI request this?" set
   before copying. Probably go on-demand with a small "warm set"
   pre-populated for `layer=0, head=0..N` to cover the default UI view.

3. **Intervention path** — llama.cpp doesn't expose hooks for masking
   heads / steering. Two options: (a) patch llama.cpp itself (fragile
   across versions), (b) custom forward in a fork. Defer this; Wave A's
   HFProxy-based intervention covers the use case until we need
   native intervention performance.

4. **GPU vs. CPU** — llama.cpp builds either way. Build option
   `LLM_ENGINE_LLAMA_CPP_CUDA=ON` mirrors llama.cpp's `LLAMA_CUDA`.
   No substrate impact — `pread` works the same whether the tensor
   data is on host or device (llama.cpp surfaces both as host pointers
   from its perspective; GPU residence is internal).

## Effort estimate

- Source / engine class boilerplate: ~400 LOC
- cb_eval capture wiring: ~300 LOC (arch-aware)
- Build system + FetchContent of llama.cpp: ~100 LOC CMake
- Smoke + deep tests: ~400 LOC
- Documentation: this file + per-arch addenda

Total: ~1500 LOC + the embedded llama.cpp build (~200 KB binary
overhead in release).

Roughly one focused session, contingent on llama.cpp's API not having
moved in a way that breaks the `cb_eval` interface.
