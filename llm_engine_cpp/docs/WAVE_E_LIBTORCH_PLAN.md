# Wave E — `LibtorchEngine` implementation plan

> Status: deferred per the substrate plan. This doc is the
> implementation contract for when a libtorch-shaped use case actually
> appears.

## Goal

Embed libtorch (PyTorch's C++ API) and serve a real model's
weights + forward pass through the `ModelView` substrate. The
distinguishing feature vs. `LlamaCppEngine`: libtorch loads HF
safetensors directly (no GGUF conversion), exposes the full ATen
tensor API, and supports custom forward hooks for fine-grained
intervention — but you have to write the architecture's forward pass
in C++.

## Why this is deferred

Three reasons:

1. **`HFProxyEngine` covers most of the use case.** Anyone who needs
   HF-shaped weight access can spin up the FastAPI backend and use
   the proxy. The Python-in-the-loop overhead is ~10 ms per HTTP
   roundtrip — acceptable for interactive UI work.
2. **Writing a transformer forward pass in C++ is ~2000 LOC per
   architecture family**, and libtorch's idiomatic patterns differ
   enough from `nn.Module` that you can't trivially port HF's
   modeling code. You'd be reimplementing what HF already wrote,
   one family at a time.
3. **GGUF + llama.cpp covers local-inference use case.** Quantised
   inference, activation capture via `cb_eval`, broad architecture
   support — Wave D delivers all of this. The libtorch backend's
   advantage (full fp16 tensors, ATen API for arbitrary ops) only
   matters for in-process surgical interventions that go beyond what
   `cb_eval` can do (e.g. patching individual weight rows during
   forward, computing custom probe outputs inline).

When you'd want libtorch:

- You need to run a training-shaped workflow (loss + backward) in
  the engine, not just inference. llama.cpp doesn't train; HFProxy
  doesn't expose backward. libtorch does.
- You need surgical intervention at a granularity llama.cpp's
  `cb_eval` can't provide (e.g. "patch the W_Q[head=3] subspace
  with a steering vector" — needs ATen slicing inside the forward).
- You need zero-copy tensor access for very large models without
  Python in the loop.

## When you'd implement it

If the user starts asking for "actually train a small probe on this
checkpoint in the engine" or "do real-time activation patching at
the head-subspace level", Wave E lands. Until then, the substrate's
design (TensorSource ABC + CaptureBundle's open-shape) is enough to
fold it in without retrofitting.

## File layout (when implemented)

| File | Purpose |
|---|---|
| `include/llm_engine/libtorch_engine.hpp` | `class LibtorchEngine : public Model` |
| `src/libtorch_engine.cpp` | ATen tensor → ModelView wiring |
| `src/libtorch_source.cpp` | `LibtorchSource : TensorSource` — wraps a `torch::Tensor` |
| `src/libtorch_arch_*.cpp` | one per architecture family (Llama, Qwen2, …) — the forward pass |
| `src/safetensors_parser.cpp` | 80-LOC safetensors header parser |

## Architecture coverage

This is the hard part. Each family needs its own ~300-LOC forward
implementation in C++. Realistic first cut: **Llama family only**
(covers Llama 1/2/3, Mistral, Qwen2.5, DeepSeek, Phi-3). Adding
Gemma is +200 LOC (RMSNorm +1 trick + SwiGLU); GPT-2 family is +400
LOC (different attention shape + learned positions).

## Build

```cmake
option(LLM_ENGINE_BUILD_LIBTORCH "Build LibtorchEngine" OFF)
if(LLM_ENGINE_BUILD_LIBTORCH)
  find_package(Torch REQUIRED)
  # ... target_sources / target_link_libraries with Torch::Torch
endif()
```

Same opt-in pattern as `LLM_ENGINE_BUILD_LLAMA_CPP`. Default-OFF; the
libtorch dependency is heavy (>500 MB build).

## Tests

Same smoke + deep pattern as the other backends. Smoke: hand-construct
a tiny safetensors buffer, load via LibtorchEngine, verify topology
+ tensor enumeration. Deep: load a real model from
`testing/.cache/models/`, run a forward pass, verify the substrate
captures match a Python reference (`torch.matmul` from
`testing/llm_surgeon/`).

## Effort estimate

- Libtorch CMake integration: ~50 LOC
- Safetensors parser: ~80 LOC
- LibtorchSource: ~80 LOC
- Engine skeleton + ModelView populator: ~400 LOC
- Llama-family forward pass: ~600 LOC
- cb_eval-style activation capture (via torch::nn::Module forward hooks): ~200 LOC
- Smoke + deep tests: ~300 LOC

Total: ~1700 LOC for Llama-family only. Each additional family is
~200-500 LOC depending on how different it is.

## Open question (decide if/when implementing)

Pick one:

**Option A — Reimplement the forward pass in C++.** Maximum control,
matches the plan. ~600 LOC per family.

**Option B — TorchScript trace the HF model in Python, save the
.pt, libtorch loads + executes the traced graph.** Far less C++ to
write, but TorchScript traces are brittle (control flow + dynamic
shapes break it) and you lose easy intervention access (can't reach
into the traced graph to mask a head).

Recommend **A** despite the LOC cost — the whole point of Wave E
over HFProxy is the C++-side surgical access. Option B gives you a
heavier llama.cpp without the activation-capture story.

## Status (2026-05-13): STILL DEFERRED — by design, not abandoned

Reaffirmed during the Wave D extension wave (state_dict, token
streaming, head ablation).  LlamaCppEngine now covers:

| Capability | LlamaCppEngine (today) | LibtorchEngine (if built) |
|---|---|---|
| HF safetensors load | no (GGUF only) | yes |
| Local inference | yes (CUDA via llama.cpp) | yes |
| Activation capture | yes (cb_eval, all standard-attention archs) | yes (forward hooks) |
| State dict enumeration | yes (parallel GGUF parse → view.tensors) | yes |
| Token streaming | yes (greedy gen loop, 24 tokens) | yes |
| Head ablation | yes (cb_eval GPU write-back, real fwd-pass mutation) | yes |
| Steering vector | partial (recorded in view.surgery; not yet wired) | yes |
| Backward pass / training | no | yes |
| Custom forward (per-row weight patching) | no | yes |

The gap: **training + truly surgical interventions** (e.g., patching
a single subspace of W_Q during forward).  Neither has a current
use case in the llm-surgeon GUI, and adding 1700 LOC for hypothetical
future use violates YAGNI hard.

Triggers that would justify the cost:
- A workspace that needs to run a probe training loop in-engine
  (current llm_surgeon Python toolkit handles this fine).
- A workspace that needs per-row weight surgery during forward,
  beyond cb_eval's tensor-level write-back.
- A user-facing request specifically for safetensors-direct loading
  without GGUF conversion.

Until then, this plan stays as the implementation contract.
