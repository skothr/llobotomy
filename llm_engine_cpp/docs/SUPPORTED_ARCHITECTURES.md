# Supported model architectures

`llm_engine` aims to support every open-source LLM architecture
distributable as GGUF, safetensors, or HuggingFace Transformers. The
substrate (`ModelView` + `TensorRegistry`) is architecture-agnostic;
arch-specific glue lives in three narrow places:

1. **Topology extraction** — pulling layer counts / dims / etc. from a
   checkpoint's metadata block. Per-format mapping (GGUF metadata keys
   vs. HF `config.json` fields vs. safetensors header JSON) lives in
   the backend that parses that format.
2. **Tensor-name normalisation** — translating each format's native
   tensor names (`blk.0.attn_q.weight` in GGUF, `model.layers.0.self_attn.q_proj.weight`
   in HF) into the canonical substrate names (`blocks.0.attn.W_Q.weight`).
   Lives in `tensor_name_map.{hpp,cpp}` (one entry per
   architecture-family, not per-model).
3. **Capture/intervention hookpoints** — where in the forward pass a
   live backend (`LlamaCppEngine`, future `LibtorchEngine`) intercepts
   tensors. Architecture-family-specific; lives next to the inference
   integration.

This document is the matrix that drives all three.

## Architecture families

Most open-source LLMs cluster into a small number of architectural
families. We target the family, not the specific model — adding "Mistral
7B" or "Qwen2.5-14B" is a metadata-keys-and-rope-base entry, not a
parser change.

| Family       | Representative models                                        | Key shape difference                          | GGUF arch key   | HF model_type   |
|--------------|--------------------------------------------------------------|-----------------------------------------------|-----------------|-----------------|
| **Llama**    | Llama 1/2/3, Mistral 7B, Qwen 1/2/2.5, Phi-3, DeepSeek, Yi  | RMSNorm + SwiGLU MLP + RoPE; GQA optional     | `llama`         | `llama`         |
| **Qwen2**    | Qwen2, Qwen2.5 (very Llama-like; separate GGUF arch tag)    | Almost identical to Llama                     | `qwen2`         | `qwen2`         |
| **Gemma**    | Gemma 1/2, CodeGemma                                         | RMSNorm with `+1` trick; SwiGLU; tied embed   | `gemma` / `gemma2` | `gemma`      |
| **Phi**      | Phi-2, Phi-3 (note: Phi-3 is Llama-family in GGUF)          | Phi-2: distinct (LayerNorm + GELU)            | `phi2` / `llama`| `phi`           |
| **Mixtral**  | Mixtral 8x7B, Qwen2-MoE                                      | Llama + sparse-MoE expert routing             | `llama` + MoE keys | `mixtral`     |
| **GPT-2**    | GPT-2, GPT-J, Pythia (sort of)                               | LayerNorm + GELU MLP + learned pos embed      | `gpt2`          | `gpt2`          |
| **GPT-NeoX** | GPT-NeoX-20B, Pythia                                         | LayerNorm + GELU + parallel attn+MLP          | `gptneox`       | `gpt_neox`      |
| **Falcon**   | Falcon 7B/40B/180B                                           | LayerNorm + GELU + multi-query attn (one-K)   | `falcon`        | `falcon`        |
| **MPT**      | MPT 7B/30B                                                   | LayerNorm + GELU + alibi-position             | `mpt`           | `mpt`           |
| **BLOOM**    | BLOOM, BLOOMZ                                                 | LayerNorm + GELU + alibi                      | `bloom`         | `bloom`         |
| **StarCoder**| StarCoder, StarCoder2, SantaCoder                            | GPT-2-ish + multi-query attn                  | `starcoder2`    | `starcoder2`    |
| **OPT**      | OPT-125M..OPT-66B                                            | GPT-2-shaped, LayerNorm + GELU                | `opt`           | `opt`           |
| **Mamba**    | Mamba, Mamba2                                                | State-space (not transformer) — separate path | `mamba`         | `mamba`         |
| **RWKV**     | RWKV-4/5/6                                                   | Linear recurrent (not transformer)            | `rwkv`          | `rwkv`          |

**Recurrent families (Mamba, RWKV)** have fundamentally different
hookpoint vocabulary — the substrate's `attention[(L,H)]` shape doesn't
apply. We handle them by extending `CaptureBundle` with optional
recurrent-state fields once a backend needs them, rather than forcing
the transformer shape onto them.

## Canonical-name scheme (recap)

Substrate-side, every tensor name follows
`[A-Za-z0-9_.]+(/[A-Za-z0-9_.]+)*` (enforced at `TensorRegistry::insert`).
The convention this codebase uses for transformer-family models:

```
tok_embeddings.weight
blocks.{L}.attn.W_Q.weight
blocks.{L}.attn.W_K.weight
blocks.{L}.attn.W_V.weight
blocks.{L}.attn.W_O.weight
blocks.{L}.attn.b_Q.weight        (optional — some archs)
blocks.{L}.attn.b_K.weight
blocks.{L}.attn.b_V.weight
blocks.{L}.attn.b_O.weight
blocks.{L}.mlp.W_gate.weight      (SwiGLU/GeGLU — Llama family)
blocks.{L}.mlp.W_up.weight
blocks.{L}.mlp.W_down.weight
blocks.{L}.mlp.W_in.weight        (GELU MLP — GPT-2 family)
blocks.{L}.mlp.W_out.weight
blocks.{L}.ln1.weight             (input layernorm — pre-attn)
blocks.{L}.ln1.bias               (optional)
blocks.{L}.ln2.weight             (post-attn layernorm — pre-MLP)
blocks.{L}.ln2.bias               (optional)
ln_f.weight                       (final layernorm)
ln_f.bias                         (optional)
lm_head.weight                    (unembed; may be tied to tok_embeddings)
pos_embeddings.weight             (learned-position archs only)
```

A backend's tensor-name normaliser is a `vector<pair<regex, format>>`
table keyed by architecture-family. The GGUF backend lives in
`gguf_inspector_engine.cpp`; the HF / libtorch backends carry their own
(largely the same — HF names are the de facto source of truth for the
canonical scheme).

## Per-family GGUF metadata keys

GGUF stores per-architecture metadata under keys prefixed with the
architecture name. Example for `llama`:

```
general.architecture           = "llama"
llama.block_count              = 32
llama.attention.head_count     = 32
llama.attention.head_count_kv  = 8           (GQA; absent ⇒ same as head_count)
llama.embedding_length         = 4096        (d_model)
llama.feed_forward_length      = 14336       (d_mlp)
llama.context_length           = 8192        (max_pos)
llama.rope.freq_base           = 500000.0
llama.rope.dimension_count     = 128         (d_head)
llama.attention.layer_norm_rms_epsilon = 1e-05
tokenizer.ggml.bos_token_id    = 128000
tokenizer.ggml.eos_token_id    = 128001
tokenizer.chat_template        = "..."
```

Other families substitute the prefix (`qwen2.block_count`,
`gemma.block_count`, `falcon.block_count`, …). Most use the same keys
under their own prefix; deviations (e.g. `gpt2.context_length` doesn't
exist — GPT-2 uses learned positions) are handled per-family.

## Adding a new architecture family

1. Add a row to the table above.
2. Add a metadata-key block in `gguf_inspector_engine.cpp` for any
   prefix that diverges from the Llama-family pattern.
3. Add a tensor-name-normalisation regex block (raw → canonical) keyed
   by the GGUF arch tag.
4. Add a smoke test: hand-construct a minimal GGUF with the new arch's
   keys, parse, verify `view().topology` and a few `view().tensors`
   entries.

If the architecture is **transformer-family** with standard
attn+MLP+layernorm, that's the whole change. If it's recurrent (Mamba,
RWKV) or otherwise structurally different, you also need to extend
`CaptureBundle` with the new hookpoint shape (e.g. SSM state buffers)
and document why the standard attention/residual maps are empty for
this family.

## Validation strategy

Smoke tests (run on every commit) hand-construct minimal GGUF buffers
covering each supported family. Deep tests (gated by env var) load real
fixture models from `testing/.cache/models/` and verify:

- topology matches expected (compared against the model's HF config.json)
- canonical tensor names are present (W_Q/W_K/W_V/W_O for every layer)
- one weight slice round-trips through `read_slice` and matches the
  same slice loaded via Python (`gguf_reader.py:read_tensor_numpy`).

The deep tests don't run in CI by default — they require model
downloads. Run locally with `ctest -L deep` when adding a new family.
