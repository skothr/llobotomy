#pragma once
//
// Model — data backend for every llobotomy workspace.
//
// Each method below is a [DATA HOOK]: it tells the engine what the UI needs
// at that point in time.  Model itself is the canonical empty backend —
// every getter has a default body returning the no-data sentinel for its
// type.  Concrete backends (HFProxyEngine, GgufInspectorEngine,
// LlamaCppEngine, ...) inherit Model directly and override only the
// getters they can actually fulfil from their data source.  Anything they
// don't override stays honestly empty in the UI.
//
// `MockModel` (declared below, defined in mock_model.cpp) is a separate
// opt-in implementation that returns deterministic fake data when
// `LLOB_USE_MOCK_DATA` is defined.  It exists for two purposes only:
//   (a) UI demo / screenshot mode, selected via `LLOB_BACKEND=mock`;
//   (b) a developer baseline when prototyping new workspace code.
// Real backends do NOT inherit from MockModel — that pattern caused
// silent mock-data leakage when a getter wasn't overridden.  The honest
// empty default + an explicit Capabilities advertisement is the contract.
//
// Conventions:
//   * Vectors return `{}` when no data is available — UI shows empty plot.
//   * Floats return `kNoFloat` (NaN) when no data — UI prints "—".
//   * Ints  return `kNoInt`  (-1)  when no data — UI prints "—".
//   * Strings return `""` — UI prints "—".
//   * Methods are declared `noexcept(false)` only where IO failure is
//     genuinely possible (tensor reads etc.); pure data accessors should
//     not throw.
//
// Per-workspace docs:
//   docs/MODEL_HOOKS.md catalogues each method, its callers, and what a
//   real backend should plumb.

#include "llm_engine/log.hpp"

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace llmengine {

// Forward declaration — the unified data structure lives in
// model_view.hpp (which includes this file to get the DTOs).  Backends
// own a ModelView and expose it via Model::view(); consumers prefer
// typed access (`m.view().topology.nLayers`) over the per-DTO getters
// kept below for source compatibility.
struct ModelView;

// ── Sentinel values for "no data yet" ─────────────────────────────────────
inline constexpr float        kNoFloat = std::numeric_limits<float>::quiet_NaN();
inline constexpr int          kNoInt   = -1;
inline constexpr std::int64_t kNoSize  = -1;

// ── Mulberry32 PRNG (kept here so MockModel + a few widget-side mock
//    helpers share one deterministic source). ──────────────────────────────
struct Mulberry32 {
    std::uint32_t a;
    explicit Mulberry32(std::uint32_t seed) : a(seed) {}
    float next() {
        a += 0x6D2B79F5u;
        std::uint32_t t = a;
        t = (t ^ (t >> 15)) * (t | 1u);
        t ^= t + ((t ^ (t >> 7)) * (t | 61u));
        return static_cast<float>((t ^ (t >> 14)) >> 0) / 4294967296.0f;
    }
};

enum class HeadBias { Diag, Prev, First, Broad, Induction };

// ──────────────────────────────────────────────────────────────────────────
// Architecture / weights — checkpoint-static (mostly)
// ──────────────────────────────────────────────────────────────────────────

// Topology of the loaded checkpoint.  Returned by Model::getModelInfo()
// after loadCheckpoint() succeeds; populated from the backend's session-info
// endpoint or a header-parse for native backends.  Field names match the
// gui_cpp consumer's existing AppState layout so workspace code can keep
// reading `s.model.nLayers` etc. without churn.
//
// Optional fields default to `kNo…` sentinels when the backend doesn't
// expose them — e.g. Ollama / GGUF backends may not surface a chat
// template, max_position_embeddings, or rope theta.
struct ModelInfo {
    std::string  name;                              // model_id or path label
    int          nLayers       = kNoInt;
    int          nHeads        = kNoInt;
    int          nKvHeads      = kNoInt;            // grouped-query attn (None ⇒ == nHeads)
    int          dModel        = kNoInt;            // hidden_size
    int          dHead         = kNoInt;            // dModel / nHeads
    int          dMlp          = kNoInt;            // intermediate_size
    int          vocab         = kNoInt;
    int          maxPos        = kNoInt;            // max_position_embeddings
    float        ropeTheta     = kNoFloat;
    std::int64_t totalParams   = kNoSize;
    std::string  chatTemplate;                      // Jinja-like; "" if none
    std::string  bosToken;                          // "" if none
    std::string  eosToken;                          // "" if none
};

struct ParamBreakdownRow {
    std::string component;     // "W_Q + W_K + W_V"
    float       params_M;      // millions
    float       pct;           // 0..1
    const char* tone;          // "info" | "warn" | "muted" | ""
};

struct LiveActivations {
    float attn_out_norm   = kNoFloat;
    float mlp_out_norm    = kNoFloat;
    float resid_post_norm = kNoFloat;
    float attn_entropy_avg = kNoFloat;   // nats
    int   dead_neurons     = kNoInt;
    int   total_neurons    = kNoInt;
};

struct TensorMeta {
    std::string      name;            // "blocks.8.attn.W_Q.weight"
    std::string      dtype;           // "fp16"
    std::vector<int> shape;           // {768, 768}
    std::vector<int> stride;          // {768, 1}
    bool             contiguous = true;
    std::string      device;          // "cuda:0"
    std::int64_t     size_bytes = kNoSize;
};

struct TensorStats {
    float min       = kNoFloat;
    float max       = kNoFloat;
    float mean      = kNoFloat;
    float std       = kNoFloat;
    float frobenius = kNoFloat;
    float op_norm   = kNoFloat;        // largest singular value
    float inf_norm  = kNoFloat;
    int   rank_eff  = kNoInt;
    int   rank_full = kNoInt;
};

struct DiffStats {
    float              frobenius_norm = kNoFloat;
    float              cosine         = kNoFloat;
    std::vector<float> top_k_dirs;
};

// ──────────────────────────────────────────────────────────────────────────
// Inference — per-token, per-layer
// ──────────────────────────────────────────────────────────────────────────

struct ResidualContribution { float attn = kNoFloat, mlp = kNoFloat; };

struct ResidualSummary {
    float attn_out_norm = kNoFloat;
    float mlp_out_norm  = kNoFloat;
    float resid_norm    = kNoFloat;
    float cos_prev      = kNoFloat;
    float kurtosis      = kNoFloat;
    int   rank_eff      = kNoInt;
    int   rank_full     = kNoInt;
};

struct LogitLensRow {
    int         layer;
    std::string top1;
    float       p1;
    std::string top2;
    float       p2;
    float       entropy;        // nats
    bool        is_resolved;    // visual cue: layer where top-1 stabilises
};

struct LogitDist {
    std::string token;
    float       prob;
    float       delta;          // 0 if no baseline; +/− vs base run
    bool        selected;
};

struct MlpFeatureActivation {
    int   idx;       // feature index in MLP basis (or SAE)
    float value;     // signed activation
};

struct ProbeEntry {
    std::string type;            // "L" / "P" / "S" — short tag for Pill
    std::string type_tone;       // "accent" | "warn" | "dim"
    std::string name;            // "linear/refusal_dir"
    std::string location;        // "L08.resid_post"
    float       accuracy;        // 0..1; kNoFloat → "—"
    bool        type_solid;      // pill style
};

struct SteeringConfig {
    bool        active   = false;
    std::string source;          // "\"refusal\" prompts (n=128)"
    std::string layer;           // "L08.resid_post"
    float       alpha    = kNoFloat;
    float       cos_sim  = kNoFloat;
};

// Sampler configuration consumed by Model::setSamplerConfig.  Describes
// the full llama.cpp sampler chain.  All fields are interpreted by the
// backend's setter — the substrate just stores the description.
//
// Default-constructed value reproduces greedy sampling (the engine's
// pre-config behavior) so unspecified consumers keep the old semantics.
struct SamplerConfig {
    // Top-level method.  When Greedy, the chain fields below are
    // ignored (LlamaCppEngine uses llama_sampler_init_greedy directly).
    // When Sampling, the chain is built per the enabled stages.
    enum class Method { Greedy, Sampling };
    Method method = Method::Greedy;

    // Truncation / temperature stages — applied in order.  Each is
    // disabled when at its default ("identity") value:
    //   * top_k       == 0    → no top-k truncation
    //   * top_p       == 1.0  → no nucleus truncation
    //   * min_p       == 0.0  → no min-probability truncation
    //   * temperature == 1.0  → no temperature scaling
    int   top_k       = 0;
    float top_p       = 1.0f;
    float min_p       = 0.0f;
    float temperature = 1.0f;

    // Mirostat — 0 = off, 1 = v1, 2 = v2.  When enabled, mirostat owns
    // the final sampling step (replaces the temp+dist tail of the chain).
    int   mirostat     = 0;
    float mirostat_tau = 5.0f;
    float mirostat_eta = 0.1f;

    // RNG seed used by the dist sampler when Method::Sampling is active.
    // Ignored under Greedy.
    std::uint32_t seed = 0xDEADBEEFu;
};

// ──────────────────────────────────────────────────────────────────────────
// Attention
// ──────────────────────────────────────────────────────────────────────────

struct QKVStats {
    float q_norm        = kNoFloat;
    float k_norm        = kNoFloat;
    float v_norm        = kNoFloat;
    float attn_to_bos   = kNoFloat;
    float attn_to_self  = kNoFloat;
    float attn_to_prev  = kNoFloat;
};

struct HeadStatRow {
    std::string metric;          // "entropy / token"
    std::string value_str;       // "1.14 nats" — pre-formatted with units
    float       vs_corpus;       // 0..1; bar fraction
    const char* tone;            // "" | "warn" — value color
};

struct PatchSourceState {
    std::string source_prompt;
    int         target_pos = kNoInt;
    std::string component;       // "attn_out"
    float       effect_nats = kNoFloat;
};

// ──────────────────────────────────────────────────────────────────────────
// Probes / SAE
// ──────────────────────────────────────────────────────────────────────────

struct FeatureSummary {
    int         id;
    std::string label;
    int         layer;
    float       l0_sparsity;
    float       acts_mean;
    std::string type;           // "SAE" | "linear" | "logit" | "mlp"
};

struct FeatureCard {
    int         id;
    std::string layer_str;       // "L08.mlp_out"
    std::string type_str;        // "SAE feature (top-K, k=64)"
    float       l0_sparsity;
    int         fires_per_million;
    float       mean_act_when_firing;
    std::string decoder_down_str;
    std::string decoder_up_str;
};

struct FeatureExample {
    float       score;
    std::string pre;
    std::string highlight;
    std::string post;
};

struct CoFiringEntry {
    int         id;
    std::string label;
    float       cos_sim;
};

struct SAETrainingMetrics {
    std::string sae_id;          // "L08.mlp_out"
    int         step;
    int         total_steps;
    float       recon_loss;
    float       l0_sparsity;
    int         dead_features;
    int         total_features;
    float       expl_var;
    std::vector<float> recon_loss_history;
    std::vector<float> l0_sparsity_history;
    std::vector<float> dead_features_history;
    std::vector<float> expl_var_history;
};

struct ProbeTrainState {
    bool        training      = false;
    int         step          = 0;
    int         total_steps   = 0;
    float       train_acc     = kNoFloat;
    float       val_acc       = kNoFloat;
    float       gap           = kNoFloat;
    std::vector<float> val_curve;
};

// ──────────────────────────────────────────────────────────────────────────
// Training (full-model run)
// ──────────────────────────────────────────────────────────────────────────

struct TrainingState {
    bool        running     = false;
    int         step        = 0;
    int         total_steps = 0;
};

struct TrainingMetricCard {
    std::string label;           // "LOSS"
    std::string value_str;       // "2.184"
    std::string sub_str;         // "-0.012"
    const char* tone;            // "good" | "warn" | ""
};

struct LossCurve {
    std::vector<float> train;
    std::vector<float> val;
};

// ──────────────────────────────────────────────────────────────────────────
// Finetune (LoRA / adapter)
// ──────────────────────────────────────────────────────────────────────────

struct LoRAConfig {
    int   rank          = kNoInt;
    float alpha         = kNoFloat;
    float dropout       = kNoFloat;
    std::string target;          // "q_proj, v_proj, o_proj"
    float trainable_M   = kNoFloat;
    float trainable_pct = kNoFloat;
    std::string method;          // "LoRA + RSLoRA scale"
};

struct OptimizerConfig {
    std::string name;            // "AdamW8bit"
    float lr            = kNoFloat;
    float beta1         = kNoFloat;
    float beta2         = kNoFloat;
    int   warmup_steps  = kNoInt;
    std::string schedule;        // "cosine"
    float weight_decay  = kNoFloat;
};

struct DataConfig {
    std::string dataset;         // "sft/instruction_v3"
    int n_train_K       = kNoInt;
    int n_eval_K        = kNoInt;
    int ctx_len         = kNoInt;
    bool packing        = false;
};

struct EvalDiffMetric {
    std::string benchmark;       // "MMLU"
    float       base   = kNoFloat;
    float       ft     = kNoFloat;
    float       delta  = kNoFloat;
};

struct ABSample {
    std::string prompt;
    std::string base_response;
    std::string ft_response;
};

// ──────────────────────────────────────────────────────────────────────────
// Datasets
// ──────────────────────────────────────────────────────────────────────────

struct DatasetSummary {
    std::string  name;
    std::int64_t size_bytes = kNoSize;
    std::int64_t n_tokens   = kNoSize;
};

struct DatasetSpan {
    int  begin = 0;
    int  end   = 0;             // half-open in chars
    int  kind  = 0;             // 0=primary (accent), 1=secondary (warn)
};

struct DatasetSample {
    int                     sample_id = kNoInt;
    std::string             doc_id;
    std::string             source;          // "arxiv/cs.CL"
    std::string             text;
    std::vector<DatasetSpan> spans;
};

struct DatasetSampleStats {
    int   len_tokens     = kNoInt;
    float ppl_base       = kNoFloat;
    float ppl_ft         = kNoFloat;
    float avg_surprisal  = kNoFloat;
    std::string top_feature;     // "f2381 attends_to_subset"
};

struct SourceMixRow {
    std::string name;
    float       fraction;
    const char* tone;            // color name
};

struct DatasetDistribution {
    std::vector<int>          doc_length_histogram;
    std::vector<SourceMixRow> source_mix;
};

// ──────────────────────────────────────────────────────────────────────────
// Engine / runtime metrics (top-of-screen + logs panel)
// ──────────────────────────────────────────────────────────────────────────

struct EngineMetrics {
    int   log_rate_per_min  = kNoInt;
    int   warn_rate_per_min = kNoInt;
    int   err_rate_per_min  = kNoInt;
    float cuda_mem_used_GB  = kNoFloat;
    float cuda_mem_total_GB = kNoFloat;
    float cpu_pct           = kNoFloat;
    float fwd_time_ms       = kNoFloat;
    float fps               = kNoFloat;
    std::string device;          // "cuda:0 A100"
    std::string dtype;           // "fp16"
};

// ──────────────────────────────────────────────────────────────────────────
// Recent-exports stub (Probes workspace)
// ──────────────────────────────────────────────────────────────────────────
struct ExportEntry {
    std::string timestamp;
    std::string filename;
};

// ══════════════════════════════════════════════════════════════════════════
// The Model interface itself
// ══════════════════════════════════════════════════════════════════════════
struct Model {
    virtual ~Model() = default;

    // ── Architecture ──────────────────────────────────────────────────────
    // Topology of the loaded checkpoint.  Backends populate this from
    // their loadCheckpoint pipeline (config + tokenizer + weights metadata).
    // Default returns an all-sentinel ModelInfo so a backend that doesn't
    // implement it doesn't break the UI — the architecture workspace just
    // shows "—" everywhere.
    virtual ModelInfo getModelInfo() { return {}; }

    // Token strings currently being inspected — typically the prompt + any
    // generated tokens for the active forward pass.  Empty when the
    // backend has nothing loaded or no active sequence.  Drives the
    // inference workspace's token strip and the activeToken cursor.
    //
    // For backends that own generation (FastAPI proxy via /generate WS),
    // these are the streamed tokens; for static-prompt inspection
    // (libtorch / llama.cpp paths) these come from tokenizing the prompt
    // and decoding each id.  Default returns empty.
    virtual std::vector<std::string> getCurrentTokens() { return {}; }

    virtual std::vector<ParamBreakdownRow> getParamBreakdown([[maybe_unused]] int layer) { return {}; }
    virtual LiveActivations                getLiveActivations([[maybe_unused]] int layer) { return {}; }

    // ── Activations / attention (per forward-pass tick) ──────────────────
    virtual std::vector<std::vector<float>>
        getAttentionPattern([[maybe_unused]] int layer, [[maybe_unused]] int head, [[maybe_unused]] int seqLen, [[maybe_unused]] HeadBias bias) { return {}; }
    virtual std::vector<float> getActivation([[maybe_unused]] int layer, [[maybe_unused]] int kind, [[maybe_unused]] int n) { return {}; }

    // ── Per-element norms (used for tinting + bars) ───────────────────────
    virtual float getHeadNorm     ([[maybe_unused]] int layer, [[maybe_unused]] int head) { return kNoFloat; }
    virtual float getComponentNorm([[maybe_unused]] int layer, [[maybe_unused]] std::string_view comp) { return kNoFloat; }

    // ── Inference workspace ──────────────────────────────────────────────
    virtual ResidualContribution      getResidualContribution([[maybe_unused]] int layer) { return {}; }
    virtual ResidualSummary           getResidualSummary    ([[maybe_unused]] int layer) { return {}; }
    virtual std::vector<LogitLensRow> getLogitLensTrajectory([[maybe_unused]] int token, [[maybe_unused]] int kLayers) { return {}; }
    virtual std::vector<LogitDist>    getOutputLogits       ([[maybe_unused]] int k) { return {}; }
    virtual std::vector<MlpFeatureActivation>
                                      getMlpFeatures        ([[maybe_unused]] int layer, [[maybe_unused]] int k) { return {}; }
    virtual std::vector<float>        getTokenLossPerToken  ([[maybe_unused]] int layer) { return {}; }
    virtual std::vector<float>        getSurprisalDelta     () { return {}; }
    virtual std::vector<ProbeEntry>   getActiveProbes       () { return {}; }
    virtual SteeringConfig            getSteering           () { return {}; }

    // ── Attention workspace ──────────────────────────────────────────────
    virtual QKVStats                  getQKVStats     ([[maybe_unused]] int layer, [[maybe_unused]] int head, [[maybe_unused]] int token) { return {}; }
    virtual std::vector<HeadStatRow>  getHeadStats    ([[maybe_unused]] int layer, [[maybe_unused]] int head) { return {}; }
    virtual PatchSourceState          getPatchSource  () { return {}; }
    virtual HeadBias                  getHeadBias     ([[maybe_unused]] int layer, [[maybe_unused]] int head) { return HeadBias::Diag; }

    // ── Probes / SAE workspace ───────────────────────────────────────────
    virtual std::vector<FeatureSummary> getFeatureLibrary([[maybe_unused]] std::string_view filter) { return {}; }
    virtual FeatureCard                 getFeatureCard   ([[maybe_unused]] int featureId) { return {}; }
    virtual std::vector<FeatureExample> getFeatureExamples([[maybe_unused]] int featureId, [[maybe_unused]] int k) { return {}; }
    virtual std::vector<CoFiringEntry>  getCoFiringFeatures([[maybe_unused]] int featureId, [[maybe_unused]] float threshold) { return {}; }
    virtual SAETrainingMetrics          getSAETrainingMetrics([[maybe_unused]] std::string_view saeId) { return {}; }
    virtual std::vector<ProbeEntry>     getProbeLibrary    () { return {}; }
    virtual ProbeTrainState             getProbeTrainState ([[maybe_unused]] std::string_view name) { return {}; }
    virtual std::vector<ExportEntry>    getRecentExports   () { return {}; }

    // Mutators (no-op when not implemented).  Marked [[maybe_unused]] on
    // params because the base default discards them.
    virtual void  startProbeTraining([[maybe_unused]] std::string_view name,
                                     [[maybe_unused]] std::string_view kind,
                                     [[maybe_unused]] std::string_view location,
                                     [[maybe_unused]] std::string_view dataset) {}
    virtual void  saveProbe         ([[maybe_unused]] std::string_view name)    {}
    virtual void  exportSnapshot    ([[maybe_unused]] std::string_view path)    {}

    // ── Checkpoint lifecycle (ENGINE_API.md §2.2) ────────────────────────
    // The base default returns a no-op success (lets MockModel keep
    // demoing).  A real backend overrides to actually load the file +
    // populate getModelInfo so AppState::loadFromModel can refresh the
    // topology display.  See ENGINE_API.md §2.2 for the async-load
    // contract: validate header synchronously, do heavy work on a worker
    // thread, fan progress through drainEngineLogs.
    struct CheckpointResult {
        bool        ok    = true;
        std::string error;
    };

    // Backend-tunable knobs honoured during loadCheckpoint.  Common keys
    // are first-class fields; backend-specific options ride in `extras`
    // (e.g. llama.cpp's `n_ctx`, libtorch's `device`, GGUF's `align`).
    //
    // Backends that don't recognise a field MUST silently ignore it —
    // the same options struct may be used across mock / hf / gguf /
    // llama / libtorch flows.  This is forward-compat by design:
    // unknown future fields cause no errors today.
    struct LoadOptions {
        std::string mode;          // quant hint: "nf4" | "q4_0" | "fp16" | "" (default)
        bool        mmap        = true;
        bool        verify_hash = false;
        std::vector<std::pair<std::string, std::string>> extras;  // backend-specific kvs
    };

    // Path-only form — the historical surface; kept as the primary
    // virtual so existing backends compile unchanged.
    virtual CheckpointResult loadCheckpoint([[maybe_unused]] std::string_view path) { return {}; }

    // Options form — preferred for new backends.  Default impl ignores
    // options and delegates to the path-only form, so a backend that
    // doesn't care about options gets correct behaviour for free.
    virtual CheckpointResult loadCheckpoint(std::string_view path,
                                            [[maybe_unused]] const LoadOptions& opts) {
        return loadCheckpoint(path);
    }
    virtual void             unloadCheckpoint() {}

    // Long-running mutator progress.  Returns an empty Progress when no
    // work is in flight; otherwise `kind` names the operation
    // ("load" / "train" / "export") and `step / total` gives the
    // fraction (total == 0 ⇒ indeterminate progress, UI shows spinner).
    // Backends increment this from their engine thread; UI reads at
    // frame rate.  Implementations must make the read cheap and
    // lock-free if possible (a struct of POD scalars in a mutex-guarded
    // member is fine — the lock is a few nanoseconds).
    struct Progress {
        std::string kind;          // "" ⇒ idle
        int         step  = 0;
        int         total = 0;
    };
    virtual Progress getProgress() const { return {}; }

    // ── Unified data structure (ModelView) ───────────────────────────────
    // The canonical accessor. Backends own a ModelView and return it here.
    // Pure virtual so every backend explicitly commits to a view — even
    // MockModel's mock-data populator writes into one.  Per-DTO getters
    // below are kept for source compatibility and prefer reading from
    // view() where the data is naturally there; backends that prefer to
    // compute per call still override individual getters.
    virtual const ModelView& view() const = 0;

    // ── Capabilities advertisement ───────────────────────────────────────
    // Bit-bag the UI consults to grey out controls a backend can't honour.
    // Each field is conservative-default-false so a new backend that
    // forgets to set them simply gets blank panels (preferred to crashing
    // on a wrong-shaped response).
    //
    // ABI policy: fields are append-only.  Re-ordering or removing a
    // field is a major-version break; adding one is forward-compatible
    // because downstream code reads named members, not packed bits.
    struct Capabilities {
        bool has_topology       = false;
        bool has_tokenizer      = false;   // encode/decode wired
        bool has_state_dict     = false;   // tensor enumeration
        bool has_attention      = false;   // live attention matrices
        bool has_residual       = false;   // live residual stream
        bool has_logit_lens     = false;   // per-layer unembed projection
        bool has_token_stream   = false;   // live generation
        bool has_captures       = false;   // can run a forward pass at all
        bool has_intervention   = false;   // setAblation / setSteering honoured
        bool has_weight_deltas  = false;   // surgery/weight_deltas writeable
        bool has_training       = false;
    };
    virtual Capabilities getCapabilities() const { return {}; }

    // ── New mutators (ENGINE_API.md §7) ──────────────────────────────────
    // Default no-op so a backend that doesn't support a hook is silent.
    //
    //   setActivePrompt — UI invokes when the inference workspace's
    //                     prompt commits.  Engine kicks off the capture
    //                     that populates ModelView::current.
    //                     OWNERSHIP: `prompt` is a borrow; the backend
    //                     MUST copy if it retains the value beyond this
    //                     call (e.g. for an async forward pass).
    //   setAblation     — full set of ablated heads + components.  Called
    //                     debounced (~200ms) when the UI's ablation set
    //                     changes; engine masks accordingly.  Heads /
    //                     components are passed by value (the engine
    //                     copies internally).
    //   setSteering /
    //   clearSteering   — install / remove a steering vector.
    virtual void setActivePrompt([[maybe_unused]] std::string_view prompt) {}

    // setAblation takes the full set of ablation targets — heads +
    // components.  Names are canonical-path strings; structured
    // refs (AttentionHeadRef / ComponentRef in model_view.hpp) have
    // canonical() helpers that produce these strings, so a typed caller
    // can do `setAblation({h.canonical()}, {})`.
    //
    // String form is the wire format for two reasons: (a) RPC / scripting
    // callers don't carry the ref types, (b) model.hpp forward-declares
    // ModelView and pulling the ref types in here would invert the
    // header layering.  Backends parse / validate the strings themselves
    // (typically via AttentionHeadRef::parse — returns nullopt on
    // malformed input, which the backend logs and skips).
    virtual void setAblation    ([[maybe_unused]] const std::vector<std::string>& head_canonical,
                                 [[maybe_unused]] const std::vector<std::string>& component_canonical) {}
    virtual void setSteering    ([[maybe_unused]] const SteeringConfig&   cfg) {}
    virtual void clearSteering  () {}

    // Sampling control — consulted by the engine's generation loop.
    // Default no-op so backends that don't generate (Gguf, HFProxy) can
    // ignore.  Backends that DO generate (LlamaCpp) override to install
    // the new config; subsequent setActivePrompt calls use it.
    virtual void setSamplerConfig([[maybe_unused]] const SamplerConfig& cfg) {}

    // Max tokens to generate per setActivePrompt call (after the prompt
    // prefill).  Default 0 leaves the backend's own default in place
    // (currently 24 for LlamaCpp).  setMaxGenerationTokens(n) clamps
    // the next decode's generation to n tokens (or EOS, whichever first).
    virtual void setMaxGenerationTokens([[maybe_unused]] int n) {}

    // ── Training workspace ───────────────────────────────────────────────
    virtual TrainingState                   getTrainingState   () { return {}; }
    virtual std::vector<TrainingMetricCard> getTrainingMetrics () { return {}; }
    virtual LossCurve                       getTrainingLoss    ([[maybe_unused]] int maxSteps) { return {}; }
    virtual std::vector<float>              getGradFlowPerLayer() { return {}; }
    virtual std::vector<std::vector<float>> getPerLayerLoss    ([[maybe_unused]] int maxSteps) { return {}; }

    virtual void resumeTraining() {}
    virtual void pauseTraining()  {}
    virtual void stepTraining()   {}
    virtual void resetTraining()  {}
    virtual void stopTraining()   {}

    // ── Finetune workspace ───────────────────────────────────────────────
    virtual LoRAConfig          getLoRAConfig     () { return {}; }
    virtual OptimizerConfig     getOptimizerConfig() { return {}; }
    virtual DataConfig          getDataConfig     () { return {}; }
    virtual EvalDiffMetric      getEvalDiff       ([[maybe_unused]] std::string_view bench) { return {}; }
    virtual std::vector<float>  getEvalLossCurve  () { return {}; }
    virtual ABSample            getABSample       () { return {}; }
    virtual std::vector<std::vector<float>>
                                getDeltaWHeatmap  ([[maybe_unused]] int numLayers, [[maybe_unused]] int numComponents) { return {}; }
    virtual std::vector<std::string>
                                getDeltaWComponentNames() { return {}; }

    // ── Datasets workspace ───────────────────────────────────────────────
    virtual std::vector<DatasetSummary> getDatasets       () { return {}; }
    virtual DatasetSample               getSample         ([[maybe_unused]] std::string_view dataset, [[maybe_unused]] int id) { return {}; }
    virtual DatasetSampleStats          getSampleStats    ([[maybe_unused]] std::string_view dataset, [[maybe_unused]] int id) { return {}; }
    virtual DatasetDistribution         getDatasetDistribution([[maybe_unused]] std::string_view dataset) { return {}; }
    virtual std::vector<float>          getTokenIds       ([[maybe_unused]] std::string_view dataset, [[maybe_unused]] int id, [[maybe_unused]] int n) { return {}; }

    // ── Raw tensors workspace ────────────────────────────────────────────
    // Defaults walk view().tensors so any backend that populates the
    // TensorRegistry gets state-dict enumeration for free (see
    // model_view.cpp for the definitions).  Backends with no tensor
    // registry inherit the empty result.
    virtual std::vector<TensorMeta> getStateDict    ();
    virtual TensorMeta              getTensorMeta   (std::string_view name);
    virtual std::vector<float>      getWeightSlice  ([[maybe_unused]] std::string_view name, [[maybe_unused]] int offset, [[maybe_unused]] int n) { return {}; }
    virtual std::vector<int>        getWeightHistogram([[maybe_unused]] std::string_view name, [[maybe_unused]] int bins) { return {}; }
    virtual TensorStats             getTensorStats  ([[maybe_unused]] std::string_view name) { return {}; }
    virtual std::vector<float>      getSingularValues([[maybe_unused]] std::string_view name, [[maybe_unused]] int k) { return {}; }
    virtual std::vector<std::vector<float>>
                                    getTensorSlice2D([[maybe_unused]] std::string_view name,
                                                     [[maybe_unused]] int axis0, [[maybe_unused]] int axis1, [[maybe_unused]] int rows, [[maybe_unused]] int cols) { return {}; }
    virtual std::vector<std::vector<float>>
                                    getDiffSlice2D  ([[maybe_unused]] std::string_view name, [[maybe_unused]] int rows, [[maybe_unused]] int cols) { return {}; }
    virtual DiffStats               getDiffStats    ([[maybe_unused]] std::string_view name) { return {}; }

    // ── Engine / runtime ─────────────────────────────────────────────────
    virtual EngineMetrics         getEngineMetrics() { return {}; }

    // ── Engine log bridge ────────────────────────────────────────────────
    // Pulled once per frame.  Implementations should return any log
    // lines the engine has emitted since the last call (FIFO ring is
    // typical) and clear their internal buffer.  MockModel returns {}.
    virtual std::vector<LogEntry> drainEngineLogs() { return {}; }
};

// MockModel — opt-in screenshot/demo backend (mock_model.cpp).  When the
// build-time flag `LLOB_USE_MOCK_DATA` is defined, every override emits
// deterministic synthetic data so the UI can be exercised without a real
// model wired up.  When the flag is undefined, MockModel inherits Model's
// empty defaults like any other backend.
//
// NOTE: concrete backends (HFProxyEngine, GgufInspectorEngine,
// LlamaCppEngine, ...) inherit Model, NOT MockModel.  Inheriting MockModel
// previously meant unimplemented getters silently emitted mock data —
// honest empties via Model are the new contract.  MockModel is only
// instantiated when the user explicitly selects `LLOB_BACKEND=mock`.
struct MockModel : Model {
#define DECL_OVERRIDE(ret, sig) ret sig override
    DECL_OVERRIDE(ModelInfo,                      getModelInfo());
    DECL_OVERRIDE(std::vector<std::string>,       getCurrentTokens());
    DECL_OVERRIDE(std::vector<ParamBreakdownRow>, getParamBreakdown(int layer));
    DECL_OVERRIDE(LiveActivations,                getLiveActivations(int layer));
    DECL_OVERRIDE(std::vector<std::vector<float>>,
                  getAttentionPattern(int layer, int head, int seqLen, HeadBias bias));
    DECL_OVERRIDE(std::vector<float>, getActivation(int layer, int kind, int n));
    DECL_OVERRIDE(float,              getHeadNorm     (int layer, int head));
    DECL_OVERRIDE(float,              getComponentNorm(int layer, std::string_view comp));

    DECL_OVERRIDE(ResidualContribution,      getResidualContribution(int layer));
    DECL_OVERRIDE(ResidualSummary,           getResidualSummary    (int layer));
    DECL_OVERRIDE(std::vector<LogitLensRow>, getLogitLensTrajectory(int token, int kLayers));
    DECL_OVERRIDE(std::vector<LogitDist>,    getOutputLogits       (int k));
    DECL_OVERRIDE(std::vector<MlpFeatureActivation>,
                                             getMlpFeatures        (int layer, int k));
    DECL_OVERRIDE(std::vector<float>,        getTokenLossPerToken  (int layer));
    DECL_OVERRIDE(std::vector<float>,        getSurprisalDelta     ());
    DECL_OVERRIDE(std::vector<ProbeEntry>,   getActiveProbes       ());
    DECL_OVERRIDE(SteeringConfig,            getSteering           ());

    DECL_OVERRIDE(QKVStats,                  getQKVStats     (int layer, int head, int token));
    DECL_OVERRIDE(std::vector<HeadStatRow>,  getHeadStats    (int layer, int head));
    DECL_OVERRIDE(PatchSourceState,          getPatchSource  ());
    DECL_OVERRIDE(HeadBias,                  getHeadBias     (int layer, int head));

    DECL_OVERRIDE(std::vector<FeatureSummary>, getFeatureLibrary(std::string_view filter));
    DECL_OVERRIDE(FeatureCard,                 getFeatureCard   (int featureId));
    DECL_OVERRIDE(std::vector<FeatureExample>, getFeatureExamples(int featureId, int k));
    DECL_OVERRIDE(std::vector<CoFiringEntry>,  getCoFiringFeatures(int featureId, float threshold));
    DECL_OVERRIDE(SAETrainingMetrics,          getSAETrainingMetrics(std::string_view saeId));
    DECL_OVERRIDE(std::vector<ProbeEntry>,     getProbeLibrary    ());
    DECL_OVERRIDE(ProbeTrainState,             getProbeTrainState (std::string_view name));
    DECL_OVERRIDE(std::vector<ExportEntry>,    getRecentExports   ());

    DECL_OVERRIDE(TrainingState,                   getTrainingState   ());
    DECL_OVERRIDE(std::vector<TrainingMetricCard>, getTrainingMetrics ());
    DECL_OVERRIDE(LossCurve,                       getTrainingLoss    (int maxSteps));
    DECL_OVERRIDE(std::vector<float>,              getGradFlowPerLayer());
    DECL_OVERRIDE(std::vector<std::vector<float>>, getPerLayerLoss    (int maxSteps));

    DECL_OVERRIDE(LoRAConfig,                  getLoRAConfig     ());
    DECL_OVERRIDE(OptimizerConfig,             getOptimizerConfig());
    DECL_OVERRIDE(DataConfig,                  getDataConfig     ());
    DECL_OVERRIDE(EvalDiffMetric,              getEvalDiff       (std::string_view bench));
    DECL_OVERRIDE(std::vector<float>,          getEvalLossCurve  ());
    DECL_OVERRIDE(ABSample,                    getABSample       ());
    DECL_OVERRIDE(std::vector<std::vector<float>>,
                                               getDeltaWHeatmap  (int numLayers, int numComponents));
    DECL_OVERRIDE(std::vector<std::string>,    getDeltaWComponentNames());

    DECL_OVERRIDE(std::vector<DatasetSummary>, getDatasets       ());
    DECL_OVERRIDE(DatasetSample,               getSample         (std::string_view dataset, int id));
    DECL_OVERRIDE(DatasetSampleStats,          getSampleStats    (std::string_view dataset, int id));
    DECL_OVERRIDE(DatasetDistribution,         getDatasetDistribution(std::string_view dataset));
    DECL_OVERRIDE(std::vector<float>,          getTokenIds       (std::string_view dataset, int id, int n));

    DECL_OVERRIDE(std::vector<TensorMeta>,         getStateDict    ());
    DECL_OVERRIDE(TensorMeta,                      getTensorMeta   (std::string_view name));
    DECL_OVERRIDE(std::vector<float>,              getWeightSlice  (std::string_view name, int offset, int n));
    DECL_OVERRIDE(std::vector<int>,                getWeightHistogram(std::string_view name, int bins));
    DECL_OVERRIDE(TensorStats,                     getTensorStats  (std::string_view name));
    DECL_OVERRIDE(std::vector<float>,              getSingularValues(std::string_view name, int k));
    DECL_OVERRIDE(std::vector<std::vector<float>>, getTensorSlice2D(std::string_view name,
                                                                     int axis0, int axis1, int rows, int cols));
    DECL_OVERRIDE(std::vector<std::vector<float>>, getDiffSlice2D  (std::string_view name, int rows, int cols));
    DECL_OVERRIDE(DiffStats,                       getDiffStats    (std::string_view name));

    DECL_OVERRIDE(EngineMetrics,           getEngineMetrics());
    DECL_OVERRIDE(std::vector<LogEntry>,   drainEngineLogs());
#undef DECL_OVERRIDE

    // Probe-training tick (mock side effect): step counter advances when
    // `m_probe_training` is true.  Real backend would tick from the
    // training loop's own callbacks.
    bool m_probe_training = false;
    int  m_probe_step     = 256;
    void startProbeTraining(std::string_view, std::string_view,
                            std::string_view, std::string_view) override {
        m_probe_training = true;
        m_probe_step     = 0;
    }
    void saveProbe(std::string_view) override {}
    void exportSnapshot(std::string_view) override {}

    // Unified data structure.  MockModel owns one ModelView; populator
    // lives in mock_model.cpp (Mulberry32-seeded topology + a synthetic
    // TensorRegistry that gives the raw-tensors workspace something to
    // enumerate even before a real backend is wired).
    const ModelView& view() const override;
    Capabilities     getCapabilities() const override;

    MockModel();
    ~MockModel() override;
    MockModel(const MockModel&)            = delete;
    MockModel& operator=(const MockModel&) = delete;

private:
    struct State;
    std::unique_ptr<State> m_state;
};

}  // namespace llmengine
