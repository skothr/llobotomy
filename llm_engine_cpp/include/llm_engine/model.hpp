#pragma once
//
// Model — abstract data backend for every llobotomy workspace.
//
// Each method below is a [DATA HOOK]: it tells the engine what the UI needs
// at that point in time.  The default implementation is `MockModel`, which
// returns deterministic fake data when `LLOB_USE_MOCK_DATA` is defined and
// empty / sentinel values otherwise.  A real backend (HuggingFace via the
// FastAPI bridge, native llama.cpp, libtorch in-process, ...) implements
// the same interface and the UI never knows the difference.
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
#include <string>
#include <string_view>
#include <vector>

namespace llmengine {

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

    virtual std::vector<ParamBreakdownRow> getParamBreakdown(int layer) = 0;
    virtual LiveActivations                getLiveActivations(int layer) = 0;

    // ── Activations / attention (per forward-pass tick) ──────────────────
    virtual std::vector<std::vector<float>>
        getAttentionPattern(int layer, int head, int seqLen, HeadBias bias) = 0;
    virtual std::vector<float> getActivation(int layer, int kind, int n) = 0;

    // ── Per-element norms (used for tinting + bars) ───────────────────────
    virtual float getHeadNorm     (int layer, int head)              = 0;
    virtual float getComponentNorm(int layer, std::string_view comp) = 0;

    // ── Inference workspace ──────────────────────────────────────────────
    virtual ResidualContribution      getResidualContribution(int layer)         = 0;
    virtual ResidualSummary           getResidualSummary    (int layer)          = 0;
    virtual std::vector<LogitLensRow> getLogitLensTrajectory(int token, int kLayers) = 0;
    virtual std::vector<LogitDist>    getOutputLogits       (int k)              = 0;
    virtual std::vector<MlpFeatureActivation>
                                      getMlpFeatures        (int layer, int k)  = 0;
    virtual std::vector<float>        getTokenLossPerToken  (int layer)          = 0;
    virtual std::vector<float>        getSurprisalDelta     ()                   = 0;
    virtual std::vector<ProbeEntry>   getActiveProbes       ()                   = 0;
    virtual SteeringConfig            getSteering           ()                   = 0;

    // ── Attention workspace ──────────────────────────────────────────────
    virtual QKVStats                  getQKVStats     (int layer, int head, int token) = 0;
    virtual std::vector<HeadStatRow>  getHeadStats    (int layer, int head)            = 0;
    virtual PatchSourceState          getPatchSource  ()                                = 0;
    virtual HeadBias                  getHeadBias     (int layer, int head)            = 0;

    // ── Probes / SAE workspace ───────────────────────────────────────────
    virtual std::vector<FeatureSummary> getFeatureLibrary(std::string_view filter) = 0;
    virtual FeatureCard                 getFeatureCard   (int featureId)            = 0;
    virtual std::vector<FeatureExample> getFeatureExamples(int featureId, int k)    = 0;
    virtual std::vector<CoFiringEntry>  getCoFiringFeatures(int featureId, float threshold) = 0;
    virtual SAETrainingMetrics          getSAETrainingMetrics(std::string_view saeId) = 0;
    virtual std::vector<ProbeEntry>     getProbeLibrary    ()                       = 0;
    virtual ProbeTrainState             getProbeTrainState (std::string_view name)  = 0;
    virtual std::vector<ExportEntry>    getRecentExports   ()                       = 0;

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
    virtual CheckpointResult loadCheckpoint([[maybe_unused]] std::string_view path) { return {}; }
    virtual void             unloadCheckpoint() {}

    // ── Training workspace ───────────────────────────────────────────────
    virtual TrainingState                   getTrainingState   ()                = 0;
    virtual std::vector<TrainingMetricCard> getTrainingMetrics ()                = 0;
    virtual LossCurve                       getTrainingLoss    (int maxSteps)   = 0;
    virtual std::vector<float>              getGradFlowPerLayer()                = 0;
    virtual std::vector<std::vector<float>> getPerLayerLoss    (int maxSteps)   = 0;

    virtual void resumeTraining() {}
    virtual void pauseTraining()  {}
    virtual void stepTraining()   {}
    virtual void resetTraining()  {}
    virtual void stopTraining()   {}

    // ── Finetune workspace ───────────────────────────────────────────────
    virtual LoRAConfig          getLoRAConfig     ()                       = 0;
    virtual OptimizerConfig     getOptimizerConfig()                       = 0;
    virtual DataConfig          getDataConfig     ()                       = 0;
    virtual EvalDiffMetric      getEvalDiff       (std::string_view bench) = 0;
    virtual std::vector<float>  getEvalLossCurve  ()                       = 0;
    virtual ABSample            getABSample       ()                       = 0;
    virtual std::vector<std::vector<float>>
                                getDeltaWHeatmap  (int numLayers, int numComponents) = 0;
    virtual std::vector<std::string>
                                getDeltaWComponentNames() = 0;

    // ── Datasets workspace ───────────────────────────────────────────────
    virtual std::vector<DatasetSummary> getDatasets       ()                                 = 0;
    virtual DatasetSample               getSample         (std::string_view dataset, int id) = 0;
    virtual DatasetSampleStats          getSampleStats    (std::string_view dataset, int id) = 0;
    virtual DatasetDistribution         getDatasetDistribution(std::string_view dataset)     = 0;
    virtual std::vector<float>          getTokenIds       (std::string_view dataset, int id, int n) = 0;

    // ── Raw tensors workspace ────────────────────────────────────────────
    virtual std::vector<TensorMeta> getStateDict    ()                                 = 0;
    virtual TensorMeta              getTensorMeta   (std::string_view name)            = 0;
    virtual std::vector<float>      getWeightSlice  (std::string_view name, int offset, int n) = 0;
    virtual std::vector<int>        getWeightHistogram(std::string_view name, int bins) = 0;
    virtual TensorStats             getTensorStats  (std::string_view name)            = 0;
    virtual std::vector<float>      getSingularValues(std::string_view name, int k)    = 0;
    virtual std::vector<std::vector<float>>
                                    getTensorSlice2D(std::string_view name,
                                                     int axis0, int axis1, int rows, int cols) = 0;
    virtual std::vector<std::vector<float>>
                                    getDiffSlice2D  (std::string_view name, int rows, int cols) = 0;
    virtual DiffStats               getDiffStats    (std::string_view name)            = 0;

    // ── Engine / runtime ─────────────────────────────────────────────────
    virtual EngineMetrics         getEngineMetrics() = 0;

    // ── Engine log bridge ────────────────────────────────────────────────
    // Pulled once per frame.  Implementations should return any log
    // lines the engine has emitted since the last call (FIFO ring is
    // typical) and clear their internal buffer.  MockModel returns {}.
    virtual std::vector<LogEntry> drainEngineLogs() = 0;
};

// MockModel — see model/mock_model.cpp.  When the build-time flag
// `LLOB_USE_MOCK_DATA` is undefined, every method returns the no-data
// sentinel for its type (empty vector / NaN / "").  This makes it
// impossible to ship a release build that silently shows fake data — the
// UI either has a real backend wired up or it shows blank panels.
struct MockModel : Model {
#define DECL_OVERRIDE(ret, sig) ret sig override
    DECL_OVERRIDE(ModelInfo,                      getModelInfo());
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
};

}  // namespace llmengine
