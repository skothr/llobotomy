// MockModel — deterministic fake data so the UI looks alive in dev / for
// screenshots.  Every method body is wrapped in `#if LLOB_USE_MOCK_DATA`.
// When the flag is not defined (i.e. release / real-backend builds), the
// methods return the empty-vector / NaN / "" sentinels declared in
// model.hpp, and the UI shows "no data" placeholders instead.

#include "model/model.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace llob {

namespace {

#if LLOB_USE_MOCK_DATA

// ── Tiny utilities used by the mock generators ───────────────────────────
std::uint32_t hashString(std::string_view s) {
    std::uint32_t h = 2166136261u;
    for (char c : s) { h ^= static_cast<std::uint8_t>(c); h *= 16777619u; }
    return h;
}

#endif  // LLOB_USE_MOCK_DATA

}  // namespace

// ──────────────────────────────────────────────────────────────────────────
// Architecture / weights
// ──────────────────────────────────────────────────────────────────────────

std::vector<ParamBreakdownRow> MockModel::getParamBreakdown([[maybe_unused]] int layer) {
#if LLOB_USE_MOCK_DATA
    return {
        { "W_Q + W_K + W_V", 1.77f, 0.25f,  "info"  },
        { "W_O",             0.59f, 0.08f,  "info"  },
        { "W_in (gate+up)",  3.15f, 0.44f,  "warn"  },
        { "W_out",           1.57f, 0.22f,  "warn"  },
        { "norm × 2",        0.0015f, 0.0002f, "muted" },
    };
#else
    return {};
#endif
}

LiveActivations MockModel::getLiveActivations([[maybe_unused]] int layer) {
#if LLOB_USE_MOCK_DATA
    return { 1.42f, 0.84f, 14.83f, 1.87f, 14, 3072 };
#else
    return {};
#endif
}

// ──────────────────────────────────────────────────────────────────────────
// Activations / attention
// ──────────────────────────────────────────────────────────────────────────

std::vector<std::vector<float>>
MockModel::getAttentionPattern([[maybe_unused]] int layer, [[maybe_unused]] int head,
                                int seqLen, [[maybe_unused]] HeadBias bias) {
#if LLOB_USE_MOCK_DATA
    const int n = std::max(1, seqLen);
    Mulberry32 rng(static_cast<std::uint32_t>(layer * 100 + head + 1));
    std::vector<std::vector<float>> m(n, std::vector<float>(n, 0.0f));
    for (int i = 0; i < n; ++i) {
        std::vector<float> row(i + 1, 0.0f);
        for (int j = 0; j <= i; ++j) {
            float v = rng.next() * 0.3f;
            switch (bias) {
                case HeadBias::Diag:
                    if (j == i)         v += 0.7f * (0.6f + rng.next() * 0.4f);
                    else if (j == i - 1) v += 0.4f * (0.6f + rng.next() * 0.4f);
                    break;
                case HeadBias::Prev:
                    if (j == i - 1) v += 0.85f * (0.7f + rng.next() * 0.3f);
                    break;
                case HeadBias::First:
                    if (j == 0) v += 0.6f;
                    if (j == i) v += 0.2f;
                    break;
                case HeadBias::Broad:
                    v = rng.next() * 0.5f + 0.1f;
                    break;
                case HeadBias::Induction: {
                    const int peak = (i * 7 + layer * 100 + head) % std::max(1, i);
                    if (j == peak) v += 0.7f;
                    break;
                }
            }
            row[j] = v;
        }
        float sum = 0.0f; for (float v : row) sum += v;
        if (sum <= 0.0f) sum = 1.0f;
        for (int j = 0; j <= i; ++j) m[i][j] = row[j] / sum;
    }
    return m;
#else
    return {};
#endif
}

std::vector<float>
MockModel::getActivation([[maybe_unused]] int layer, [[maybe_unused]] int kind, int n) {
#if LLOB_USE_MOCK_DATA
    Mulberry32 rng(static_cast<std::uint32_t>(layer * 31 + kind * 7 + 1));
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) v[i] = (rng.next() - 0.45f) * 2.2f;
    return v;
#else
    return {};
#endif
}

float MockModel::getHeadNorm([[maybe_unused]] int layer, [[maybe_unused]] int head) {
#if LLOB_USE_MOCK_DATA
    Mulberry32 rng(static_cast<std::uint32_t>(layer * 7 + head * 13 + 1));
    return 0.15f + rng.next() * 0.8f;
#else
    return kNoFloat;
#endif
}

float MockModel::getComponentNorm([[maybe_unused]] int layer,
                                   [[maybe_unused]] std::string_view comp) {
#if LLOB_USE_MOCK_DATA
    Mulberry32 rng(static_cast<std::uint32_t>(layer * 31) ^ hashString(comp));
    return 0.2f + rng.next() * 0.7f;
#else
    return kNoFloat;
#endif
}

// ──────────────────────────────────────────────────────────────────────────
// Inference
// ──────────────────────────────────────────────────────────────────────────

ResidualContribution MockModel::getResidualContribution([[maybe_unused]] int layer) {
#if LLOB_USE_MOCK_DATA
    Mulberry32 rng(7u + static_cast<std::uint32_t>(layer));
    return { 0.2f + rng.next() * 0.7f, 0.15f + rng.next() * 0.7f };
#else
    return {};
#endif
}

ResidualSummary MockModel::getResidualSummary([[maybe_unused]] int layer) {
#if LLOB_USE_MOCK_DATA
    auto c = getResidualContribution(layer);
    return { c.attn, c.mlp, 14.832f, 0.991f, 4.32f, 257, 384 };
#else
    return {};
#endif
}

std::vector<LogitLensRow>
MockModel::getLogitLensTrajectory([[maybe_unused]] int token, int kLayers) {
#if LLOB_USE_MOCK_DATA
    // Synthesised so probability rises monotonically and entropy falls —
    // matches the visual story in the React mock.
    static const struct { const char* t1; const char* t2; } toks[] = {
        {"the","a"},{"the","of"},{"the","a"},{"to","the"},
        {"to","on"},{"to","towards"},{"to","towards"},{"to","towards"},
        {"to","towards"},{"to","towards"},{"to","towards"},{"to","towards"},
    };
    const int n = std::clamp(kLayers, 1, int(std::size(toks)));
    std::vector<LogitLensRow> out; out.reserve(n);
    for (int i = 0; i < n; ++i) {
        const float p1 = 0.04f + 0.08f * float(i);
        const float p2 = 0.03f + 0.005f * float(i);
        const float ent = std::max(0.6f, 4.82f - 0.4f * float(i));
        const bool resolved = (i == 8);
        out.push_back({ i, toks[i].t1, p1, toks[i].t2, p2, ent, resolved });
    }
    return out;
#else
    return {};
#endif
}

std::vector<LogitDist> MockModel::getOutputLogits(int k) {
#if LLOB_USE_MOCK_DATA
    static const LogitDist all[] = {
        { "to",       0.92f,  +0.04f,  true  },
        { "towards",  0.022f, -0.01f,  false },
        { "on",       0.018f,  0.0f,   false },
        { "at",       0.011f, -0.002f, false },
        { "for",      0.008f, +0.001f, false },
        { "through",  0.005f,  0.0f,   false },
    };
    std::vector<LogitDist> out;
    for (int i = 0; i < std::min<int>(k, int(std::size(all))); ++i) out.push_back(all[i]);
    return out;
#else
    return {};
#endif
}

std::vector<MlpFeatureActivation>
MockModel::getMlpFeatures([[maybe_unused]] int layer, int k) {
#if LLOB_USE_MOCK_DATA
    static const MlpFeatureActivation all[] = {
        {2381, 2.94f}, {471, 2.61f}, {1923, 2.18f}, {3014, 1.92f},
        {  80, 1.74f}, {2557,1.68f}, {1134, 1.61f}, {2202, 1.42f},
    };
    std::vector<MlpFeatureActivation> out;
    for (int i = 0; i < std::min<int>(k, int(std::size(all))); ++i) out.push_back(all[i]);
    return out;
#else
    return {};
#endif
}

std::vector<float> MockModel::getTokenLossPerToken([[maybe_unused]] int layer) {
#if LLOB_USE_MOCK_DATA
    // Length-20 mock loss curve — caller overlays SAMPLE_TOKENS.
    std::vector<float> v(20);
    for (int i = 0; i < 20; ++i)
        v[i] = std::abs(std::sin(i * 0.7f + layer * 0.3f));
    return v;
#else
    return {};
#endif
}

std::vector<float> MockModel::getSurprisalDelta() {
#if LLOB_USE_MOCK_DATA
    std::vector<float> v(20);
    for (int i = 0; i < 20; ++i) v[i] = std::sin(i * 0.7f) * 0.5f + 0.5f;
    return v;
#else
    return {};
#endif
}

std::vector<ProbeEntry> MockModel::getActiveProbes() {
#if LLOB_USE_MOCK_DATA
    return {
        {"L","accent","linear/refusal_dir",  "L08.resid_post", 0.91f, true},
        {"P","accent","logistic/sentiment",  "L11.resid_post", 0.84f, false},
        {"S","warn",  "SAE/feature_2381",    "L08.mlp_out",    0.62f, false},
    };
#else
    return {};
#endif
}

SteeringConfig MockModel::getSteering() {
#if LLOB_USE_MOCK_DATA
    return { true, "\"refusal\" prompts (n=128)", "L08.resid_post", 1.40f, 0.873f };
#else
    return {};
#endif
}

// ──────────────────────────────────────────────────────────────────────────
// Attention
// ──────────────────────────────────────────────────────────────────────────

QKVStats MockModel::getQKVStats([[maybe_unused]] int layer, [[maybe_unused]] int head,
                                 [[maybe_unused]] int token) {
#if LLOB_USE_MOCK_DATA
    return { 0.421f, 0.388f, 0.623f, 0.04f, 0.31f, 0.42f };
#else
    return {};
#endif
}

std::vector<HeadStatRow> MockModel::getHeadStats([[maybe_unused]] int layer,
                                                  [[maybe_unused]] int head) {
#if LLOB_USE_MOCK_DATA
    return {
        { "entropy / token",     "1.14 nats", 0.45f, "" },
        { "diagonal weight",     "0.62",      0.62f, "" },
        { "previous-token bias", "0.41",      0.41f, "" },
        { "BOS attention",       "0.07",      0.07f, "muted" },
        { "induction score",     "0.78",      0.78f, "warn" },
    };
#else
    return {};
#endif
}

PatchSourceState MockModel::getPatchSource() {
#if LLOB_USE_MOCK_DATA
    return { "\"The ship sailed...\"", 11, "attn_out", 0.31f };
#else
    return {};
#endif
}

HeadBias MockModel::getHeadBias([[maybe_unused]] int layer, [[maybe_unused]] int head) {
#if LLOB_USE_MOCK_DATA
    static const HeadBias bm[] = { HeadBias::Diag, HeadBias::Prev, HeadBias::First,
                                    HeadBias::Broad, HeadBias::Induction };
    return bm[(layer + head) % 5];
#else
    return HeadBias::Diag;
#endif
}

// ──────────────────────────────────────────────────────────────────────────
// Probes / SAE
// ──────────────────────────────────────────────────────────────────────────

std::vector<FeatureSummary>
MockModel::getFeatureLibrary([[maybe_unused]] std::string_view filter) {
#if LLOB_USE_MOCK_DATA
    return {
        { 2381, "attends_to_subset",   8, 17.4f, 0.412f, "SAE"    },
        {  471, "attention/head_pat",  7, 22.1f, 0.218f, "SAE"    },
        { 1923, "induction-A->B",      9, 14.0f, 0.181f, "SAE"    },
        { 3014, "self-reference",      6, 31.2f, 0.094f, "linear" },
        {   80, "syntax/clause",       5, 18.4f, 0.142f, "linear" },
        { 2557, "code/python_def",     3, 12.0f, 0.301f, "SAE"    },
        { 1134, "json/key",            4, 41.1f, 0.061f, "mlp"    },
        { 2202, "url/path",            7, 22.5f, 0.184f, "logit"  },
        {   42, "refusal_dir",         8,  9.8f, 0.421f, "linear" },
        {  998, "sentiment_neg",      10, 26.8f, 0.114f, "linear" },
        {  111, "sentiment_pos",      10, 19.4f, 0.182f, "linear" },
        { 4119, "list-bullet",         4, 15.2f, 0.244f, "SAE"    },
        { 8412, "sql/select",          6, 12.0f, 0.272f, "SAE"    },
        { 9221, "paren-depth-3",       5, 33.4f, 0.091f, "mlp"    },
        { 7100, "next-token-pred",    11, 21.8f, 0.171f, "logit"  },
        { 6210, "multilingual",        9, 27.2f, 0.130f, "linear" },
    };
#else
    return {};
#endif
}

FeatureCard MockModel::getFeatureCard([[maybe_unused]] int featureId) {
#if LLOB_USE_MOCK_DATA
    return {
        featureId, "L08.mlp_out", "SAE feature (top-K, k=64)",
        17.4f, 4182, 2.184f,
        "+the (+0.34)  +to (+0.21)  +of (+0.18)",
        "-of (-0.41)  -if (-0.22)  -in (-0.18)",
    };
#else
    return {};
#endif
}

std::vector<FeatureExample>
MockModel::getFeatureExamples([[maybe_unused]] int featureId, int k) {
#if LLOB_USE_MOCK_DATA
    static const FeatureExample all[] = {
        { 4.21f, "...each attention head ", "attends", " to a subset of the previous tokens." },
        { 3.84f, "...the model ",           "attends", " selectively when computing..." },
        { 3.62f, "...the operator ",        "attends", " to relevant context windows..." },
        { 2.94f, "...where ",               "each",    " query token gates the values..." },
        { 2.81f, "...in the next layer it ","attends", " to a different position" },
    };
    std::vector<FeatureExample> out;
    for (int i = 0; i < std::min<int>(k, int(std::size(all))); ++i) out.push_back(all[i]);
    return out;
#else
    return {};
#endif
}

std::vector<CoFiringEntry>
MockModel::getCoFiringFeatures([[maybe_unused]] int featureId,
                                [[maybe_unused]] float threshold) {
#if LLOB_USE_MOCK_DATA
    return {
        { 471,  "attention/head_pattern", 0.841f },
        { 1923, "induction/A->B",         0.712f },
        { 3014, "self-reference",         0.481f },
        {   80, "syntax/clause",          0.412f },
    };
#else
    return {};
#endif
}

SAETrainingMetrics MockModel::getSAETrainingMetrics([[maybe_unused]] std::string_view saeId) {
#if LLOB_USE_MOCK_DATA
    return {
        "L08.mlp_out", 412'000, 1'000'000,
        0.412f, 34.2f, 22, 16384, 0.864f,
        {1.2f,1.0f,0.84f,0.71f,0.62f,0.55f,0.50f,0.47f,0.45f,0.43f,0.42f,0.41f},
        {80,72,65,58,52,47,43,40,38,36,35,34},
        {120,82,61,42,38,31,28,26,25,24,23,22},
        {0.41f,0.52f,0.61f,0.69f,0.74f,0.78f,0.81f,0.83f,0.84f,0.85f,0.86f,0.86f},
    };
#else
    return {};
#endif
}

std::vector<ProbeEntry> MockModel::getProbeLibrary() {
#if LLOB_USE_MOCK_DATA
    return {
        {"L","accent","refusal_dir",      "L08.resid_post", 0.91f, true },
        {"L","accent","truthfulness",     "L11.resid_post", 0.81f, false},
        {"L","accent","code_v_prose",     "L06.resid_post", 0.94f, false},
        {"S","warn",  "SAE/feature_2381", "L08.mlp_out",    0.62f, false},
        {"S","warn",  "SAE/feature_471",  "L07.mlp_out",    0.58f, false},
        {"P","dim",   "multilingual",     "L09.resid_post", 0.74f, false},
    };
#else
    return {};
#endif
}

ProbeTrainState MockModel::getProbeTrainState([[maybe_unused]] std::string_view name) {
#if LLOB_USE_MOCK_DATA
    if (m_probe_training) {
        m_probe_step = std::min(512, m_probe_step + 8);
        if (m_probe_step >= 512) m_probe_training = false;
    }
    return {
        m_probe_training, m_probe_step, 512,
        0.92f, 0.89f, 0.03f,
        {0.51f,0.55f,0.6f,0.65f,0.71f,0.78f,0.83f,0.86f,0.88f,0.9f,0.91f,0.91f},
    };
#else
    return {};
#endif
}

std::vector<ExportEntry> MockModel::getRecentExports() {
#if LLOB_USE_MOCK_DATA
    return {
        { "2026-05-06_14:03", "refusal_probe_L08.npz" },
        { "2026-05-06_13:41", "ablation_set_v3.json" },
        { "2026-05-06_12:18", "feature_2381_examples.csv" },
    };
#else
    return {};
#endif
}

// ──────────────────────────────────────────────────────────────────────────
// Training
// ──────────────────────────────────────────────────────────────────────────

TrainingState MockModel::getTrainingState() {
#if LLOB_USE_MOCK_DATA
    // Free-running step counter so the workspace shows live numbers.
    static int step = 24'180;
    static auto last = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last).count() >= 200) {
        ++step; last = now;
    }
    return { /*running=*/true, step, 100'000 };
#else
    return {};
#endif
}

std::vector<TrainingMetricCard> MockModel::getTrainingMetrics() {
#if LLOB_USE_MOCK_DATA
    return {
        { "LOSS",      "2.184",   "-0.012",  "good" },
        { "VAL LOSS",  "2.241",   "-0.008",  "good" },
        { "LR",        "3.0e-4",  "cosine",  ""     },
        { "TOK/SEC",   "184,210", "+1.2%",   "good" },
        { "GPU UTIL",  "94.2%",   "A100x8",  "good" },
        { "GRAD NORM", "1.48",    "clipped", "warn" },
        { "EPOCH",     "0.40",    "/3.00",   ""     },
    };
#else
    return {};
#endif
}

LossCurve MockModel::getTrainingLoss(int maxSteps) {
#if LLOB_USE_MOCK_DATA
    const int n = std::clamp(maxSteps, 8, 512);
    LossCurve c; c.train.resize(n); c.val.resize(n);
    for (int i = 0; i < n; ++i) {
        c.train[i] = 3.2f * std::exp(-i * 0.05f) + 0.3f + std::sin(i * 0.4f) * 0.05f;
        c.val[i]   = c.train[i] + 0.06f + std::sin(i * 0.3f) * 0.02f;
    }
    return c;
#else
    return {};
#endif
}

std::vector<float> MockModel::getGradFlowPerLayer() {
#if LLOB_USE_MOCK_DATA
    // Matches the active model — caller (training workspace) trims to nLayers.
    std::vector<float> v(64);
    for (int L = 0; L < int(v.size()); ++L)
        v[L] = std::exp(-L * 0.02f) * (0.6f + std::sin(float(L)) * 0.1f);
    return v;
#else
    return {};
#endif
}

std::vector<std::vector<float>> MockModel::getPerLayerLoss(int maxSteps) {
#if LLOB_USE_MOCK_DATA
    const int steps = std::clamp(maxSteps, 8, 256);
    std::vector<std::vector<float>> rows(64, std::vector<float>(steps, 0.0f));
    for (int L = 0; L < int(rows.size()); ++L) {
        for (int i = 0; i < steps; ++i) {
            rows[L][i] = (3.0f - L * 0.05f) * std::exp(-i * 0.06f)
                       + (0.4f - L * 0.005f)
                       + std::sin(i * 0.6f + L) * 0.04f;
        }
    }
    return rows;
#else
    return {};
#endif
}

// ──────────────────────────────────────────────────────────────────────────
// Finetune
// ──────────────────────────────────────────────────────────────────────────

LoRAConfig MockModel::getLoRAConfig() {
#if LLOB_USE_MOCK_DATA
    return { 16, 32.0f, 0.05f, "q_proj, v_proj, o_proj", 4.72f, 0.006f, "LoRA + RSLoRA scale" };
#else
    return {};
#endif
}

OptimizerConfig MockModel::getOptimizerConfig() {
#if LLOB_USE_MOCK_DATA
    return { "AdamW8bit", 2.0e-4f, 0.9f, 0.95f, 100, "cosine", 0.01f };
#else
    return {};
#endif
}

DataConfig MockModel::getDataConfig() {
#if LLOB_USE_MOCK_DATA
    return { "sft/instruction_v3", 128, 2, 4096, true };
#else
    return {};
#endif
}

EvalDiffMetric MockModel::getEvalDiff([[maybe_unused]] std::string_view bench) {
#if LLOB_USE_MOCK_DATA
    return { std::string(bench), 0.412f, 0.448f, 0.036f };
#else
    return {};
#endif
}

std::vector<float> MockModel::getEvalLossCurve() {
#if LLOB_USE_MOCK_DATA
    return {2.81f,2.62f,2.44f,2.31f,2.18f,2.06f,1.98f,1.91f,1.86f,1.82f,1.79f,1.78f};
#else
    return {};
#endif
}

ABSample MockModel::getABSample() {
#if LLOB_USE_MOCK_DATA
    return {
        "\"Explain the residual stream\"",
        "\"The residual stream is a connection that...\"",
        "\"The residual stream is the additive backbone...\"",
    };
#else
    return {};
#endif
}

std::vector<std::vector<float>>
MockModel::getDeltaWHeatmap(int numLayers, int numComponents) {
#if LLOB_USE_MOCK_DATA
    std::vector<std::vector<float>> data(numLayers, std::vector<float>(numComponents));
    for (int L = 0; L < numLayers; ++L)
        for (int c = 0; c < numComponents; ++c)
            data[L][c] = std::abs(std::sin(L * 0.7f + c * 1.3f)) * (1.0f - L * 0.02f);
    return data;
#else
    (void)numLayers; (void)numComponents;
    return {};
#endif
}

std::vector<std::string> MockModel::getDeltaWComponentNames() {
#if LLOB_USE_MOCK_DATA
    return { "Q","K","V","O","g_in","g_up","g_out","dn","rn1","rn2","b1","b2" };
#else
    return {};
#endif
}

// ──────────────────────────────────────────────────────────────────────────
// Datasets
// ──────────────────────────────────────────────────────────────────────────

std::vector<DatasetSummary> MockModel::getDatasets() {
#if LLOB_USE_MOCK_DATA
    return {
        { "the_pile_v2",        std::int64_t(412) << 30, std::int64_t(82'400'000'000ll) },
        { "fineweb_edu",        std::int64_t(132) << 30, std::int64_t(28'100'000'000ll) },
        { "sft/instruction_v3", std::int64_t(2'400'000'000ll), std::int64_t(128'000'000) },
        { "harmful_v2",         std::int64_t(42'000'000),     std::int64_t(2'000'000)   },
        { "mmlu_eval",          std::int64_t(12'000'000),     std::int64_t(500'000)     },
        { "truthful_qa",        std::int64_t( 8'000'000),     std::int64_t(300'000)     },
    };
#else
    return {};
#endif
}

DatasetSample MockModel::getSample([[maybe_unused]] std::string_view dataset, int id) {
#if LLOB_USE_MOCK_DATA
    DatasetSample s;
    s.sample_id = id;
    s.doc_id    = "82a17e";
    s.source    = "arxiv/cs.CL";
    s.text =
        "When the transformer processes a sentence, each attention head attends "
        "to a subset of the previous tokens. The residual stream carries "
        "information from layer to layer through the network, with each block "
        "adding (and only ever adding) its contribution to the running sum.\n\n"
        "The self-attention sublayer computes Q, K, V from the input residual:\n"
        "  Q = LN(x) W_Q\n  K = LN(x) W_K\n  V = LN(x) W_V";
    // Two highlight spans into s.text — primary on "The residual stream",
    // secondary on "contribution".
    if (auto p = s.text.find("The residual stream"); p != std::string::npos)
        s.spans.push_back({ int(p), int(p + std::string_view("The residual stream").size()), 0 });
    if (auto p = s.text.find("contribution"); p != std::string::npos)
        s.spans.push_back({ int(p), int(p + std::string_view("contribution").size()), 1 });
    return s;
#else
    return {};
#endif
}

DatasetSampleStats MockModel::getSampleStats([[maybe_unused]] std::string_view dataset,
                                              [[maybe_unused]] int id) {
#if LLOB_USE_MOCK_DATA
    return { 482, 4.21f, 3.84f, 2.07f, "f2381 attends_to_subset" };
#else
    return {};
#endif
}

DatasetDistribution MockModel::getDatasetDistribution([[maybe_unused]] std::string_view dataset) {
#if LLOB_USE_MOCK_DATA
    DatasetDistribution d;
    d.doc_length_histogram = {12,18,28,42,68,108,140,160,168,156,128,98,72,48,32,22,14,8,5,3};
    d.source_mix = {
        { "common_crawl", 0.62f, "accent" },
        { "arxiv",        0.14f, "info"   },
        { "github",       0.11f, "warn"   },
        { "books",        0.08f, "good"   },
        { "wiki",         0.05f, "magenta"},
    };
    return d;
#else
    return {};
#endif
}

std::vector<float> MockModel::getTokenIds([[maybe_unused]] std::string_view dataset,
                                           [[maybe_unused]] int id, int n) {
#if LLOB_USE_MOCK_DATA
    Mulberry32 rng(4182u);
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) v[i] = (rng.next() - 0.5f) * 1.4f;
    return v;
#else
    return {};
#endif
}

// ──────────────────────────────────────────────────────────────────────────
// Raw tensors
// ──────────────────────────────────────────────────────────────────────────

std::vector<TensorMeta> MockModel::getStateDict() {
#if LLOB_USE_MOCK_DATA
    auto tm = [](const char* name, std::vector<int> shape, std::int64_t bytes) {
        TensorMeta m; m.name = name; m.dtype = "fp16";
        m.shape = std::move(shape); m.stride = { m.shape.empty() ? 1 : m.shape.back(), 1 };
        m.contiguous = true; m.device = "cuda:0"; m.size_bytes = bytes;
        return m;
    };
    return {
        tm("embed.weight",                {32000, 384}, 24'576'000),
        tm("pos_embed.freqs",             {2048, 64},     262'144),
        tm("blocks.8.attn.W_Q.weight",    {384, 384},     294'912),
        tm("blocks.8.attn.W_K.weight",    {384, 384},     294'912),
        tm("blocks.8.attn.W_V.weight",    {384, 384},     294'912),
        tm("blocks.8.attn.W_O.weight",    {384, 384},     294'912),
        tm("blocks.8.attn.b_Q",           {384},                768),
        tm("blocks.8.mlp.W_in.weight",    {384, 1536},  1'179'648),
        tm("blocks.8.mlp.W_out.weight",   {1536, 384},  1'179'648),
        tm("blocks.8.norm1.weight",       {384},                768),
        tm("final_norm.weight",           {384},                768),
        tm("unembed.weight",              {384, 32000}, 24'576'000),
    };
#else
    return {};
#endif
}

TensorMeta MockModel::getTensorMeta([[maybe_unused]] std::string_view name) {
#if LLOB_USE_MOCK_DATA
    for (const auto& m : getStateDict()) if (m.name == name) return m;
    TensorMeta fallback; fallback.name = std::string(name);
    return fallback;
#else
    return {};
#endif
}

std::vector<float> MockModel::getWeightSlice([[maybe_unused]] std::string_view name,
                                              int offset, int n) {
#if LLOB_USE_MOCK_DATA
    Mulberry32 rng(hashString(name) ^ static_cast<std::uint32_t>(offset));
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) v[i] = (rng.next() - 0.5f) * 1.4f;
    return v;
#else
    return {};
#endif
}

std::vector<int> MockModel::getWeightHistogram([[maybe_unused]] std::string_view name, int bins) {
#if LLOB_USE_MOCK_DATA
    Mulberry32 rng(hashString(name));
    std::vector<int> h(bins, 0);
    for (int i = 0; i < 8000; ++i) {
        float g = (rng.next() + rng.next() + rng.next() + rng.next() - 2.0f) / 1.2f;
        g = std::clamp(g, -1.0f, 1.0f);
        const int idx = std::min(bins - 1, std::max(0, static_cast<int>((g + 1.0f) / 2.0f * bins)));
        ++h[idx];
    }
    return h;
#else
    return {};
#endif
}

TensorStats MockModel::getTensorStats([[maybe_unused]] std::string_view name) {
#if LLOB_USE_MOCK_DATA
    return { -0.412f, 0.408f, 0.001f, 0.142f, 9.41e2f, 24.82f, 0.412f, 512, 768 };
#else
    return {};
#endif
}

std::vector<float> MockModel::getSingularValues([[maybe_unused]] std::string_view name, int k) {
#if LLOB_USE_MOCK_DATA
    static const float all[] = {24,18,14,11,9,7,6,5,4.2f,3.6f,3.1f,2.7f,2.4f,2.1f,1.9f,1.7f};
    std::vector<float> out;
    for (int i = 0; i < std::min<int>(k, int(std::size(all))); ++i) out.push_back(all[i]);
    return out;
#else
    return {};
#endif
}

std::vector<std::vector<float>>
MockModel::getTensorSlice2D([[maybe_unused]] std::string_view name,
                             [[maybe_unused]] int axis0, [[maybe_unused]] int axis1,
                             int rows, int cols) {
#if LLOB_USE_MOCK_DATA
    std::vector<std::vector<float>> data(rows, std::vector<float>(cols));
    for (int i = 0; i < rows; ++i) for (int j = 0; j < cols; ++j)
        data[i][j] = std::sin(i * 0.4f + j * 0.3f) * std::cos(i * 0.2f - j * 0.15f);
    return data;
#else
    (void)rows; (void)cols;
    return {};
#endif
}

std::vector<std::vector<float>>
MockModel::getDiffSlice2D([[maybe_unused]] std::string_view name, int rows, int cols) {
#if LLOB_USE_MOCK_DATA
    std::vector<std::vector<float>> data(rows, std::vector<float>(cols));
    for (int i = 0; i < rows; ++i) for (int j = 0; j < cols; ++j)
        data[i][j] = std::sin(i * 0.5f + j * 0.4f) * 0.1f
                   * std::exp(-std::abs(i - rows * 0.5f) * 0.2f);
    return data;
#else
    (void)rows; (void)cols;
    return {};
#endif
}

DiffStats MockModel::getDiffStats([[maybe_unused]] std::string_view name) {
#if LLOB_USE_MOCK_DATA
    return { 0.084f, 0.9912f, {14,8,5,3,2.4f,2,1.7f,1.5f} };
#else
    return {};
#endif
}

// ──────────────────────────────────────────────────────────────────────────
// Engine metrics
// ──────────────────────────────────────────────────────────────────────────

EngineMetrics MockModel::getEngineMetrics() {
#if LLOB_USE_MOCK_DATA
    EngineMetrics m;
    m.log_rate_per_min  = 12;
    m.warn_rate_per_min = 4;
    m.err_rate_per_min  = 0;
    m.cuda_mem_used_GB  = 38.4f;
    m.cuda_mem_total_GB = 80.0f;
    m.cpu_pct           = 11.2f;
    m.fwd_time_ms       = 14.2f;
    m.fps               = 60.0f;
    m.device            = "cuda:0 A100";
    m.dtype             = "fp16";
    return m;
#else
    return {};
#endif
}

std::vector<LogEntry> MockModel::drainEngineLogs() {
    // No engine bridged.  A real backend implementation pops from its
    // own ring buffer here; the UI fans the result through Logger so the
    // entries land in both the in-app console and the on-disk log.
    return {};
}

}  // namespace llob
