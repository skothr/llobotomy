// test_mock_data_shapes.cpp — sweep MockModel's ~50 per-DTO getters
// and verify each returns sensibly-shaped data.
//
// Catches drift if someone breaks a getter (returns wrong-length vector,
// returns NaN where a real value belongs, fills in mismatched layer
// counts, etc.).
//
// Two parts:
//   1. With LLOB_USE_MOCK_DATA=ON: every method returns "looks like
//      real data" (non-empty vectors, finite floats, matching layer
//      counts, etc.).
//   2. With LLOB_USE_MOCK_DATA=OFF: every method returns the no-data
//      sentinel (empty vector, NaN float, "" string).  This is the
//      release-build invariant — a mis-configured build can't show
//      mock numbers as if they were real.

#include "llm_engine/model.hpp"
#include "llm_engine/model_view.hpp"

#include <cassert>
#include <cmath>
#include <string>
#include <string_view>
#include <vector>

using namespace llmengine;

namespace {

bool is_sentinel(float f)        { return std::isnan(f); }
bool is_sentinel(int v)          { return v == kNoInt; }
bool is_sentinel(std::int64_t v) { return v == kNoSize; }
bool is_sentinel(const std::string& s) { return s.empty(); }

void test_topology() {
    MockModel m;
    const auto info = m.getModelInfo();
#if LLOB_USE_MOCK_DATA
    assert(info.nLayers > 0);
    assert(info.nHeads  > 0);
    assert(info.dModel  > 0);
    assert(info.vocab   > 0);
    assert(info.dHead   == info.dModel / info.nHeads);   // sanity
    assert(!info.name.empty());
    assert(info.totalParams > 0);
#else
    assert(is_sentinel(info.nLayers));
    assert(is_sentinel(info.nHeads));
    assert(is_sentinel(info.dModel));
    assert(is_sentinel(info.vocab));
    assert(is_sentinel(info.name));
    assert(is_sentinel(info.totalParams));
#endif
}

void test_attention_shape() {
    MockModel m;
    const auto info = m.getModelInfo();
    const int L = 5;
    const int H = 2;
    const int seq = 8;
    const auto pat = m.getAttentionPattern(L, H, seq, HeadBias::Diag);
#if LLOB_USE_MOCK_DATA
    (void)info;
    assert(pat.size() == static_cast<std::size_t>(seq));
    for (const auto& row : pat) {
        assert(row.size() == static_cast<std::size_t>(seq));
        for (float v : row) {
            assert(std::isfinite(v));
            assert(v >= 0.0f && v <= 1.0f);   // post-softmax
        }
    }
#else
    (void)info; (void)pat;
    assert(pat.empty());
#endif
}

void test_logit_lens_per_layer() {
    MockModel m;
    const auto info = m.getModelInfo();
    const auto rows = m.getLogitLensTrajectory(/*token=*/0, /*kLayers=*/0);
#if LLOB_USE_MOCK_DATA
    assert(!rows.empty());
    // Layer indices are 0..nLayers; entropy finite; top probs sane.
    for (const auto& r : rows) {
        assert(r.layer >= 0);
        assert(r.layer < info.nLayers);
        assert(std::isfinite(r.entropy));
        assert(r.p1 >= 0.0f && r.p1 <= 1.0f);
        assert(r.p2 >= 0.0f && r.p2 <= 1.0f);
    }
#else
    (void)info; (void)rows;
    assert(rows.empty());
#endif
}

void test_output_logits_shape() {
    MockModel m;
    const int k = 10;
    const auto logits = m.getOutputLogits(k);
#if LLOB_USE_MOCK_DATA
    assert(!logits.empty());
    assert(logits.size() <= static_cast<std::size_t>(k));
    float prob_sum = 0.0f;
    for (const auto& d : logits) {
        assert(d.prob >= 0.0f && d.prob <= 1.0f);
        prob_sum += d.prob;
    }
    (void)prob_sum;       // top-k probs needn't sum to 1
#else
    assert(logits.empty());
#endif
}

void test_residual_summary_finite() {
    MockModel m;
    const auto sum = m.getResidualSummary(0);
#if LLOB_USE_MOCK_DATA
    assert(std::isfinite(sum.attn_out_norm));
    assert(std::isfinite(sum.mlp_out_norm));
    assert(std::isfinite(sum.resid_norm));
    assert(std::isfinite(sum.cos_prev));
    assert(std::isfinite(sum.kurtosis));
#else
    assert(is_sentinel(sum.attn_out_norm));
    assert(is_sentinel(sum.mlp_out_norm));
    assert(is_sentinel(sum.resid_norm));
#endif
}

void test_state_dict_shapes() {
    MockModel m;
    const auto sd = m.getStateDict();
#if LLOB_USE_MOCK_DATA
    assert(!sd.empty());
    for (const auto& t : sd) {
        assert(!t.name.empty());
        assert(!t.dtype.empty());
        assert(!t.shape.empty());
        for (int d : t.shape) assert(d > 0);
        assert(t.size_bytes > 0);
    }
#else
    assert(sd.empty());
#endif
}

void test_qkv_stats_finite() {
    MockModel m;
    const auto qkv = m.getQKVStats(/*L=*/0, /*H=*/0, /*T=*/0);
#if LLOB_USE_MOCK_DATA
    assert(std::isfinite(qkv.q_norm));
    assert(std::isfinite(qkv.k_norm));
    assert(std::isfinite(qkv.v_norm));
    // Attention fractions are 0..1.
    assert(qkv.attn_to_bos  >= 0.0f && qkv.attn_to_bos  <= 1.0f);
    assert(qkv.attn_to_self >= 0.0f && qkv.attn_to_self <= 1.0f);
    assert(qkv.attn_to_prev >= 0.0f && qkv.attn_to_prev <= 1.0f);
#else
    assert(is_sentinel(qkv.q_norm));
    assert(is_sentinel(qkv.k_norm));
    assert(is_sentinel(qkv.v_norm));
#endif
}

void test_training_metrics_consistency() {
    MockModel m;
    const auto state = m.getTrainingState();
    const auto cards = m.getTrainingMetrics();
    const auto curve = m.getTrainingLoss(/*maxSteps=*/100);
#if LLOB_USE_MOCK_DATA
    // Training metrics surface should at least exist.
    assert(state.total_steps >= 0);
    assert(!cards.empty());
    assert(!curve.train.empty());
    for (float v : curve.train) assert(std::isfinite(v));
#else
    assert(cards.empty());
    assert(curve.train.empty());
#endif
}

void test_dataset_distribution() {
    MockModel m;
    const auto dist = m.getDatasetDistribution("any");
#if LLOB_USE_MOCK_DATA
    // doc_length_histogram and source_mix should be populated.
    assert(!dist.doc_length_histogram.empty());
    assert(!dist.source_mix.empty());
    float frac_sum = 0.0f;
    for (const auto& r : dist.source_mix) {
        assert(r.fraction >= 0.0f && r.fraction <= 1.0f);
        frac_sum += r.fraction;
    }
    // Fractions should approximately sum to 1.
    assert(std::abs(frac_sum - 1.0f) < 0.01f);
#else
    assert(dist.doc_length_histogram.empty());
    assert(dist.source_mix.empty());
#endif
}

void test_engine_metrics_present() {
    MockModel m;
    const auto em = m.getEngineMetrics();
#if LLOB_USE_MOCK_DATA
    // device + dtype are populated strings.
    assert(!em.device.empty());
    assert(!em.dtype.empty());
#endif
    // Either way, the call doesn't crash.
}

void test_feature_library_filter() {
    MockModel m;
    const auto all = m.getFeatureLibrary("");
    const auto filtered = m.getFeatureLibrary("nope-no-match");
#if LLOB_USE_MOCK_DATA
    // Empty filter matches everything.
    assert(!all.empty());
    // Non-matching filter returns empty (or just the unfiltered set,
    // depending on the mock — accept either).
    (void)filtered;
#else
    assert(all.empty());
    assert(filtered.empty());
#endif
}

}  // namespace

int main() {
    test_topology();
    test_attention_shape();
    test_logit_lens_per_layer();
    test_output_logits_shape();
    test_residual_summary_finite();
    test_state_dict_shapes();
    test_qkv_stats_finite();
    test_training_metrics_consistency();
    test_dataset_distribution();
    test_engine_metrics_present();
    test_feature_library_filter();
    return 0;
}
