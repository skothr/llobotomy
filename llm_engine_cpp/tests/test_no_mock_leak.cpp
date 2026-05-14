// test_no_mock_leak.cpp
//
// Regression: concrete backends (HFProxyEngine, GgufInspectorEngine,
// LlamaCppEngine) must NEVER emit MockModel's deterministic synthetic
// data — not even under LLOB_USE_MOCK_DATA=ON.  Each backend inherits
// Model directly; any getter it doesn't override should return the
// no-data sentinel for its type.
//
// Before this contract was enforced, backends inherited MockModel, so an
// unimplemented getter silently fell through to mock data in dev builds.
// That was misleading: workspaces showed plausible-looking numbers for
// features a real backend hadn't actually wired up.  The fix was to
// inherit Model and lift the empty/sentinel implementations there.
//
// This test compiles in any build configuration:
//   * MOCK_DATA=ON  — catches a regression by failing if a backend
//     reverts to inheriting MockModel (synthetic vectors would re-appear).
//   * MOCK_DATA=OFF — trivially passes (everything is empty anyway).
//
// We exercise unloaded backends (no loadCheckpoint call).  For all three
// the topology, capability mirror, captures, surgery, derived cache, and
// every DTO getter must be in the no-data state.

#include "llm_engine/gguf_inspector_engine.hpp"
#include "llm_engine/hf_proxy_engine.hpp"
#include "llm_engine/llama_cpp_engine.hpp"
#include "llm_engine/model.hpp"
#include "llm_engine/model_view.hpp"

#include <cmath>
#include <cstdio>
#include <string>

using namespace llmengine;

namespace {

int g_fail = 0;
const char* g_backend = "?";

#define EXPECT(cond, msg)                                                                          \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::fprintf(stderr, "FAIL [%s]: %s\n", g_backend, msg);                               \
            ++g_fail;                                                                              \
        }                                                                                          \
    } while (0)

// Topology fields default to kNoInt / kNoFloat sentinels in ModelInfo{}.
// A backend that "loaded nothing" must return that exact shape.
void check_topology(Model& m) {
    auto info = m.getModelInfo();
    EXPECT(info.nLayers == kNoInt, "getModelInfo().nLayers leaked non-sentinel");
    EXPECT(info.nHeads == kNoInt, "getModelInfo().nHeads leaked non-sentinel");
    EXPECT(info.dModel == kNoInt, "getModelInfo().dModel leaked non-sentinel");
    EXPECT(info.vocab == kNoInt, "getModelInfo().vocab leaked non-sentinel");
    EXPECT(std::isnan(info.ropeTheta), "getModelInfo().ropeTheta leaked non-sentinel");

    const ModelView& view = m.view();
    EXPECT(view.topology.nLayers == kNoInt, "view().topology.nLayers leaked non-sentinel");
    EXPECT(view.topology.dModel == kNoInt, "view().topology.dModel leaked non-sentinel");
    EXPECT(view.tensors.size() == 0, "view().tensors leaked synthetic entries");
    EXPECT(view.captures.empty(), "view().captures leaked synthetic bundles");
    EXPECT(view.current.load() == nullptr, "view().current leaked synthetic capture");
    EXPECT(view.surgery.ablated_heads.empty(), "view().surgery leaked synthetic heads");
    EXPECT(view.surgery.ablated_components.empty(), "view().surgery leaked synthetic components");
}

// Every per-DTO getter must return the no-data sentinel for its type.
// NaN for float, 0/empty for collections, default-constructed for structs.
void check_per_dto(Model& m) {
    EXPECT(m.getCurrentTokens().empty(), "getCurrentTokens leaked tokens");
    EXPECT(m.getParamBreakdown(0).empty(), "getParamBreakdown leaked rows");
    EXPECT(m.getAttentionPattern(0, 0, 16, HeadBias::Diag).empty(),
           "getAttentionPattern leaked a matrix");
    EXPECT(m.getActivation(0, 0, 16).empty(), "getActivation leaked a vector");
    EXPECT(std::isnan(m.getHeadNorm(0, 0)), "getHeadNorm leaked a real number");
    EXPECT(std::isnan(m.getComponentNorm(0, "attn")), "getComponentNorm leaked a real number");

    auto live = m.getLiveActivations(0);
    EXPECT(std::isnan(live.attn_out_norm), "getLiveActivations.attn_out_norm leaked");
    EXPECT(std::isnan(live.resid_post_norm), "getLiveActivations.resid_post_norm leaked");

    auto rc = m.getResidualContribution(0);
    EXPECT(std::isnan(rc.attn), "getResidualContribution.attn leaked");
    EXPECT(std::isnan(rc.mlp), "getResidualContribution.mlp leaked");

    auto rs = m.getResidualSummary(0);
    EXPECT(std::isnan(rs.attn_out_norm), "getResidualSummary.attn_out_norm leaked");
    EXPECT(std::isnan(rs.resid_norm), "getResidualSummary.resid_norm leaked");

    EXPECT(m.getLogitLensTrajectory(0, 10).empty(), "getLogitLensTrajectory leaked rows");
    EXPECT(m.getOutputLogits(5).empty(), "getOutputLogits leaked a top-k");
    EXPECT(m.getMlpFeatures(0, 8).empty(), "getMlpFeatures leaked activations");
    EXPECT(m.getTokenLossPerToken(0).empty(), "getTokenLossPerToken leaked a curve");
    EXPECT(m.getSurprisalDelta().empty(), "getSurprisalDelta leaked deltas");
    EXPECT(m.getActiveProbes().empty(), "getActiveProbes leaked probes");

    auto qkv = m.getQKVStats(0, 0, 0);
    EXPECT(std::isnan(qkv.q_norm), "getQKVStats.q_norm leaked");
    EXPECT(std::isnan(qkv.k_norm), "getQKVStats.k_norm leaked");

    EXPECT(m.getHeadStats(0, 0).empty(), "getHeadStats leaked rows");

    EXPECT(m.getFeatureLibrary("").empty(), "getFeatureLibrary leaked entries");
    EXPECT(m.getFeatureExamples(0, 5).empty(), "getFeatureExamples leaked examples");
    EXPECT(m.getCoFiringFeatures(0, 0.0f).empty(), "getCoFiringFeatures leaked entries");
    EXPECT(m.getProbeLibrary().empty(), "getProbeLibrary leaked probes");
    EXPECT(m.getRecentExports().empty(), "getRecentExports leaked exports");

    EXPECT(m.getTrainingMetrics().empty(), "getTrainingMetrics leaked cards");
    EXPECT(m.getGradFlowPerLayer().empty(), "getGradFlowPerLayer leaked a curve");
    EXPECT(m.getPerLayerLoss(10).empty(), "getPerLayerLoss leaked a matrix");

    EXPECT(m.getEvalLossCurve().empty(), "getEvalLossCurve leaked a curve");
    EXPECT(m.getDeltaWHeatmap(4, 4).empty(), "getDeltaWHeatmap leaked a matrix");
    EXPECT(m.getDeltaWComponentNames().empty(), "getDeltaWComponentNames leaked names");

    EXPECT(m.getDatasets().empty(), "getDatasets leaked entries");
    EXPECT(m.getTokenIds("ds", 0, 16).empty(), "getTokenIds leaked ids");

    EXPECT(m.getStateDict().empty(), "getStateDict leaked tensor entries");
    EXPECT(m.getWeightSlice("blocks.0.attn.W_Q.weight", 0, 16).empty(),
           "getWeightSlice leaked a slice");
    EXPECT(m.getWeightHistogram("blocks.0.attn.W_Q.weight", 16).empty(),
           "getWeightHistogram leaked bins");
    EXPECT(m.getSingularValues("blocks.0.attn.W_Q.weight", 5).empty(),
           "getSingularValues leaked singular values");
    EXPECT(m.getTensorSlice2D("blocks.0.attn.W_Q.weight", 0, 1, 8, 8).empty(),
           "getTensorSlice2D leaked a matrix");

    // NOTE: drainEngineLogs() is intentionally NOT checked.  Engine logs
    // are real runtime records (connection failures, parse status, etc.),
    // not synthetic mock data.  An HFProxyEngine with a dead URL will
    // legitimately emit "connection refused" log lines from its heartbeat
    // worker — those are honest signal, not a leak.
}

// Capabilities for an unloaded backend should all be conservative-false
// (the Model::Capabilities{} default).  A backend that pre-asserts true
// without having loaded data lies to the UI.
void check_capabilities(Model& m) {
    auto caps = m.getCapabilities();
    EXPECT(!caps.has_topology, "capabilities.has_topology=true before load");
    EXPECT(!caps.has_tokenizer, "capabilities.has_tokenizer=true before load");
    EXPECT(!caps.has_state_dict, "capabilities.has_state_dict=true before load");
    EXPECT(!caps.has_attention, "capabilities.has_attention=true before load");
    EXPECT(!caps.has_residual, "capabilities.has_residual=true before load");
    EXPECT(!caps.has_logit_lens, "capabilities.has_logit_lens=true before load");
    EXPECT(!caps.has_token_stream, "capabilities.has_token_stream=true before load");
    EXPECT(!caps.has_captures, "capabilities.has_captures=true before load");
    EXPECT(!caps.has_intervention, "capabilities.has_intervention=true before load");
    EXPECT(!caps.has_weight_deltas, "capabilities.has_weight_deltas=true before load");
    EXPECT(!caps.has_training, "capabilities.has_training=true before load");
}

void check_unloaded(Model& m, const char* name) {
    g_backend = name;
    std::fprintf(stderr, "checking %s ...\n", name);
    check_topology(m);
    check_per_dto(m);
    check_capabilities(m);
}

}  // namespace

int main() {
    // HFProxyEngine constructed with a dead URL — never connects, so it
    // stays in the no-data state.
    {
        HFProxyEngine eng("http://127.0.0.1:1");
        check_unloaded(eng, "HFProxyEngine");
    }

    {
        GgufInspectorEngine eng;
        check_unloaded(eng, "GgufInspectorEngine");
    }

    {
        LlamaCppEngine eng;
        check_unloaded(eng, "LlamaCppEngine");
    }

    if (g_fail > 0) {
        std::fprintf(stderr, "no_mock_leak: %d FAILURES\n", g_fail);
        return 1;
    }
    std::fprintf(stderr, "no_mock_leak: OK (no backend leaks mock data)\n");
    return 0;
}
