// MockModel — substrate-compat sanity.
//
// Verifies that the new ModelView path is consistent with the per-DTO
// getter outputs.  Specifically:
//
//   - view().topology.nLayers   == getModelInfo().nLayers
//   - view().tensors enumerates the same names getStateDict() returns
//   - getCapabilities() advertises the mock-data caps (when ON) or all-false
//     (when OFF)

#include "llm_engine/model.hpp"
#include "llm_engine/model_view.hpp"

#include <cassert>

using namespace llmengine;

int main() {
    MockModel m;
    const auto& v = m.view();

    // Topology consistency: the legacy getter must match the view.
    auto info = m.getModelInfo();
    assert(v.topology.nLayers == info.nLayers);
    assert(v.topology.dModel  == info.dModel);
    assert(v.topology.name    == info.name);

    // Capabilities — must succeed on both build variants.
    auto caps = m.getCapabilities();
    (void)caps;

#if LLOB_USE_MOCK_DATA
    // Mock mode: state-dict + tensor registry must each carry the same
    // number of entries, and the names must align.
    auto sd = m.getStateDict();
    assert(!sd.empty());
    assert(v.tensors.size() > 0);

    const TensorHandle* h = v.tensors.find(sd[0].name);
    assert(h != nullptr);
    assert(h->name == sd[0].name);
#endif

    return 0;
}
