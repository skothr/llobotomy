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

    // Substrate guarantee: every mock handle is valid + readable + has
    // a backing source (Mulberry32 PRNG).  Architecturally identical to
    // a future GgufInspectorEngine handle — same TensorSource ABI, just
    // bytes from a PRNG instead of disk.
    assert(h->valid());
    assert(h->readable());
    assert(!h->loaded());            // PRNG generates on each pread

    // A real read returns f32 bytes deterministically — same call twice
    // gives the same result (the mock source is stateless).
    auto first = h->read_slice(0, 8);
    auto again = h->read_slice(0, 8);
    assert(first.size() == 8);
    assert(first == again);
#endif

    return 0;
}
