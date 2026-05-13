// test_real_gguf.cpp — deep test against a real .gguf file.
//
// Verifies the GgufInspectorEngine handles bytes produced by an actual
// HF→GGUF conversion (not just our hand-constructed test buffers).
// Catches drift between the parser and real-world GGUF emitters.
//
// Skip behaviour: when LLOB_DEEP_GGUF_PATH is unset, exit 0 silently.
// This is the deep-tier contract — test fixtures live outside the repo
// and CI doesn't have model weights checked in.

#include "llm_engine/gguf_inspector_engine.hpp"
#include "llm_engine/model.hpp"
#include "llm_engine/model_view.hpp"
#include "llm_engine/tensor_handle.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace llmengine;

int main() {
    const char* path = std::getenv("LLOB_DEEP_GGUF_PATH");
    if (!path || !*path) {
        std::printf("test_real_gguf: skipped — LLOB_DEEP_GGUF_PATH not set\n");
        return 0;
    }

    GgufInspectorEngine eng;
    auto r = eng.loadCheckpoint(path);
    if (!r.ok) {
        std::fprintf(stderr,
                     "test_real_gguf: load failed: %s\n", r.error.c_str());
        return 1;
    }

    const auto& v = eng.view();

    // Provenance — format should be "gguf"; path should round-trip.
    assert(v.provenance.format == "gguf");
    assert(v.provenance.path == std::string(path));

    // Topology — every transformer GGUF should have these populated.
    assert(v.topology.nLayers > 0);
    assert(v.topology.nHeads  > 0);
    assert(v.topology.dModel  > 0);
    assert(v.topology.vocab   > 0);
    std::printf("test_real_gguf: arch=%s layers=%d heads=%d d_model=%d vocab=%d\n",
                v.topology.name.c_str(),
                v.topology.nLayers, v.topology.nHeads,
                v.topology.dModel, v.topology.vocab);

    // Tensor registry — non-empty, every entry valid().
    assert(!v.tensors.empty());
    std::printf("test_real_gguf: %zu tensors enumerated\n", v.tensors.size());

    std::size_t readable = 0;
    std::size_t valid_only = 0;
    std::size_t unsupported_dtype = 0;
    for (const auto& h : v.tensors.all) {
        assert(h.valid());                  // shape + dtype known
        ++valid_only;
        if (h.readable()) {
            ++readable;
        } else {
            // No source — shouldn't happen for a properly-loaded file.
            std::fprintf(stderr, "  unreadable: %s (dtype=%d)\n",
                         h.name.c_str(), static_cast<int>(h.dtype));
        }
        // Block-quantised dtypes have no dequantiser yet; bpe == 0 is
        // expected for Q4_K / Q4_0 / Q8_0 in this build.
        if (dtype_element_bytes(h.dtype) == 0) ++unsupported_dtype;
    }
    assert(readable == valid_only);
    std::printf("test_real_gguf: %zu readable handles (%zu with unsupported dtype)\n",
                readable, unsupported_dtype);

    // Find a supported-dtype tensor and verify read_slice returns
    // finite values.  Embedding-like tensors (large rows) make a good
    // target — they're usually F16/F32 even in quantised models.
    const TensorHandle* sample = nullptr;
    for (const auto& h : v.tensors.all) {
        if (dtype_element_bytes(h.dtype) > 0 && h.element_count() >= 16) {
            sample = &h;
            break;
        }
    }
    if (sample) {
        auto bytes = sample->read_slice(0, 16);
        assert(bytes.size() == 16);
        // Every value should be finite (not NaN / inf).  Catches a
        // dequantiser bug that produces garbage.
        for (float f : bytes) {
            assert(std::isfinite(f));
        }
        std::printf("test_real_gguf: read_slice('%s', 0, 16) returned 16 finite floats\n",
                    sample->name.c_str());
    } else {
        std::printf("test_real_gguf: no supported-dtype tensor found (file is fully block-quantised?)\n");
    }

    // Capabilities — GgufInspector advertises topology + state_dict + tokenizer.
    const auto caps = eng.getCapabilities();
    assert(caps.has_topology);
    assert(caps.has_state_dict);
    // tokenizer wiring depends on whether the GGUF carries it; not asserted.

    // unloadCheckpoint clears the view.
    eng.unloadCheckpoint();
    assert(eng.view().topology.nLayers == kNoInt);
    assert(eng.view().tensors.empty());

    std::printf("test_real_gguf: PASS\n");
    return 0;
}
