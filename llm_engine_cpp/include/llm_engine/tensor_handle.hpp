#pragma once
//
// TensorHandle — lazy, named, file-backed (or memory-backed) view of a
// tensor. Holds enough metadata to describe shape / dtype / location
// and read element slices on demand. No bytes loaded until read_slice
// is called.
//
// The handle is a POD value type. Multiple handles can share a
// TensorSource (e.g. every weight in a .gguf shares one mmap'd
// source).
//
// All reads dequantise to f32 — the UI layer wants float for plotting
// and statistics. Block-quantised dtypes that lack a dequantiser in
// this build return an empty vector (no exception); a warn line is
// pushed through the engine log so the user sees why.

#include "llm_engine/tensor_source.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace llmengine {

struct TensorHandle {
    std::shared_ptr<TensorSource> source;
    std::string                   name;          // canonical: "blocks.0.attn.W_Q.weight"
    DType                         dtype = DType::Unknown;
    std::vector<std::int64_t>     shape;
    std::vector<std::int64_t>     stride;        // in elements; empty ⇒ row-major
    std::size_t                   byte_offset = 0;
    std::size_t                   byte_length = 0;
    bool                          contiguous = true;

    // Total element count = product of shape. 0 when shape is empty.
    std::size_t element_count() const;

    // Read n elements starting at element_offset. Returns {} on any of:
    //   - source is null
    //   - dtype not supported by this build
    //   - element_offset + n exceeds element_count()
    std::vector<float> read_slice(std::size_t element_offset, std::size_t n) const;

    // 2-D slice for heatmap views. Requires shape.size() >= 2. Walks
    // rows individually so a row-major source can mmap-fast-path.
    // Out-of-range returns {}.
    std::vector<std::vector<float>>
    read_slice_2d(std::size_t row_offset, std::size_t row_count,
                  std::size_t col_offset, std::size_t col_count) const;

    // Tri-state liveness — pick the strictest predicate that fits.
    //
    //   valid()    — handle's configuration is internally consistent:
    //                dtype known, shape non-empty, byte_length matches.
    //                Doesn't require a source — useful for
    //                "header-parsed, source-not-yet-attached" handoffs.
    //
    //   readable() — read_slice() can produce bytes.  Requires a source
    //                AND valid().  A Mulberry32Source-backed handle is
    //                readable even though no bytes are in RAM.
    //
    //   loaded()   — bytes are already in addressable memory.  True for
    //                mmap'd / in-memory sources, false for sources that
    //                compute or fetch each pread (Mulberry32, HTTP).
    //                Callers that want to avoid I/O latency consult
    //                loaded() before scheduling expensive analyses.
    bool valid   () const;
    bool readable() const { return source != nullptr && valid(); }
    bool loaded  () const { return readable() && source->loaded(); }
};

// TensorRegistry — name → handle, plus enumeration order. Used by
// ModelView::tensors and by anything that wants to walk the full
// state_dict.
struct TensorRegistry {
    std::vector<TensorHandle> all;                       // enumeration order
    std::unordered_map<std::string, std::size_t> index;  // name → all[index]

    // Insert a handle. The name lookup is overwritten on duplicate name
    // (last wins) — the enumeration order is preserved.
    void insert(TensorHandle h);

    // O(1) lookup. Returns nullptr when name is not registered.
    const TensorHandle* find(std::string_view name) const;

    // Throwing variant — only when callers know the name must exist.
    const TensorHandle& at(std::string_view name) const;

    bool        empty() const { return all.empty(); }
    std::size_t size () const { return all.size(); }
};

}  // namespace llmengine
