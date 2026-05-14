#pragma once
//
// GgufSource — TensorSource backed by a .gguf file's data section.
//
// All tensors in a single .gguf file share one GgufSource instance
// (the handles differ only by byte_offset + byte_length).  This is
// the same "one source, many handles" pattern used by MockModel's
// Mulberry32Source: the TensorRegistry owns the handles; the source
// owns the file descriptor / mmap region.
//
// Construction:
//   1. Call parse_gguf() to get a GgufHeader containing
//      data_section_offset and the tensor table.
//   2. Construct one GgufSource(path, data_section_offset).
//   3. Build one TensorHandle per GgufTensorInfo, all sharing the
//      same std::shared_ptr<GgufSource>.
//
// Threading:
//   pread() is stateless — it seeks to (data_section_offset + offset)
//   on each call.  Multiple handles reading concurrently require
//   either: (a) their own file descriptors (safest), or (b) a mutex
//   over the seek+read pair.  This implementation uses a mutex so the
//   source can be shared without the caller managing fd duplication.
//   Reads are rare (user-triggered from the UI — not on the hot path)
//   so the mutex cost is negligible.
//
// mmap:
//   When try_mmap() is called for the first time (and mmap is
//   available), the source maps the file and returns a span covering
//   the full data section.  Subsequent calls return the cached span.
//   Handles that have the mmap fast-path skip the mutex entirely.

#include "llm_engine/tensor_source.hpp"

#include <cstddef>
#include <memory>
#include <mutex>
#include <span>
#include <string>

namespace llmengine {

class GgufSource final : public TensorSource {
public:
    // path               — path to the .gguf file (kept for error messages)
    // data_section_offset — byte offset in file where tensor bytes begin;
    //                       returned by parse_gguf() as GgufHeader::data_section_offset
    GgufSource(std::string path, std::size_t data_section_offset);
    ~GgufSource() override;

    GgufSource(const GgufSource&)            = delete;
    GgufSource& operator=(const GgufSource&) = delete;

    // Read n_bytes from (data_section_offset + offset) into out.
    // Silently no-ops on out-of-range or I/O error — handle layer
    // surfaces the sentinel.
    void pread(std::size_t offset, std::size_t n_bytes, void* out) const override;

    // Returns a span covering the full data section if mmap succeeded,
    // empty span otherwise.  Callers that get a non-empty span can
    // memcpy directly and skip the mutex.
    std::span<const std::byte> try_mmap() const override;

    // Total size of the data section.  Set to (file_size - data_section_offset)
    // on construction; 0 if the file could not be stat'd.
    std::size_t size_bytes() const override { return m_data_size; }

    // True when mmap is active — bytes are RAM-addressable.
    bool loaded() const override;

private:
    // Attempt to mmap lazily.  Called under m_mu.
    void ensure_mmap() const;

    struct Impl;
    std::string              m_path;
    std::size_t              m_data_offset;  // data_section_offset
    std::size_t              m_data_size;    // file_size - data_section_offset
    mutable std::mutex       m_mu;
    mutable std::unique_ptr<Impl> m_impl;
};

}  // namespace llmengine
