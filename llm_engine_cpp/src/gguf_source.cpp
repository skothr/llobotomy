// gguf_source.cpp — TensorSource implementation for .gguf file data sections.
//
// The implementation chooses the best I/O path available:
//   1. mmap (preferred, zero-copy, loaded() returns true) — avoids
//      repeated seek+read calls for the same or adjacent byte ranges.
//      Falls back silently when mmap fails (e.g. on Windows, or when the
//      file is on a network filesystem that doesn't support MAP_SHARED).
//   2. pread-based reads under a mutex — always available; slightly
//      higher latency than mmap but still fine for the UI's
//      on-demand-slice access pattern.
//
// Why mmap is preferred for this use case:
//   A GGUF file's data section is large (gigabytes) and the UI reads
//   small, non-contiguous slices (column of a weight matrix, a few
//   rows of an attention pattern).  mmap lets the OS page-fault in
//   only the touched pages, amortising the cost over many queries
//   without requiring the application to manage a buffer cache.

#include "llm_engine/gguf_source.hpp"

#include <cassert>
#include <cerrno>
#include <cstdio>
#include <cstring>

// POSIX headers — available on Linux and macOS (all current targets).
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace llmengine {

struct GgufSource::Impl {
    int    fd        = -1;
    void*  mmap_ptr  = MAP_FAILED;
    std::size_t mmap_len = 0;      // length passed to mmap (file_size, not data_size)
    bool   mmap_tried = false;     // set after first ensure_mmap() call

    ~Impl() {
        if (mmap_ptr != MAP_FAILED) ::munmap(mmap_ptr, mmap_len);
        if (fd >= 0)                ::close(fd);
    }
};

GgufSource::GgufSource(std::string path, std::size_t data_section_offset)
    : m_path(std::move(path)),
      m_data_offset(data_section_offset),
      m_data_size(0),
      m_impl(std::make_unique<Impl>()) {
    m_impl->fd = ::open(m_path.c_str(), O_RDONLY);
    if (m_impl->fd < 0) return;

    // Determine total file size to set data_size and mmap_len.
    struct stat st{};
    if (::fstat(m_impl->fd, &st) != 0) return;
    const auto fsize = static_cast<std::size_t>(st.st_size);
    if (fsize <= m_data_offset) return;  // degenerate
    m_data_size  = fsize - m_data_offset;
    m_impl->mmap_len = fsize;  // map the whole file — OS uses same physical pages
}

GgufSource::~GgufSource() = default;

void GgufSource::ensure_mmap() const {
    // Caller holds m_mu.
    if (m_impl->mmap_tried) return;
    m_impl->mmap_tried = true;
    if (m_impl->fd < 0 || m_impl->mmap_len == 0) return;

    void* p = ::mmap(nullptr, m_impl->mmap_len, PROT_READ, MAP_SHARED,
                     m_impl->fd, /*offset=*/0);
    if (p == MAP_FAILED) return;  // silently fall back to pread
    m_impl->mmap_ptr = p;

    // Advise the OS that we'll access randomly — prevents read-ahead on
    // large sequential fetches that we won't use.
    ::madvise(p, m_impl->mmap_len, MADV_RANDOM);
}

bool GgufSource::loaded() const {
    std::lock_guard<std::mutex> lk(m_mu);
    ensure_mmap();
    return m_impl->mmap_ptr != MAP_FAILED;
}

std::span<const std::byte> GgufSource::try_mmap() const {
    std::lock_guard<std::mutex> lk(m_mu);
    ensure_mmap();
    if (m_impl->mmap_ptr == MAP_FAILED) return {};
    // Return only the data section slice (not the header bytes).
    const auto* base = static_cast<const std::byte*>(m_impl->mmap_ptr);
    return { base + m_data_offset, m_data_size };
}

void GgufSource::pread(std::size_t offset, std::size_t n_bytes, void* out) const {
    if (n_bytes == 0 || out == nullptr) return;
    if (offset + n_bytes > m_data_size) return;  // out-of-range — leave out unchanged

    // Fast-path: if mmap is active, memcpy from the mapped region.
    // We still take the mutex for the lazy ensure_mmap() check; after
    // that the pointer stays valid for the lifetime of the source.
    {
        std::lock_guard<std::mutex> lk(m_mu);
        ensure_mmap();
        if (m_impl->mmap_ptr != MAP_FAILED) {
            const auto* base = static_cast<const std::byte*>(m_impl->mmap_ptr);
            std::memcpy(out, base + m_data_offset + offset, n_bytes);
            return;
        }
    }

    // Fallback: pread into out under the mutex (pread is atomic per-call on
    // Linux/macOS, but we use the mutex to avoid interlaced seeks if a future
    // refactor introduces dup-fd-less sharing across threads).
    if (m_impl->fd < 0) return;

    std::lock_guard<std::mutex> lk(m_mu);
    const off_t file_offset = static_cast<off_t>(m_data_offset + offset);
    auto* dst = static_cast<char*>(out);
    std::size_t remaining = n_bytes;
    while (remaining > 0) {
        const ::ssize_t got = ::pread(m_impl->fd, dst, remaining, file_offset +
                                      static_cast<off_t>(n_bytes - remaining));
        if (got <= 0) return;  // EOF or error — leave rest of out unchanged
        dst       += got;
        remaining -= static_cast<std::size_t>(got);
    }
}

}  // namespace llmengine
