#pragma once
//
// TensorSource — abstract byte-supplier behind a TensorHandle.
//
// One TensorSource per underlying storage; a TensorHandle holds a
// shared_ptr to one of these plus the offset/length describing its
// slice. Implementations live across backends:
//
//   InMemoryTensorSource   wraps an already-allocated buffer (used by
//                          MockModel and CaptureBundle).
//   GgufSource             pread/mmap a .gguf file. [future]
//   SafetensorsSource      pread/mmap a .safetensors file. [future]
//   HfProxySource          HTTP Range GET against the FastAPI weight
//                          endpoint. [future]
//
// Reads are byte-level so the same source can back tensors of any
// dtype. The handle layer is responsible for shape / dtype semantics.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace llmengine {

// DTypes the engine understands. Block-quantised types (Q4_K, Q4_0,
// Q8_0) are listed so TensorRegistry can enumerate them from a GGUF
// header without forcing dequantisation to be implemented yet — a
// handle of an unsupported dtype simply returns an empty vector from
// read_slice() and a warn line in the engine log.
enum class DType : std::uint8_t {
    F32, F16, BF16, F8,
    I32, I16, I8, U8,
    Q4_K, Q4_0, Q8_0,
    Unknown,
};

// Bytes per *logical* element for non-block dtypes. For block-quantised
// dtypes this returns 0 — callers must take the block-aware path.
std::size_t dtype_element_bytes(DType d);

const char* dtype_name(DType d);

// Source interface. Implementations choose the cheapest path (mmap +
// memcpy, pread, HTTP range, etc.). pread() is named after the POSIX
// call but does not require a real file descriptor.
//
// loaded() distinguishes "bytes are sitting in addressable memory right
// now" (mmap, in-memory) from "bytes can be produced on demand but each
// read pays compute or I/O" (Mulberry32Source, HfProxySource).  Lets
// callers cheaply skip work that's only worth it when reads are free.
class TensorSource {
public:
    virtual ~TensorSource() = default;

    // Read n_bytes starting at offset into out. Must not throw on
    // out-of-range — return without touching out and let the handle
    // layer surface the sentinel.
    virtual void pread(std::size_t offset, std::size_t n_bytes, void* out) const = 0;

    // Optional mmap fast-path. Returns an empty span when not mapped.
    virtual std::span<const std::byte> try_mmap() const { return {}; }

    // Total size of the source's data region, in bytes. Used for
    // bounds checks at handle-construction time.
    virtual std::size_t size_bytes() const = 0;

    // True when bytes are already in addressable RAM (mmap'd file, in-
    // memory buffer).  False when each pread() computes / fetches.
    // Default false — implementations that are truly resident override.
    virtual bool loaded() const { return false; }
};

// In-memory source — owns a std::vector<std::byte>. Used by CaptureBundle
// (per-forward-pass activations / attention matrices), and by any backend
// that has already loaded a tensor into RAM.
class InMemoryTensorSource final : public TensorSource {
public:
    explicit InMemoryTensorSource(std::vector<std::byte> bytes);

    // Convenience: take ownership of a float vector. Cheap; no copy.
    static std::shared_ptr<InMemoryTensorSource> from_floats(std::vector<float> v);

    void pread(std::size_t offset, std::size_t n_bytes, void* out) const override;
    std::span<const std::byte> try_mmap() const override;
    std::size_t size_bytes() const override { return m_bytes.size(); }
    bool        loaded   () const override { return true; }

private:
    std::vector<std::byte> m_bytes;
};

// Mulberry32-backed source — generates deterministic f16/f32 bytes on each
// pread().  Used by MockModel so its TensorHandles are architecturally
// identical to real-backend handles (valid + readable + lazy bytes) rather
// than empty pseudo-handles.  Same byte stream every call; no allocation.
//
// readable=true (pread always succeeds within size_bytes), loaded=false
// (each call re-runs the PRNG).
class Mulberry32Source final : public TensorSource {
public:
    Mulberry32Source(std::uint32_t seed, std::size_t size_bytes);

    void pread(std::size_t offset, std::size_t n_bytes, void* out) const override;
    std::size_t size_bytes() const override { return m_size; }

private:
    std::uint32_t m_seed;
    std::size_t   m_size;
};

}  // namespace llmengine
