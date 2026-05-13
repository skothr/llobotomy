#include "llm_engine/tensor_source.hpp"

#include <cstring>
#include <utility>

namespace llmengine {

std::size_t dtype_element_bytes(DType d) {
    switch (d) {
        case DType::F32:  return 4;
        case DType::F16:  return 2;
        case DType::BF16: return 2;
        case DType::F8:   return 1;
        case DType::I32:  return 4;
        case DType::I16:  return 2;
        case DType::I8:   return 1;
        case DType::U8:   return 1;
        // Block-quantised: 0 signals "not addressable per element". The
        // dequantiser owns the block-aware path.
        case DType::Q4_K:
        case DType::Q4_0:
        case DType::Q8_0:
        case DType::Unknown:
            return 0;
    }
    return 0;
}

const char* dtype_name(DType d) {
    switch (d) {
        case DType::F32:     return "f32";
        case DType::F16:     return "f16";
        case DType::BF16:    return "bf16";
        case DType::F8:      return "f8";
        case DType::I32:     return "i32";
        case DType::I16:     return "i16";
        case DType::I8:      return "i8";
        case DType::U8:      return "u8";
        case DType::Q4_K:    return "q4_k";
        case DType::Q4_0:    return "q4_0";
        case DType::Q8_0:    return "q8_0";
        case DType::Unknown: return "unknown";
    }
    return "unknown";
}

InMemoryTensorSource::InMemoryTensorSource(std::vector<std::byte> bytes)
    : m_bytes(std::move(bytes)) {}

std::shared_ptr<InMemoryTensorSource>
InMemoryTensorSource::from_floats(std::vector<float> v) {
    // Reinterpret the float storage as bytes by copying — std::vector
    // has strict alias rules and no portable in-place reinterpret.
    std::vector<std::byte> bytes(v.size() * sizeof(float));
    std::memcpy(bytes.data(), v.data(), bytes.size());
    return std::make_shared<InMemoryTensorSource>(std::move(bytes));
}

void InMemoryTensorSource::pread(std::size_t offset, std::size_t n_bytes, void* out) const {
    if (offset >= m_bytes.size()) return;
    const std::size_t avail = m_bytes.size() - offset;
    const std::size_t to_copy = n_bytes < avail ? n_bytes : avail;
    std::memcpy(out, m_bytes.data() + offset, to_copy);
}

std::span<const std::byte> InMemoryTensorSource::try_mmap() const {
    return {m_bytes.data(), m_bytes.size()};
}

}  // namespace llmengine
