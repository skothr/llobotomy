#include "llm_engine/tensor_handle.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace llmengine {

namespace {

// IEEE 754 binary16 → float32. Handles subnormals + inf/NaN.
float f16_to_f32(std::uint16_t h) {
    const std::uint32_t sign = (h & 0x8000u) << 16;
    const std::uint32_t exp  = (h & 0x7C00u) >> 10;
    const std::uint32_t mant = (h & 0x03FFu);
    std::uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;                                   // ±0
        } else {
            // subnormal: normalise
            std::uint32_t e = 1;
            std::uint32_t m = mant;
            while ((m & 0x0400u) == 0) { m <<= 1; ++e; }
            m &= 0x03FFu;
            bits = sign | ((127 - 15 - e + 1) << 23) | (m << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (mant << 13);          // inf / NaN
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

// bfloat16 → float32. bf16 shares the f32 exponent layout — just zero-
// extend into the high 16 bits of an f32.
float bf16_to_f32(std::uint16_t b) {
    std::uint32_t bits = static_cast<std::uint32_t>(b) << 16;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

// Dequantise n raw elements at byte offset `raw_offset` into `out`. Returns
// false on unsupported dtype (block-quantised types stay false until a
// dequantiser lands). The handle layer falls back to {} on false.
bool dequant_into(DType d, const TensorSource& src, std::size_t raw_offset,
                  std::size_t n_elements, float* out) {
    switch (d) {
        case DType::F32: {
            src.pread(raw_offset, n_elements * sizeof(float), out);
            return true;
        }
        case DType::F16: {
            std::vector<std::uint16_t> tmp(n_elements);
            src.pread(raw_offset, n_elements * sizeof(std::uint16_t), tmp.data());
            for (std::size_t i = 0; i < n_elements; ++i) out[i] = f16_to_f32(tmp[i]);
            return true;
        }
        case DType::BF16: {
            std::vector<std::uint16_t> tmp(n_elements);
            src.pread(raw_offset, n_elements * sizeof(std::uint16_t), tmp.data());
            for (std::size_t i = 0; i < n_elements; ++i) out[i] = bf16_to_f32(tmp[i]);
            return true;
        }
        case DType::I8: {
            std::vector<std::int8_t> tmp(n_elements);
            src.pread(raw_offset, n_elements, tmp.data());
            for (std::size_t i = 0; i < n_elements; ++i) out[i] = static_cast<float>(tmp[i]);
            return true;
        }
        case DType::U8: {
            std::vector<std::uint8_t> tmp(n_elements);
            src.pread(raw_offset, n_elements, tmp.data());
            for (std::size_t i = 0; i < n_elements; ++i) out[i] = static_cast<float>(tmp[i]);
            return true;
        }
        case DType::I16: {
            std::vector<std::int16_t> tmp(n_elements);
            src.pread(raw_offset, n_elements * sizeof(std::int16_t), tmp.data());
            for (std::size_t i = 0; i < n_elements; ++i) out[i] = static_cast<float>(tmp[i]);
            return true;
        }
        case DType::I32: {
            std::vector<std::int32_t> tmp(n_elements);
            src.pread(raw_offset, n_elements * sizeof(std::int32_t), tmp.data());
            for (std::size_t i = 0; i < n_elements; ++i) out[i] = static_cast<float>(tmp[i]);
            return true;
        }
        // F8 + block-quantised types: not yet implemented. A first-PR
        // GGUF / safetensors backend lands the F8 path; the Q* types
        // need block-aware dequantisers that mirror llama.cpp's tables.
        case DType::F8:
        case DType::Q4_K:
        case DType::Q4_0:
        case DType::Q8_0:
        case DType::Unknown:
            return false;
    }
    return false;
}

}  // namespace

std::size_t TensorHandle::element_count() const {
    if (shape.empty()) return 0;
    std::size_t n = 1;
    for (auto d : shape) {
        if (d <= 0) return 0;
        n *= static_cast<std::size_t>(d);
    }
    return n;
}

std::vector<float> TensorHandle::read_slice(std::size_t element_offset,
                                            std::size_t n) const {
    if (!source || n == 0) return {};
    const std::size_t total = element_count();
    if (element_offset >= total) return {};
    if (n > total - element_offset) n = total - element_offset;

    const std::size_t bpe = dtype_element_bytes(dtype);
    if (bpe == 0) return {};   // unsupported dtype this build

    const std::size_t raw_offset = byte_offset + element_offset * bpe;
    if (raw_offset + n * bpe > byte_offset + byte_length) return {};

    std::vector<float> out(n);
    if (!dequant_into(dtype, *source, raw_offset, n, out.data())) {
        return {};
    }
    return out;
}

std::vector<std::vector<float>>
TensorHandle::read_slice_2d(std::size_t row_offset, std::size_t row_count,
                            std::size_t col_offset, std::size_t col_count) const {
    if (!source || row_count == 0 || col_count == 0) return {};
    if (shape.size() < 2) return {};
    const auto rows_total = static_cast<std::size_t>(shape[0]);
    const auto cols_total = static_cast<std::size_t>(shape[1]);
    if (row_offset >= rows_total || col_offset >= cols_total) return {};
    if (row_count > rows_total - row_offset) row_count = rows_total - row_offset;
    if (col_count > cols_total - col_offset) col_count = cols_total - col_offset;

    std::vector<std::vector<float>> out;
    out.reserve(row_count);
    for (std::size_t r = 0; r < row_count; ++r) {
        // Row-major assumption: row stride = cols_total elements.
        const std::size_t off = (row_offset + r) * cols_total + col_offset;
        auto row = read_slice(off, col_count);
        if (row.empty()) return {};   // partial failure ⇒ empty 2-D
        out.emplace_back(std::move(row));
    }
    return out;
}

void TensorRegistry::insert(TensorHandle h) {
    auto it = index.find(h.name);
    if (it != index.end()) {
        all[it->second] = std::move(h);
        return;
    }
    const std::size_t pos = all.size();
    index.emplace(h.name, pos);
    all.emplace_back(std::move(h));
}

const TensorHandle* TensorRegistry::find(std::string_view name) const {
    // Two lookups (string_view → string for the map) acceptable here —
    // this is not on the per-frame hot path. Backends call it on UI
    // navigation events.
    auto it = index.find(std::string(name));
    if (it == index.end()) return nullptr;
    return &all[it->second];
}

const TensorHandle& TensorRegistry::at(std::string_view name) const {
    if (const TensorHandle* p = find(name)) return *p;
    throw std::out_of_range("TensorRegistry::at: " + std::string(name));
}

}  // namespace llmengine
