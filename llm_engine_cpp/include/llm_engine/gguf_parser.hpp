#pragma once
//
// gguf_parser — zero-dependency header-only-friendly GGUF header parser.
//
// Reads magic, version, KV metadata, and the tensor table from a .gguf
// file without mapping or reading any tensor bytes.  The returned
// GgufHeader contains everything needed to:
//
//   1. Populate a ModelView::topology from architecture-specific metadata.
//   2. Build a TensorRegistry (one TensorHandle per GgufTensorInfo, all
//      sharing one GgufSource over the data section).
//
// Supported architectures (metadata-key conventions):
//   - llama   (LLaMA 1/2/3, Mistral, Mixtral)
//   - qwen2   (Qwen2 / Qwen2.5)
//   - gemma   (Gemma 1/2)
//
// Adding a new architecture is one entry in the ArchProfile table in
// gguf_parser.cpp — no code-path changes needed.
//
// GGUF format spec summary (all fields little-endian):
//   magic        : u32 = 0x46554747 ("GGUF")
//   version      : u32 (3 is current; 1/2 also accepted)
//   n_tensors    : u64
//   n_kv         : u64
//   kv pairs     : key-string + type-tag + value, repeated n_kv times
//   tensor table : name + n_dims + dims[] + ggml_type + offset, repeated
//   <padding to alignment (default 32, overridden by general.alignment)>
//   data section : tensor bytes at data_section_offset, tensor-relative offsets

#include "llm_engine/tensor_source.hpp"  // DType

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace llmengine {

// Per-tensor entry from the GGUF tensor table.
struct GgufTensorInfo {
    std::string                   name;         // raw GGUF name, e.g. "blk.0.attn_q.weight"
    std::string                   canonical;    // engine canonical, e.g. "blocks.0.attn.W_Q.weight"
    DType                         dtype;
    std::vector<std::int64_t>     shape;        // C-order (innermost last), GGUF dims reversed
    std::size_t                   byte_offset;  // relative to GgufHeader::data_section_offset
    std::size_t                   byte_length;  // total bytes on disk (block-aware)
};

// Parsed GGUF header.  Metadata strings that are absent in the file are
// left at their zero-value ("" / 0 / kNoFloat / kNoInt) — callers treat
// sentinel values exactly as they do for any other backend.
struct GgufHeader {
    // ── provenance ───────────────────────────────────────────────────────
    std::string architecture;       // e.g. "llama", "qwen2", "gemma"
    std::uint32_t version = 0;

    // ── topology ────────────────────────────────────────────────────────
    int   n_layers  = -1;
    int   n_heads   = -1;
    int   n_kv_heads = -1;
    int   d_model   = -1;
    int   d_head    = -1;          // derived: d_model / n_heads
    int   d_mlp     = -1;
    int   vocab     = -1;
    int   max_pos   = -1;
    float rope_theta = std::numeric_limits<float>::quiet_NaN();

    // ── tokenizer surface ────────────────────────────────────────────────
    std::string chat_template;
    std::string bos_token;
    std::string eos_token;

    // ── data layout ──────────────────────────────────────────────────────
    std::size_t data_section_offset = 0;   // byte offset in file where tensors start
    std::size_t alignment = 32;            // as declared by general.alignment (default 32)

    // ── tensor table ────────────────────────────────────────────────────
    std::vector<GgufTensorInfo> tensors;

    // ── raw KV metadata (for future callers) ────────────────────────────
    // Each value is one of the primitive types the GGUF spec defines.
    // Array types are stored as std::vector<Value>.
    using Value = std::variant<
        std::monostate,         // unknown / unsupported type
        bool, std::uint8_t, std::int8_t,
        std::uint16_t, std::int16_t,
        std::uint32_t, std::int32_t,
        std::uint64_t, std::int64_t,
        float, double,
        std::string,
        std::vector<bool>, std::vector<std::uint8_t>, std::vector<std::int8_t>,
        std::vector<std::uint16_t>, std::vector<std::int16_t>,
        std::vector<std::uint32_t>, std::vector<std::int32_t>,
        std::vector<std::uint64_t>, std::vector<std::int64_t>,
        std::vector<float>, std::vector<double>,
        std::vector<std::string>
    >;
    std::unordered_map<std::string, Value> kv;

    // Helper: look up a key and return it as int if the stored type is any
    // integer variant.  Returns fallback when key is absent or non-integer.
    int kv_int(std::string_view key, int fallback = -1) const;
    float kv_float(std::string_view key, float fallback = std::numeric_limits<float>::quiet_NaN()) const;
    std::string kv_string(std::string_view key) const;
};

// Parse the header of a GGUF file.  Returns nullopt on any of:
//   - file not found or not readable
//   - bad magic / unsupported version
//   - truncated read
// On success the returned GgufHeader owns all parsed data; caller can
// construct a GgufSource from data_section_offset and build the registry.
std::optional<GgufHeader> parse_gguf(const std::string& path);

}  // namespace llmengine
