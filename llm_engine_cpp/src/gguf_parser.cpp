// gguf_parser.cpp — self-contained GGUF header parser.
//
// No external dependencies.  Reads the GGUF binary format sequentially
// using POSIX FILE* so the implementation is the same on every platform
// the engine targets.
//
// Design choices:
//   - We parse every KV pair and stash raw values in GgufHeader::kv so
//     future callers can inspect arbitrary metadata without touching this
//     file.  Architecture-specific extraction (topology, tokenizer) is
//     done as a second pass over kv once the full table is built.
//
//   - Tensor names are run through a per-architecture normalisation table
//     that maps raw GGUF names (e.g. "blk.7.attn_q.weight") to engine
//     canonical names (e.g. "blocks.7.attn.W_Q.weight").  Unknown names
//     fall back to the raw name — the handle is still valid and enumerable
//     in the raw-tensors workspace, the UI just shows the original name.
//
//   - Block sizes for quantised dtypes follow the same table as the Python
//     gguf_reader.py (GGML_BLOCK_SIZE).  We compute byte_length from
//     element count and block geometry so TensorHandle::byte_length is
//     always meaningful (a prerequisite for valid() to return true).
//
//   - GGUF dims are stored innermost-first (ne[0] is the fast axis).
//     Engine convention follows PyTorch/numpy (innermost last, i.e. shape
//     is in "C order").  We reverse the dims vector on the way out.

#include "llm_engine/gguf_parser.hpp"
#include "llm_engine/model.hpp"  // kNoInt, kNoFloat

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace llmengine {

// ── GGUF value type tags (spec §2.2) ─────────────────────────────────────
enum class GgufVType : std::uint32_t {
    UINT8   =  0, INT8   =  1, UINT16 =  2, INT16  =  3,
    UINT32  =  4, INT32  =  5, FLOAT32=  6, BOOL   =  7,
    STRING  =  8, ARRAY  =  9, UINT64 = 10, INT64  = 11,
    FLOAT64 = 12,
};

// ── GGML dtype numeric codes ──────────────────────────────────────────────
// Mirrors GGML_TYPE_* constants in gguf_reader.py.
enum class GgmlType : std::uint32_t {
    F32  =  0, F16  =  1, Q4_0 =  2, Q4_1 =  3,
    Q5_0 =  6, Q5_1 =  7, Q8_0 =  8,
    Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14,
    BF16 = 26,
};

// Map from GGML numeric code → (DType, values_per_block, bytes_per_block).
// values_per_block == 1 means dtype is not block-quantised (BPE == bytes_per_block).
struct BlockInfo { DType dtype; std::size_t vpb; std::size_t bpb; };

static const std::unordered_map<std::uint32_t, BlockInfo> kBlockInfo = {
    {  0, { DType::F32,   1,   4 }},
    {  1, { DType::F16,   1,   2 }},
    {  2, { DType::Q4_0, 32,  18 }},
    {  3, { DType::Unknown,32, 20 }},  // Q4_1 — Unknown until we add DType::Q4_1
    {  6, { DType::Unknown,32, 22 }},  // Q5_0
    {  7, { DType::Unknown,32, 24 }},  // Q5_1
    {  8, { DType::Q8_0, 32,  34 }},
    { 10, { DType::Unknown,256, 84 }}, // Q2_K
    { 11, { DType::Unknown,256,110 }}, // Q3_K
    { 12, { DType::Q4_K, 256,144 }},
    { 13, { DType::Unknown,256,176 }}, // Q5_K
    { 14, { DType::Unknown,256,210 }}, // Q6_K
    { 26, { DType::BF16,  1,   2 }},
};

// ── Architecture tensor-name normalisation ────────────────────────────────
//
// Each architecture has a table of (raw_prefix, canonical_prefix) pairs for
// per-layer tensors (where "blk.N." → "blocks.N.") and a separate table for
// global tensors.  The tables are processed in order; first match wins.
//
// Adding Mistral/Phi/Falcon/etc. later is one new ArchProfile entry plus its
// two tables.  No if/else chains needed.

struct NameRule {
    const char* raw_suffix;        // after "blk.N." for layer rules, bare for global
    const char* canonical_suffix;  // after "blocks.N." for layer rules, bare for global
};

struct ArchProfile {
    const char* arch;
    // Global tensor name rules (full GGUF name → full canonical name)
    std::vector<std::pair<std::string,std::string>> global;
    // Per-layer suffix rules ("blk.N.<raw>" → "blocks.N.<canon>")
    std::vector<std::pair<std::string,std::string>> layer;
};

// Returns all registered architecture profiles.
static const std::vector<ArchProfile>& arch_profiles() {
    static const std::vector<ArchProfile> kProfiles = {
        {
            "llama",
            // global
            {
                { "token_embd.weight",    "tok_embeddings.weight" },
                { "output_norm.weight",   "ln_f.weight"           },
                { "output.weight",        "lm_head.weight"        },
            },
            // per-layer
            {
                { "attn_q.weight",     "attn.W_Q.weight"   },
                { "attn_k.weight",     "attn.W_K.weight"   },
                { "attn_v.weight",     "attn.W_V.weight"   },
                { "attn_output.weight","attn.W_O.weight"   },
                { "ffn_gate.weight",   "mlp.W_gate.weight" },
                { "ffn_up.weight",     "mlp.W_up.weight"   },
                { "ffn_down.weight",   "mlp.W_down.weight" },
                { "attn_norm.weight",  "ln1.weight"        },
                { "ffn_norm.weight",   "ln2.weight"        },
            },
        },
        {
            "qwen2",
            // global — Qwen2 uses the same global names as llama
            {
                { "token_embd.weight",    "tok_embeddings.weight" },
                { "output_norm.weight",   "ln_f.weight"           },
                { "output.weight",        "lm_head.weight"        },
            },
            // per-layer — Qwen2 is llama-compatible at this level
            {
                { "attn_q.weight",     "attn.W_Q.weight"   },
                { "attn_k.weight",     "attn.W_K.weight"   },
                { "attn_v.weight",     "attn.W_V.weight"   },
                { "attn_output.weight","attn.W_O.weight"   },
                { "ffn_gate.weight",   "mlp.W_gate.weight" },
                { "ffn_up.weight",     "mlp.W_up.weight"   },
                { "ffn_down.weight",   "mlp.W_down.weight" },
                { "attn_norm.weight",  "ln1.weight"        },
                { "ffn_norm.weight",   "ln2.weight"        },
            },
        },
        {
            "gemma",
            // global
            {
                { "token_embd.weight",    "tok_embeddings.weight" },
                { "output_norm.weight",   "ln_f.weight"           },
                // Gemma shares the embedding weight with lm_head (tied)
                { "output.weight",        "lm_head.weight"        },
            },
            // per-layer — Gemma uses slightly different norm names
            {
                { "attn_q.weight",     "attn.W_Q.weight"     },
                { "attn_k.weight",     "attn.W_K.weight"     },
                { "attn_v.weight",     "attn.W_V.weight"     },
                { "attn_output.weight","attn.W_O.weight"     },
                { "ffn_gate.weight",   "mlp.W_gate.weight"   },
                { "ffn_up.weight",     "mlp.W_up.weight"     },
                { "ffn_down.weight",   "mlp.W_down.weight"   },
                { "attn_norm.weight",  "ln1.weight"          },
                { "ffn_norm.weight",   "ln2.weight"          },
                { "post_ffw_norm.weight", "ln_post_mlp.weight" },
                { "post_attention_norm.weight", "ln_post_attn.weight" },
            },
        },
    };
    return kProfiles;
}

// Normalise a raw GGUF tensor name given the file's architecture string.
// Returns the canonical engine name, or raw_name if no rule matches.
static std::string normalise_name(const std::string& raw_name,
                                  const std::string& arch) {
    // Find profile
    const ArchProfile* prof = nullptr;
    for (const auto& p : arch_profiles()) {
        if (p.arch == arch) { prof = &p; break; }
    }
    if (!prof) return raw_name;  // unknown arch — pass through

    // Try global rules first
    for (const auto& [raw, canon] : prof->global) {
        if (raw_name == raw) return canon;
    }

    // Try per-layer: pattern "blk.<N>.<suffix>"
    if (raw_name.size() > 4 && raw_name.substr(0, 4) == "blk.") {
        const auto dot = raw_name.find('.', 4);
        if (dot != std::string::npos) {
            const std::string layer_str = raw_name.substr(4, dot - 4);
            const std::string suffix    = raw_name.substr(dot + 1);
            for (const auto& [raw_sfx, canon_sfx] : prof->layer) {
                if (suffix == raw_sfx) {
                    return "blocks." + layer_str + "." + canon_sfx;
                }
            }
            // Layer tensor with unknown suffix — keep layer prefix, pass suffix
            return "blocks." + layer_str + "." + suffix;
        }
    }
    return raw_name;  // fallback
}

// ── Low-level binary reader ───────────────────────────────────────────────

class Reader {
public:
    explicit Reader(const std::string& path)
        : m_file(std::fopen(path.c_str(), "rb")) {
        if (!m_file) {
            m_error = "cannot open file: " + path;
        }
    }
    ~Reader() { if (m_file) std::fclose(m_file); }

    Reader(const Reader&)            = delete;
    Reader& operator=(const Reader&) = delete;

    bool ok() const { return m_file && m_error.empty(); }
    const std::string& error() const { return m_error; }

    // Current byte offset in the file.
    std::size_t tell() const {
        if (!m_file) return 0;
        const long pos = std::ftell(m_file);
        return pos >= 0 ? static_cast<std::size_t>(pos) : 0;
    }

    // Read exactly n bytes into dst.  Sets error on short read.
    bool read(void* dst, std::size_t n) {
        if (!ok()) return false;
        if (n == 0) return true;
        const std::size_t got = std::fread(dst, 1, n, m_file);
        if (got != n) {
            m_error = "short read: wanted " + std::to_string(n) +
                      " got " + std::to_string(got);
            return false;
        }
        return true;
    }

    template<typename T>
    std::optional<T> read_le() {
        T v{};
        if (!read(&v, sizeof(T))) return std::nullopt;
        // GGUF is little-endian.  On LE hosts this is a no-op; on BE we
        // would need to byte-swap.  We assume LE (all current engine targets).
        return v;
    }

    std::optional<std::string> read_gguf_string() {
        auto len = read_le<std::uint64_t>();
        if (!len) return std::nullopt;
        if (*len > (1u << 28u)) {
            m_error = "GGUF string length implausible: " + std::to_string(*len);
            return std::nullopt;
        }
        std::string s(*len, '\0');
        if (!read(s.data(), *len)) return std::nullopt;
        return s;
    }

private:
    FILE*       m_file = nullptr;
    std::string m_error;
};

// ── KV value parsing ──────────────────────────────────────────────────────

// Forward declaration for recursive array parsing.
static std::optional<GgufHeader::Value>
read_gguf_value(Reader& r, std::uint32_t vtype);

static std::optional<GgufHeader::Value>
read_gguf_array(Reader& r) {
    auto arr_type = r.read_le<std::uint32_t>();
    auto arr_len  = r.read_le<std::uint64_t>();
    if (!arr_type || !arr_len) return std::nullopt;

    // We only materialise arrays of types that have dedicated vector arms in
    // GgufHeader::Value.  Unknown element types consume bytes so the cursor
    // stays correct, then return monostate.
    const std::uint32_t et = *arr_type;
    const std::uint64_t n  = *arr_len;

    // Helper: read n scalars of known type into a vector variant.
#define READ_VEC(elem_t, vtype_tag, vec_variant_t) \
    if (et == static_cast<std::uint32_t>(vtype_tag)) { \
        std::vector<elem_t> vec; \
        vec.reserve(static_cast<std::size_t>(n)); \
        for (std::uint64_t i = 0; i < n; ++i) { \
            auto v = r.read_le<elem_t>(); \
            if (!v) return std::nullopt; \
            vec.push_back(*v); \
        } \
        return GgufHeader::Value{vec_variant_t(std::move(vec))}; \
    }

    READ_VEC(std::uint8_t,  GgufVType::UINT8,   std::vector<std::uint8_t>)
    READ_VEC(std::int8_t,   GgufVType::INT8,    std::vector<std::int8_t>)
    READ_VEC(std::uint16_t, GgufVType::UINT16,  std::vector<std::uint16_t>)
    READ_VEC(std::int16_t,  GgufVType::INT16,   std::vector<std::int16_t>)
    READ_VEC(std::uint32_t, GgufVType::UINT32,  std::vector<std::uint32_t>)
    READ_VEC(std::int32_t,  GgufVType::INT32,   std::vector<std::int32_t>)
    READ_VEC(std::uint64_t, GgufVType::UINT64,  std::vector<std::uint64_t>)
    READ_VEC(std::int64_t,  GgufVType::INT64,   std::vector<std::int64_t>)
    READ_VEC(float,         GgufVType::FLOAT32, std::vector<float>)
    READ_VEC(double,        GgufVType::FLOAT64, std::vector<double>)
#undef READ_VEC

    // BOOL arrays: stored as u8 in GGUF.
    if (et == static_cast<std::uint32_t>(GgufVType::BOOL)) {
        std::vector<bool> vec;
        vec.reserve(static_cast<std::size_t>(n));
        for (std::uint64_t i = 0; i < n; ++i) {
            auto v = r.read_le<std::uint8_t>();
            if (!v) return std::nullopt;
            vec.push_back(*v != 0);
        }
        return GgufHeader::Value{std::move(vec)};
    }

    // String arrays
    if (et == static_cast<std::uint32_t>(GgufVType::STRING)) {
        std::vector<std::string> vec;
        vec.reserve(static_cast<std::size_t>(n));
        for (std::uint64_t i = 0; i < n; ++i) {
            auto s = r.read_gguf_string();
            if (!s) return std::nullopt;
            vec.push_back(std::move(*s));
        }
        return GgufHeader::Value{std::move(vec)};
    }

    // Nested arrays or unknown element type — skip by reading each element
    // individually (recursively).  We discard the result but keep the cursor
    // moving so subsequent parsing stays correct.
    for (std::uint64_t i = 0; i < n; ++i) {
        if (!read_gguf_value(r, et)) return std::nullopt;
    }
    return GgufHeader::Value{std::monostate{}};
}

static std::optional<GgufHeader::Value>
read_gguf_value(Reader& r, std::uint32_t vtype) {
    switch (static_cast<GgufVType>(vtype)) {
    case GgufVType::UINT8:   if (auto v = r.read_le<std::uint8_t> ()) return GgufHeader::Value{*v}; break;
    case GgufVType::INT8:    if (auto v = r.read_le<std::int8_t>  ()) return GgufHeader::Value{*v}; break;
    case GgufVType::UINT16:  if (auto v = r.read_le<std::uint16_t>()) return GgufHeader::Value{*v}; break;
    case GgufVType::INT16:   if (auto v = r.read_le<std::int16_t> ()) return GgufHeader::Value{*v}; break;
    case GgufVType::UINT32:  if (auto v = r.read_le<std::uint32_t>()) return GgufHeader::Value{*v}; break;
    case GgufVType::INT32:   if (auto v = r.read_le<std::int32_t> ()) return GgufHeader::Value{*v}; break;
    case GgufVType::FLOAT32: if (auto v = r.read_le<float>        ()) return GgufHeader::Value{*v}; break;
    case GgufVType::BOOL:    if (auto v = r.read_le<std::uint8_t> ()) return GgufHeader::Value{static_cast<bool>(*v != 0)}; break;
    case GgufVType::STRING:  if (auto v = r.read_gguf_string()      ) return GgufHeader::Value{*v}; break;
    case GgufVType::UINT64:  if (auto v = r.read_le<std::uint64_t>()) return GgufHeader::Value{*v}; break;
    case GgufVType::INT64:   if (auto v = r.read_le<std::int64_t> ()) return GgufHeader::Value{*v}; break;
    case GgufVType::FLOAT64: if (auto v = r.read_le<double>        ()) return GgufHeader::Value{*v}; break;
    case GgufVType::ARRAY:   return read_gguf_array(r);
    }
    return std::nullopt;
}

// ── GgufHeader KV helpers ─────────────────────────────────────────────────

int GgufHeader::kv_int(std::string_view key, int fallback) const {
    const auto it = kv.find(std::string(key));
    if (it == kv.end()) return fallback;
    return std::visit([fallback](const auto& v) -> int {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_integral_v<T> && !std::is_same_v<T,bool>)
            return static_cast<int>(v);
        return fallback;
    }, it->second);
}

float GgufHeader::kv_float(std::string_view key, float fallback) const {
    const auto it = kv.find(std::string(key));
    if (it == kv.end()) return fallback;
    return std::visit([fallback](const auto& v) -> float {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_floating_point_v<T>)
            return static_cast<float>(v);
        if constexpr (std::is_integral_v<T> && !std::is_same_v<T,bool>)
            return static_cast<float>(v);
        return fallback;
    }, it->second);
}

std::string GgufHeader::kv_string(std::string_view key) const {
    const auto it = kv.find(std::string(key));
    if (it == kv.end()) return {};
    if (const auto* s = std::get_if<std::string>(&it->second)) return *s;
    return {};
}

// ── byte_length calculation ───────────────────────────────────────────────
// Given element count and GGML type code, compute byte length on disk.
// Returns 0 for unknown types (caller sets DType::Unknown; handle returns {}).

static std::size_t compute_byte_length(std::uint32_t ggml_type,
                                       std::size_t n_elements) {
    const auto it = kBlockInfo.find(ggml_type);
    if (it == kBlockInfo.end()) return 0;
    const auto& bi = it->second;
    if (bi.vpb == 1) {
        // Non-block: bytes = n * bpb
        return n_elements * bi.bpb;
    }
    // Block-quantised: full blocks only (spec guarantees divisibility)
    const std::size_t n_blocks = (n_elements + bi.vpb - 1) / bi.vpb;
    return n_blocks * bi.bpb;
}

// ── parse_gguf ────────────────────────────────────────────────────────────

std::optional<GgufHeader> parse_gguf(const std::string& path) {
    Reader r(path);
    if (!r.ok()) return std::nullopt;

    GgufHeader hdr;

    // ── magic + version ──────────────────────────────────────────────────
    {
        char magic[4] = {};
        if (!r.read(magic, 4)) return std::nullopt;
        if (magic[0] != 'G' || magic[1] != 'G' ||
            magic[2] != 'U' || magic[3] != 'F') return std::nullopt;
    }
    {
        auto ver = r.read_le<std::uint32_t>();
        if (!ver) return std::nullopt;
        if (*ver < 1 || *ver > 3) return std::nullopt;  // reject future versions
        hdr.version = *ver;
    }

    // ── counts ──────────────────────────────────────────────────────────
    auto n_tensors = r.read_le<std::uint64_t>();
    auto n_kv      = r.read_le<std::uint64_t>();
    if (!n_tensors || !n_kv) return std::nullopt;
    if (*n_tensors > (1u << 20u)) return std::nullopt;  // sanity cap: 1M tensors
    if (*n_kv      > (1u << 20u)) return std::nullopt;

    // ── KV pairs ─────────────────────────────────────────────────────────
    hdr.kv.reserve(static_cast<std::size_t>(*n_kv));
    for (std::uint64_t i = 0; i < *n_kv; ++i) {
        auto key = r.read_gguf_string();
        if (!key) return std::nullopt;
        auto vtype = r.read_le<std::uint32_t>();
        if (!vtype) return std::nullopt;
        auto val = read_gguf_value(r, *vtype);
        if (!val) return std::nullopt;
        hdr.kv[std::move(*key)] = std::move(*val);
    }
    if (!r.ok()) return std::nullopt;

    // ── architecture-specific metadata extraction ─────────────────────────
    hdr.architecture = hdr.kv_string("general.architecture");

    // general.alignment overrides the default 32.
    {
        const int align_i = hdr.kv_int("general.alignment");
        if (align_i > 0) hdr.alignment = static_cast<std::size_t>(align_i);
    }

    // Topology — keys are arch-prefixed: "{arch}.block_count", etc.
    const std::string& a = hdr.architecture;
    hdr.n_layers   = hdr.kv_int(a + ".block_count");
    hdr.n_heads    = hdr.kv_int(a + ".attention.head_count");
    hdr.n_kv_heads = hdr.kv_int(a + ".attention.head_count_kv", hdr.n_heads);
    hdr.d_model    = hdr.kv_int(a + ".embedding_length");
    hdr.d_mlp      = hdr.kv_int(a + ".feed_forward_length");
    hdr.max_pos    = hdr.kv_int(a + ".context_length");
    hdr.rope_theta = hdr.kv_float(a + ".rope.freq_base");
    if (hdr.n_heads > 0 && hdr.d_model > 0)
        hdr.d_head = hdr.d_model / hdr.n_heads;

    // Vocab — prefer tokenizer.ggml.token_count, fall back to
    // {arch}.vocab_size (some writers emit only one or the other).
    hdr.vocab = hdr.kv_int("tokenizer.ggml.token_count");
    if (hdr.vocab < 0) hdr.vocab = hdr.kv_int(a + ".vocab_size");

    // Tokenizer surface.  Tokens are stored as a string array under
    // "tokenizer.ggml.tokens"; BOS/EOS are IDs, not strings, in GGUF.
    // We derive the string by index-lookup when possible.
    {
        const auto it = hdr.kv.find("tokenizer.ggml.tokens");
        if (it != hdr.kv.end()) {
            if (const auto* toks = std::get_if<std::vector<std::string>>(&it->second)) {
                const int bos_id = hdr.kv_int("tokenizer.ggml.bos_token_id");
                const int eos_id = hdr.kv_int("tokenizer.ggml.eos_token_id");
                const auto sz = static_cast<int>(toks->size());
                if (bos_id >= 0 && bos_id < sz) hdr.bos_token = (*toks)[bos_id];
                if (eos_id >= 0 && eos_id < sz) hdr.eos_token = (*toks)[eos_id];
            }
        }
    }
    hdr.chat_template = hdr.kv_string("tokenizer.chat_template");

    // ── tensor table ──────────────────────────────────────────────────────
    struct RawTensor {
        std::string name;
        std::uint32_t ggml_type;
        std::vector<std::int64_t> shape_innermost_first;
        std::size_t offset;  // relative to data section
    };
    std::vector<RawTensor> raw_tensors;
    raw_tensors.reserve(static_cast<std::size_t>(*n_tensors));

    for (std::uint64_t i = 0; i < *n_tensors; ++i) {
        auto name = r.read_gguf_string();
        if (!name) return std::nullopt;
        auto n_dims = r.read_le<std::uint32_t>();
        if (!n_dims) return std::nullopt;
        if (*n_dims > 8) return std::nullopt;  // sanity
        std::vector<std::int64_t> dims;
        dims.reserve(*n_dims);
        for (std::uint32_t d = 0; d < *n_dims; ++d) {
            auto dim = r.read_le<std::uint64_t>();
            if (!dim) return std::nullopt;
            dims.push_back(static_cast<std::int64_t>(*dim));
        }
        auto ggml_type = r.read_le<std::uint32_t>();
        if (!ggml_type) return std::nullopt;
        auto offset = r.read_le<std::uint64_t>();
        if (!offset) return std::nullopt;

        raw_tensors.push_back({
            std::move(*name),
            *ggml_type,
            std::move(dims),
            static_cast<std::size_t>(*offset),
        });
    }
    if (!r.ok()) return std::nullopt;

    // ── data section offset ───────────────────────────────────────────────
    // Header ends here; data section starts at the next alignment boundary.
    {
        const std::size_t header_end = r.tell();
        const std::size_t al = hdr.alignment;
        hdr.data_section_offset = ((header_end + al - 1) / al) * al;
    }

    // ── build GgufTensorInfo list ─────────────────────────────────────────
    hdr.tensors.reserve(raw_tensors.size());
    for (auto& rt : raw_tensors) {
        // GGUF dims are innermost-first (ne[0] = fast axis).
        // Engine canonical = outermost-first (C order).  Reverse.
        std::vector<std::int64_t> shape_c = rt.shape_innermost_first;
        std::reverse(shape_c.begin(), shape_c.end());

        // Element count
        std::size_t n_elem = 1;
        for (const auto d : shape_c) {
            if (d <= 0) { n_elem = 0; break; }
            n_elem *= static_cast<std::size_t>(d);
        }

        // DType + byte_length
        const auto bi_it = kBlockInfo.find(rt.ggml_type);
        const DType dtype = bi_it != kBlockInfo.end()
            ? bi_it->second.dtype
            : DType::Unknown;
        const std::size_t blen = compute_byte_length(rt.ggml_type, n_elem);

        GgufTensorInfo ti;
        ti.name        = rt.name;
        ti.canonical   = normalise_name(rt.name, hdr.architecture);
        ti.dtype       = dtype;
        ti.shape       = std::move(shape_c);
        ti.byte_offset = rt.offset;
        ti.byte_length = blen;
        hdr.tensors.push_back(std::move(ti));
    }

    return hdr;
}

}  // namespace llmengine
