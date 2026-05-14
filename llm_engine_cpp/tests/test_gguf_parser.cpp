// test_gguf_parser.cpp — smoke tests for the GGUF header parser.
//
// Constructs minimal well-formed GGUF byte buffers in-process, writes
// them to temp files, and verifies parse_gguf() returns the expected
// GgufHeader.  No real model file required.
//
// GGUF binary layout (little-endian throughout):
//   4B  magic "GGUF"
//   4B  version (u32)
//   8B  n_tensors (u64)
//   8B  n_kv (u64)
//   [n_kv KV pairs]
//     8B  key_len (u64)
//     key_len B  key (UTF-8)
//     4B  value_type (u32)
//     [value bytes]
//   [n_tensors tensor records]
//     8B  name_len
//     name_len B  name
//     4B  n_dims (u32)
//     n_dims * 8B  dims (u64 each, innermost-first)
//     4B  ggml_type (u32)
//     8B  byte_offset (u64, relative to data section)
//   <alignment padding to `general.alignment` (default 32)>
//   [data section — tensor bytes at their declared offsets]

#include "llm_engine/gguf_parser.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

using namespace llmengine;

// ── Binary writer helpers ─────────────────────────────────────────────────

struct Buffer {
    std::vector<std::uint8_t> data;

    void u8(std::uint8_t v)   { data.push_back(v); }
    void u16(std::uint16_t v) { data.push_back(v & 0xFF); data.push_back(v >> 8); }
    void u32(std::uint32_t v) {
        data.push_back( v        & 0xFF);
        data.push_back((v >>  8) & 0xFF);
        data.push_back((v >> 16) & 0xFF);
        data.push_back((v >> 24) & 0xFF);
    }
    void u64(std::uint64_t v) {
        for (int i = 0; i < 8; ++i) data.push_back((v >> (8*i)) & 0xFF);
    }
    void i32(std::int32_t v) { u32(static_cast<std::uint32_t>(v)); }
    void f32(float v)        { std::uint32_t bits; std::memcpy(&bits, &v, 4); u32(bits); }
    void f64(double v)       { std::uint64_t bits; std::memcpy(&bits, &v, 8); u64(bits); }

    // GGUF string: 8B length + bytes (no NUL terminator)
    void gguf_string(const std::string& s) {
        u64(static_cast<std::uint64_t>(s.size()));
        for (char c : s) data.push_back(static_cast<std::uint8_t>(c));
    }

    // KV string value (type tag 8)
    void kv_string(const std::string& key, const std::string& val) {
        gguf_string(key);
        u32(8);  // GGUF_VALUE_TYPE_STRING
        gguf_string(val);
    }

    // KV uint32 value (type tag 4)
    void kv_u32(const std::string& key, std::uint32_t val) {
        gguf_string(key);
        u32(4);  // GGUF_VALUE_TYPE_UINT32
        u32(val);
    }

    // KV float32 value (type tag 6)
    void kv_f32(const std::string& key, float val) {
        gguf_string(key);
        u32(6);  // GGUF_VALUE_TYPE_FLOAT32
        f32(val);
    }

    // Tensor record: name + n_dims + dims + ggml_type + offset
    // dims = GGUF innermost-first (ne[0] is fast axis)
    void tensor_record(const std::string& name,
                       const std::vector<std::uint64_t>& dims,
                       std::uint32_t ggml_type, std::uint64_t offset) {
        gguf_string(name);
        u32(static_cast<std::uint32_t>(dims.size()));
        for (auto d : dims) u64(d);
        u32(ggml_type);
        u64(offset);
    }

    // Pad to the given alignment boundary
    void pad_to(std::size_t alignment) {
        while (data.size() % alignment != 0) data.push_back(0);
    }

    // Write to a temp file; return the path.  Caller must unlink.
    std::string write_tmpfile(const std::string& suffix = ".gguf") const {
        std::string tmpl = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp")
                           + "/test_gguf_XXXXXX" + suffix;
        // mkstemps: N suffix chars
        const int slen = static_cast<int>(suffix.size());
        int fd = ::mkstemps(tmpl.data(), slen);
        assert(fd >= 0);
        const std::size_t written = ::write(fd, data.data(), data.size());
        assert(written == data.size());
        ::close(fd);
        return tmpl;
    }
};

// Additional POSIX header for write() / close() used by write_tmpfile.
#include <unistd.h>

// ── Test: minimal valid GGUF (header only, no tensors, no KV) ───────────

static void test_minimal() {
    Buffer b;
    b.data.insert(b.data.end(), {'G','G','U','F'});  // magic
    b.u32(3);  // version 3
    b.u64(0);  // n_tensors
    b.u64(0);  // n_kv
    // No header padding needed (no tensors, so data_section_offset = file_end)

    const auto path = b.write_tmpfile();
    const auto hdr  = parse_gguf(path);
    ::unlink(path.c_str());

    assert(hdr.has_value());
    assert(hdr->version == 3);
    assert(hdr->tensors.empty());
    assert(hdr->architecture.empty());
    assert(hdr->n_layers == -1);
}

// ── Test: KV metadata extraction (llama architecture) ───────────────────

static void test_kv_metadata_llama() {
    // Build a GGUF with the standard llama topology KV keys.
    Buffer b;
    b.data.insert(b.data.end(), {'G','G','U','F'});
    b.u32(3);
    b.u64(0);   // n_tensors = 0 (topology only, no weights)
    b.u64(9);   // n_kv pairs

    b.kv_string("general.architecture", "llama");
    b.kv_u32("llama.block_count",          22);
    b.kv_u32("llama.attention.head_count",  32);
    b.kv_u32("llama.attention.head_count_kv", 4);
    b.kv_u32("llama.embedding_length",      2048);
    b.kv_u32("llama.feed_forward_length",   5632);
    b.kv_u32("llama.context_length",        2048);
    b.kv_f32("llama.rope.freq_base",        10000.0f);
    b.kv_u32("tokenizer.ggml.token_count",  32000);

    const auto path = b.write_tmpfile();
    const auto hdr  = parse_gguf(path);
    ::unlink(path.c_str());

    assert(hdr.has_value());
    assert(hdr->architecture == "llama");
    assert(hdr->n_layers   == 22);
    assert(hdr->n_heads    == 32);
    assert(hdr->n_kv_heads ==  4);
    assert(hdr->d_model    == 2048);
    assert(hdr->d_head     == 64);   // 2048 / 32
    assert(hdr->d_mlp      == 5632);
    assert(hdr->max_pos    == 2048);
    assert(hdr->vocab      == 32000);
    // rope_theta should be close to 10000.0 (float comparison with tolerance)
    assert(hdr->rope_theta > 9999.0f && hdr->rope_theta < 10001.0f);
}

// ── Test: tensor table — names, dtypes, shapes, offsets ─────────────────

static void test_tensor_table() {
    // Two tensors: one F32 1-D, one Q8_0 2-D
    //   F32 vector [4]:    shape (4,) innermost = {4}     → 4 * 4 = 16 bytes
    //   Q8_0 matrix [2,64]: shape innermost = {64, 2}     → Q8_0 block=(32,34)
    //                        elements = 128 → 4 blocks → 4*34 = 136 bytes
    //   Offsets must be block-aligned (not checked by parser per-se).

    const std::uint64_t off_f32  = 0;
    const std::uint64_t off_q8_0 = 32;  // put after 16B + 16B pad = 32

    Buffer b;
    b.data.insert(b.data.end(), {'G','G','U','F'});
    b.u32(3);
    b.u64(2);   // n_tensors
    b.u64(1);   // n_kv
    b.kv_string("general.architecture", "llama");

    // Tensor 0: "token_embd.weight" — F32(0), shape [4]
    b.tensor_record("token_embd.weight", {4}, 0, off_f32);

    // Tensor 1: "blk.0.attn_q.weight" — Q8_0(8), shape [64, 2] (innermost first)
    b.tensor_record("blk.0.attn_q.weight", {64, 2}, 8, off_q8_0);

    // Pad to default alignment 32
    b.pad_to(32);

    // Data section: 16 bytes F32 tensor + 136 bytes Q8_0 tensor (at offset 32)
    // F32 data (4 floats: 1.0, 2.0, 3.0, 4.0)
    for (int i = 1; i <= 4; ++i) b.f32(static_cast<float>(i));
    // Q8_0 data placeholder (136 bytes)
    for (int i = 0; i < 136; ++i) b.u8(0);

    const auto path = b.write_tmpfile();
    const auto hdr  = parse_gguf(path);
    ::unlink(path.c_str());

    assert(hdr.has_value());
    assert(hdr->tensors.size() == 2);

    // Tensor 0: "token_embd.weight" → canonical "tok_embeddings.weight"
    const auto& t0 = hdr->tensors[0];
    assert(t0.name == "token_embd.weight");
    assert(t0.canonical == "tok_embeddings.weight");
    assert(t0.dtype == DType::F32);
    assert(t0.shape.size() == 1);
    assert(t0.shape[0] == 4);
    assert(t0.byte_offset == off_f32);
    assert(t0.byte_length == 16);  // 4 floats * 4 bytes

    // Tensor 1: "blk.0.attn_q.weight" → canonical "blocks.0.attn.W_Q.weight"
    const auto& t1 = hdr->tensors[1];
    assert(t1.name == "blk.0.attn_q.weight");
    assert(t1.canonical == "blocks.0.attn.W_Q.weight");
    assert(t1.dtype == DType::Q8_0);
    // Shape: GGUF innermost-first {64, 2} → C-order {2, 64}
    assert(t1.shape.size() == 2);
    assert(t1.shape[0] == 2);
    assert(t1.shape[1] == 64);
    assert(t1.byte_offset == off_q8_0);
    // Q8_0: 128 elements / 32 vpb = 4 blocks * 34 bpb = 136 bytes
    assert(t1.byte_length == 136);
}

// ── Test: bad magic rejected ──────────────────────────────────────────────

static void test_bad_magic() {
    Buffer b;
    b.data.insert(b.data.end(), {'G','G','F','U'});  // wrong magic
    b.u32(3); b.u64(0); b.u64(0);
    const auto path = b.write_tmpfile();
    const auto hdr  = parse_gguf(path);
    ::unlink(path.c_str());
    assert(!hdr.has_value());
}

// ── Test: unsupported version rejected ───────────────────────────────────

static void test_bad_version() {
    Buffer b;
    b.data.insert(b.data.end(), {'G','G','U','F'});
    b.u32(99); b.u64(0); b.u64(0);  // version 99
    const auto path = b.write_tmpfile();
    const auto hdr  = parse_gguf(path);
    ::unlink(path.c_str());
    assert(!hdr.has_value());
}

// ── Test: custom alignment honoured ──────────────────────────────────────

static void test_alignment() {
    Buffer b;
    b.data.insert(b.data.end(), {'G','G','U','F'});
    b.u32(3);
    b.u64(0);  // n_tensors
    b.u64(1);  // n_kv — just alignment
    b.kv_u32("general.alignment", 64);

    const auto path = b.write_tmpfile();
    const auto hdr  = parse_gguf(path);
    ::unlink(path.c_str());

    assert(hdr.has_value());
    assert(hdr->alignment == 64);
    // data_section_offset must be a multiple of 64
    assert(hdr->data_section_offset % 64 == 0);
}

// ── Test: kv_int / kv_string helpers ─────────────────────────────────────

static void test_kv_helpers() {
    Buffer b;
    b.data.insert(b.data.end(), {'G','G','U','F'});
    b.u32(3);
    b.u64(0);  // n_tensors
    b.u64(2);
    b.kv_string("general.architecture", "qwen2");
    b.kv_u32("qwen2.block_count", 28);

    const auto path = b.write_tmpfile();
    const auto hdr  = parse_gguf(path);
    ::unlink(path.c_str());

    assert(hdr.has_value());
    assert(hdr->kv_string("general.architecture") == "qwen2");
    assert(hdr->kv_int("qwen2.block_count") == 28);
    assert(hdr->kv_int("nonexistent.key") == -1);  // fallback
    assert(hdr->kv_int("nonexistent.key", 42) == 42);
}

// ── main ──────────────────────────────────────────────────────────────────

int main() {
    test_minimal();
    test_kv_metadata_llama();
    test_tensor_table();
    test_bad_magic();
    test_bad_version();
    test_alignment();
    test_kv_helpers();
    return 0;
}
