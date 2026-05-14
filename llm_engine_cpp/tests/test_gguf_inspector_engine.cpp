// test_gguf_inspector_engine.cpp — smoke tests for GgufInspectorEngine.
//
// Constructs a minimal GGUF file (same in-process technique as
// test_gguf_parser.cpp), loads it via GgufInspectorEngine, and verifies:
//   - view().tensors is populated (correct count, names, dtypes, shapes)
//   - view().topology matches the file's KV metadata
//   - view().capabilities reports has_topology + has_state_dict = true
//   - view().provenance.format == "gguf"
//   - A round-trip read_slice() on an F32 tensor returns the expected bytes

#include "llm_engine/gguf_inspector_engine.hpp"
#include "llm_engine/model.hpp"
#include "llm_engine/model_view.hpp"
#include "llm_engine/tensor_handle.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <unistd.h>   // write / close / unlink

using namespace llmengine;

// ── Minimal GGUF builder (copy of the one in test_gguf_parser.cpp) ────────
// (Kept local so each test binary compiles independently without a shared lib.)

struct Buf {
    std::vector<std::uint8_t> data;

    void u8(std::uint8_t v)   { data.push_back(v); }
    void u32(std::uint32_t v) {
        for (int i = 0; i < 4; ++i) data.push_back((v >> (8*i)) & 0xFF);
    }
    void u64(std::uint64_t v) {
        for (int i = 0; i < 8; ++i) data.push_back((v >> (8*i)) & 0xFF);
    }
    void f32(float v) { std::uint32_t bits; std::memcpy(&bits, &v, 4); u32(bits); }

    void gguf_string(const std::string& s) {
        u64(static_cast<std::uint64_t>(s.size()));
        for (char c : s) data.push_back(static_cast<std::uint8_t>(c));
    }
    void kv_string(const std::string& key, const std::string& val) {
        gguf_string(key); u32(8); gguf_string(val);
    }
    void kv_u32(const std::string& key, std::uint32_t val) {
        gguf_string(key); u32(4); u32(val);
    }
    void kv_f32(const std::string& key, float val) {
        gguf_string(key); u32(6); f32(val);
    }
    void tensor_record(const std::string& name,
                       const std::vector<std::uint64_t>& dims,
                       std::uint32_t ggml_type, std::uint64_t offset) {
        gguf_string(name);
        u32(static_cast<std::uint32_t>(dims.size()));
        for (auto d : dims) u64(d);
        u32(ggml_type);
        u64(offset);
    }
    void pad_to(std::size_t al) { while (data.size() % al != 0) data.push_back(0); }

    std::string write_tmpfile() const {
        const char* tmpdir = std::getenv("TMPDIR");
        if (!tmpdir) tmpdir = "/tmp";
        std::string tmpl = std::string(tmpdir) + "/test_gguf_engine_XXXXXX.gguf";
        int fd = ::mkstemps(tmpl.data(), 5);
        assert(fd >= 0);
        const std::size_t written = ::write(fd, data.data(), data.size());
        assert(written == data.size());
        ::close(fd);
        return tmpl;
    }
};

// ── Build a minimal but complete GGUF with one F32 tensor ────────────────
//
// Layout:
//   Header (magic + version + counts)
//   KV: general.architecture = "llama"
//       llama.block_count = 4
//       llama.attention.head_count = 8
//       llama.embedding_length = 64
//   Tensor record: "token_embd.weight" F32(0) shape [16] (innermost {16})
//   <padding to 32>
//   Data section: 16 floats (1.0f .. 16.0f)

static std::string make_test_gguf(std::vector<float>& out_data) {
    // The floats we'll write into the data section
    out_data.clear();
    for (int i = 1; i <= 16; ++i) out_data.push_back(static_cast<float>(i));

    Buf b;
    b.data.insert(b.data.end(), {'G','G','U','F'});
    b.u32(3);        // version
    b.u64(1);        // n_tensors
    b.u64(4);        // n_kv

    b.kv_string("general.architecture", "llama");
    b.kv_u32("llama.block_count",        4);
    b.kv_u32("llama.attention.head_count", 8);
    b.kv_u32("llama.embedding_length",   64);

    // tensor record: "token_embd.weight", F32, shape {16} innermost-first,
    // offset 0 (relative to data section)
    b.tensor_record("token_embd.weight", {16}, 0u, 0u);

    b.pad_to(32);  // default alignment

    // Data section: 16 * 4 = 64 bytes
    for (float v : out_data) b.f32(v);

    return b.write_tmpfile();
}

// ── Tests ──────────────────────────────────────────────────────────────────

static void test_load_success() {
    std::vector<float> expected;
    const std::string path = make_test_gguf(expected);

    GgufInspectorEngine eng;
    const auto result = eng.loadCheckpoint(path);
    ::unlink(path.c_str());

    assert(result.ok);
    assert(result.error.empty());

    // Drain logs — should have at least one info line
    const auto logs = eng.drainEngineLogs();
    assert(!logs.empty());
}

static void test_tensor_registry_populated() {
    std::vector<float> expected;
    const std::string path = make_test_gguf(expected);

    GgufInspectorEngine eng;
    eng.loadCheckpoint(path);
    ::unlink(path.c_str());

    const auto& v = eng.view();
    assert(!v.tensors.empty());
    assert(v.tensors.size() == 1);

    // Name normalised to canonical: "token_embd.weight" → "tok_embeddings.weight"
    const TensorHandle* h = v.tensors.find("tok_embeddings.weight");
    assert(h != nullptr);
    assert(h->dtype == DType::F32);
    assert(h->shape.size() == 1);
    assert(h->shape[0] == 16);
}

static void test_topology_populated() {
    std::vector<float> expected;
    const std::string path = make_test_gguf(expected);

    GgufInspectorEngine eng;
    eng.loadCheckpoint(path);
    ::unlink(path.c_str());

    const auto& v = eng.view();
    assert(v.topology.nLayers == 4);
    assert(v.topology.nHeads  == 8);
    assert(v.topology.dModel  == 64);
    // dHead derived: 64 / 8 = 8
    assert(v.topology.dHead   == 8);
}

static void test_capabilities() {
    std::vector<float> expected;
    const std::string path = make_test_gguf(expected);

    GgufInspectorEngine eng;
    eng.loadCheckpoint(path);
    ::unlink(path.c_str());

    const auto caps = eng.getCapabilities();
    assert(caps.has_topology);
    assert(caps.has_state_dict);
    assert(!caps.has_attention);
    assert(!caps.has_captures);
    assert(!caps.has_training);
}

static void test_provenance() {
    std::vector<float> expected;
    const std::string path = make_test_gguf(expected);

    GgufInspectorEngine eng;
    eng.loadCheckpoint(path);
    ::unlink(path.c_str());

    const auto& prov = eng.view().provenance;
    assert(prov.format == "gguf");
    assert(!prov.path.empty());
    assert(!prov.source_label.empty());
}

static void test_read_slice_round_trip() {
    // Verify that bytes written into the GGUF data section round-trip
    // through a TensorHandle::read_slice() call.
    std::vector<float> expected;
    const std::string path = make_test_gguf(expected);

    GgufInspectorEngine eng;
    eng.loadCheckpoint(path);
    ::unlink(path.c_str());

    const auto& v = eng.view();
    const TensorHandle* h = v.tensors.find("tok_embeddings.weight");
    assert(h != nullptr);
    assert(h->readable());

    // Read all 16 elements
    const auto sliced = h->read_slice(0, 16);
    assert(sliced.size() == 16);
    for (std::size_t i = 0; i < 16; ++i) {
        assert(sliced[i] == expected[i]);
    }

    // Read a mid-tensor slice
    const auto mid = h->read_slice(4, 8);
    assert(mid.size() == 8);
    for (std::size_t i = 0; i < 8; ++i) {
        assert(mid[i] == expected[4 + i]);
    }

    // Out-of-range returns empty
    const auto oob = h->read_slice(100, 4);
    assert(oob.empty());
}

static void test_unload_clears_view() {
    std::vector<float> expected;
    const std::string path = make_test_gguf(expected);

    GgufInspectorEngine eng;
    eng.loadCheckpoint(path);
    ::unlink(path.c_str());

    assert(!eng.view().tensors.empty());

    eng.unloadCheckpoint();
    assert(eng.view().tensors.empty());
    assert(eng.view().provenance.format.empty());
}

static void test_load_bad_path() {
    GgufInspectorEngine eng;
    const auto result = eng.loadCheckpoint("/tmp/this_file_definitely_does_not_exist_xyzzy.gguf");
    assert(!result.ok);
    assert(!result.error.empty());
}

// ── main ──────────────────────────────────────────────────────────────────

int main() {
    test_load_success();
    test_tensor_registry_populated();
    test_topology_populated();
    test_capabilities();
    test_provenance();
    test_read_slice_round_trip();
    test_unload_clears_view();
    test_load_bad_path();
    return 0;
}
