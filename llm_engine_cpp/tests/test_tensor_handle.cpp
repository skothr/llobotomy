// TensorHandle round-trip tests.
//
// Covers:
//   - InMemoryTensorSource construction from a float vector
//   - F32 read_slice with various offsets
//   - F16 / BF16 dequantisation round-trip
//   - read_slice_2d for the heatmap path
//   - Out-of-range slices return {} (not a throw)
//   - dtype_element_bytes table is internally consistent

#include "llm_engine/tensor_handle.hpp"
#include "llm_engine/tensor_source.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace llmengine;

namespace {

void test_f32_round_trip() {
    std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto src = InMemoryTensorSource::from_floats(data);

    TensorHandle h;
    h.source      = src;
    h.name        = "test/f32";
    h.dtype       = DType::F32;
    h.shape       = {2, 3};
    h.byte_offset = 0;
    h.byte_length = data.size() * sizeof(float);

    assert(h.element_count() == 6);
    assert(h.valid());

    auto all = h.read_slice(0, 6);
    assert(all.size() == 6);
    for (std::size_t i = 0; i < 6; ++i) assert(all[i] == data[i]);

    // Mid-tensor slice
    auto mid = h.read_slice(2, 3);
    assert(mid.size() == 3);
    assert(mid[0] == 3.0f && mid[1] == 4.0f && mid[2] == 5.0f);

    // Out-of-range returns empty
    auto oob = h.read_slice(10, 1);
    assert(oob.empty());

    // Partial overlap clamps to what's available
    auto partial = h.read_slice(5, 100);
    assert(partial.size() == 1);
    assert(partial[0] == 6.0f);
}

void test_f16_dequant() {
    // Encode 1.0, 2.0, 3.0 as f16 bits and verify round-trip.
    // f16(1.0) = 0x3C00, f16(2.0) = 0x4000, f16(3.0) = 0x4200.
    std::vector<std::uint16_t> bits{0x3C00, 0x4000, 0x4200};
    std::vector<std::byte> raw(bits.size() * 2);
    std::memcpy(raw.data(), bits.data(), raw.size());
    auto src = std::make_shared<InMemoryTensorSource>(std::move(raw));

    TensorHandle h;
    h.source      = src;
    h.name        = "test/f16";
    h.dtype       = DType::F16;
    h.shape       = {3};
    h.byte_offset = 0;
    h.byte_length = bits.size() * 2;

    auto out = h.read_slice(0, 3);
    assert(out.size() == 3);
    assert(out[0] == 1.0f);
    assert(out[1] == 2.0f);
    assert(out[2] == 3.0f);
}

void test_bf16_dequant() {
    // bf16(1.0) = 0x3F80, bf16(-2.0) = 0xC000.
    std::vector<std::uint16_t> bits{0x3F80, 0xC000};
    std::vector<std::byte> raw(bits.size() * 2);
    std::memcpy(raw.data(), bits.data(), raw.size());
    auto src = std::make_shared<InMemoryTensorSource>(std::move(raw));

    TensorHandle h;
    h.source      = src;
    h.name        = "test/bf16";
    h.dtype       = DType::BF16;
    h.shape       = {2};
    h.byte_offset = 0;
    h.byte_length = bits.size() * 2;

    auto out = h.read_slice(0, 2);
    assert(out.size() == 2);
    assert(out[0] == 1.0f);
    assert(out[1] == -2.0f);
}

void test_2d_slice() {
    // 3×4 row-major matrix:  rows = [10..13], [20..23], [30..33].
    std::vector<float> data{
        10, 11, 12, 13,
        20, 21, 22, 23,
        30, 31, 32, 33,
    };
    auto src = InMemoryTensorSource::from_floats(data);
    TensorHandle h;
    h.source      = src;
    h.name        = "test/2d";
    h.dtype       = DType::F32;
    h.shape       = {3, 4};
    h.byte_offset = 0;
    h.byte_length = data.size() * sizeof(float);

    // Sub-rectangle (rows 1-2, cols 1-3) → [[21,22,23],[31,32,33]]
    auto sub = h.read_slice_2d(1, 2, 1, 3);
    assert(sub.size() == 2);
    assert(sub[0].size() == 3);
    assert(sub[0][0] == 21.0f && sub[0][1] == 22.0f && sub[0][2] == 23.0f);
    assert(sub[1][0] == 31.0f && sub[1][1] == 32.0f && sub[1][2] == 33.0f);

    // Out-of-range row returns empty
    auto bad = h.read_slice_2d(10, 1, 0, 1);
    assert(bad.empty());

    // Clamping: ask for more cols than exist
    auto clamp = h.read_slice_2d(0, 1, 2, 100);
    assert(clamp.size() == 1);
    assert(clamp[0].size() == 2);
    assert(clamp[0][0] == 12.0f && clamp[0][1] == 13.0f);
}

void test_registry() {
    TensorRegistry reg;
    assert(reg.empty());

    TensorHandle a;
    a.name = "alpha";
    TensorHandle b;
    b.name = "beta";
    reg.insert(a);
    reg.insert(b);

    assert(reg.size() == 2);
    assert(reg.find("alpha") != nullptr);
    assert(reg.find("beta")  != nullptr);
    assert(reg.find("gamma") == nullptr);

    // Re-insert with same name overwrites, doesn't append.
    TensorHandle a2;
    a2.name        = "alpha";
    a2.byte_offset = 999;
    reg.insert(a2);
    assert(reg.size() == 2);
    assert(reg.find("alpha")->byte_offset == 999);
}

void test_canonical_path_validation() {
    TensorRegistry reg;

    // Canonical names accepted: letters, digits, dots, underscores, slashes.
    auto ok = [&](std::string name) {
        TensorHandle h;
        h.name = std::move(name);
        reg.insert(h);
    };
    ok("blocks.0.attn.W_Q.weight");
    ok("ln_f.weight");
    ok("alpha/beta/gamma");
    ok("lm_head.weight");

    // Non-canonical names throw — spaces, hashes, query chars, brackets, …
    auto rejects = [&](std::string name) {
        TensorHandle h;
        h.name = std::move(name);
        bool threw = false;
        try { reg.insert(h); }
        catch (const std::invalid_argument&) { threw = true; }
        assert(threw);
    };
    rejects("foo bar");
    rejects("");
    rejects("foo/");
    rejects("/foo");
    rejects("foo//bar");
    rejects("foo#stats");
    rejects("foo?bins=64");
    rejects("foo[0]");
}

void test_tri_state() {
    // Empty handle: not valid, not readable, not loaded.
    TensorHandle empty;
    assert(!empty.valid());
    assert(!empty.readable());
    assert(!empty.loaded());

    // Valid + readable + loaded: in-memory source.
    std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f};
    TensorHandle h_loaded;
    h_loaded.source      = InMemoryTensorSource::from_floats(data);
    h_loaded.name        = "x";
    h_loaded.dtype       = DType::F32;
    h_loaded.shape       = {4};
    h_loaded.byte_length = data.size() * sizeof(float);
    assert(h_loaded.valid());
    assert(h_loaded.readable());
    assert(h_loaded.loaded());

    // Valid + readable, NOT loaded: PRNG source generates on demand.
    TensorHandle h_lazy;
    h_lazy.source      = std::make_shared<Mulberry32Source>(42u, 16);
    h_lazy.name        = "y";
    h_lazy.dtype       = DType::F32;
    h_lazy.shape       = {4};
    h_lazy.byte_length = 16;
    assert(h_lazy.valid());
    assert(h_lazy.readable());
    assert(!h_lazy.loaded());

    // Source-less but config-correct: valid but not readable.
    TensorHandle h_config_only;
    h_config_only.name        = "z";
    h_config_only.dtype       = DType::F32;
    h_config_only.shape       = {4};
    h_config_only.byte_length = 16;
    assert(h_config_only.valid());
    assert(!h_config_only.readable());
    assert(!h_config_only.loaded());

    // Read from the PRNG source — verify determinism.
    auto first = h_lazy.read_slice(0, 4);
    auto again = h_lazy.read_slice(0, 4);
    assert(first.size() == 4);
    assert(first == again);
}

void test_integer_dequant_round_trip() {
    // I8 / I16 / I32 / U8 all cast to f32 — verify a small known vector
    // round-trips through each integer dtype.
    auto check_i = [](DType dt, std::size_t bpe) {
        // Use values -3..+4 to exercise sign extension on signed types.
        std::vector<int> ints{-3, -2, -1, 0, 1, 2, 3, 4};
        std::vector<std::byte> raw(ints.size() * bpe);
        // Pack into the right byte size.  We assume little-endian host
        // (build target matches; cross-host is GGUF-spec territory and
        // belongs in a separate test).
        for (std::size_t i = 0; i < ints.size(); ++i) {
            const std::int64_t v = ints[i];
            for (std::size_t b = 0; b < bpe; ++b) {
                raw[i * bpe + b] = static_cast<std::byte>((v >> (8 * b)) & 0xFF);
            }
        }
        auto src = std::make_shared<InMemoryTensorSource>(std::move(raw));
        TensorHandle h;
        h.source      = src;
        h.name        = "test/int";
        h.dtype       = dt;
        h.shape       = {static_cast<std::int64_t>(ints.size())};
        h.byte_offset = 0;
        h.byte_length = ints.size() * bpe;
        auto out = h.read_slice(0, ints.size());
        assert(out.size() == ints.size());
        for (std::size_t i = 0; i < ints.size(); ++i) {
            assert(out[i] == static_cast<float>(ints[i]));
        }
    };
    check_i(DType::I8,  1);
    check_i(DType::I16, 2);
    check_i(DType::I32, 4);

    // U8: same harness but unsigned.
    std::vector<std::uint8_t> us{0, 1, 127, 128, 255};
    std::vector<std::byte> raw(us.size());
    for (std::size_t i = 0; i < us.size(); ++i) raw[i] = static_cast<std::byte>(us[i]);
    auto src = std::make_shared<InMemoryTensorSource>(std::move(raw));
    TensorHandle h;
    h.source      = src;
    h.name        = "test/u8";
    h.dtype       = DType::U8;
    h.shape       = {static_cast<std::int64_t>(us.size())};
    h.byte_offset = 0;
    h.byte_length = us.size();
    auto out = h.read_slice(0, us.size());
    assert(out.size() == us.size());
    for (std::size_t i = 0; i < us.size(); ++i) assert(out[i] == static_cast<float>(us[i]));
}

void test_registry_at_throws() {
    TensorRegistry reg;
    TensorHandle h;
    h.name = "alpha";
    reg.insert(h);

    // Hit succeeds.
    const auto& hit = reg.at("alpha");
    assert(hit.name == "alpha");

    // Miss throws std::out_of_range.
    bool threw = false;
    try { (void)reg.at("missing"); }
    catch (const std::out_of_range&) { threw = true; }
    assert(threw);
}

void test_unsupported_dtype_returns_empty() {
    // Q4_0 has no dequantiser in this build — read_slice should return {}
    // rather than throwing or asserting.
    std::vector<float> data(16, 1.0f);
    auto src = InMemoryTensorSource::from_floats(data);

    TensorHandle h;
    h.source      = src;
    h.name        = "test/q4_0";
    h.dtype       = DType::Q4_0;
    h.shape       = {16};
    h.byte_offset = 0;
    h.byte_length = data.size() * sizeof(float);

    auto out = h.read_slice(0, 4);
    assert(out.empty());
}

}  // namespace

int main() {
    test_f32_round_trip();
    test_f16_dequant();
    test_bf16_dequant();
    test_2d_slice();
    test_registry();
    test_canonical_path_validation();
    test_tri_state();
    test_integer_dequant_round_trip();
    test_registry_at_throws();
    test_unsupported_dtype_returns_empty();
    return 0;
}
