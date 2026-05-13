// DerivedCache tests.
//
// Covers:
//   - get_or_compute calls the lambda exactly once for repeated reads
//   - invalidate(prefix) drops only matching keys
//   - clear() drops everything
//   - LRU eviction kicks in once size_bytes() crosses the soft cap

#include "llm_engine/derived_cache.hpp"

#include <cassert>
#include <string>
#include <vector>

using namespace llmengine;

namespace {

void test_compute_once() {
    // The default get_or_compute<T> uses sizeof(T) for accounting —
    // right for fixed-size POD results.
    DerivedCache c;
    int calls = 0;

    auto v1 = c.get_or_compute<int>("foo", [&]() {
        ++calls;
        return 42;
    });
    assert(*v1 == 42);
    assert(calls == 1);

    // Second hit: cached, no recompute.
    auto v2 = c.get_or_compute<int>("foo", [&]() {
        ++calls;
        return 99;
    });
    assert(*v2 == 42);
    assert(calls == 1);

    // Accounting via sizeof(int) = 4.
    assert(c.size_bytes() == sizeof(int));
}

void test_sized_variant_for_heap_types() {
    // get_or_compute_sized<T> takes (value, size) — right for types
    // whose true footprint isn't captured by sizeof(T).
    DerivedCache c;
    auto v = c.get_or_compute_sized<std::vector<int>>(
        "foo",
        []() { return std::pair{std::vector<int>{1,2,3,4,5,6,7,8}, 32ull}; });
    assert(v->size() == 8);
    assert((*v)[0] == 1);
    assert(c.size_bytes() == 32);
}

void test_invalidate_prefix() {
    DerivedCache c;
    c.get_or_compute<int>("tensors/a", [] { return 1; });
    c.get_or_compute<int>("tensors/b", [] { return 2; });
    c.get_or_compute<int>("surgery/x", [] { return 3; });
    assert(c.size_bytes() == 3 * sizeof(int));

    c.invalidate("tensors/");
    assert(c.size_bytes() == sizeof(int));

    int calls = 0;
    auto v = c.get_or_compute<int>("tensors/a", [&]() {
        ++calls;
        return 7;
    });
    assert(*v == 7);
    assert(calls == 1);
}

void test_clear() {
    DerivedCache c;
    c.get_or_compute_sized<int>("foo",
        [] { return std::pair{1, 100ull}; });
    c.get_or_compute_sized<int>("bar",
        [] { return std::pair{2, 200ull}; });
    assert(c.size_bytes() == 300);
    c.clear();
    assert(c.size_bytes() == 0);
}

void test_lru_eviction() {
    // Soft cap = 100 bytes; insertions sum > 100 should trigger eviction
    // of the LRU entry.
    DerivedCache c(/*soft_cap_bytes=*/100);
    c.get_or_compute_sized<int>("a",
        [] { return std::pair{1, 60ull}; });
    c.get_or_compute_sized<int>("b",
        [] { return std::pair{2, 30ull}; });
    assert(c.size_bytes() == 90);

    // Insert "c" (40 bytes) — total would be 130. Eviction trims the
    // LRU ("a") to make room.
    c.get_or_compute_sized<int>("c",
        [] { return std::pair{3, 40ull}; });
    assert(c.size_bytes() <= 100);

    // "a" should be gone; refetching invokes the compute lambda.
    int calls = 0;
    c.get_or_compute_sized<int>("a", [&]() {
        ++calls;
        return std::pair{1, 10ull};
    });
    assert(calls == 1);
}

}  // namespace

int main() {
    test_compute_once();
    test_sized_variant_for_heap_types();
    test_invalidate_prefix();
    test_clear();
    test_lru_eviction();
    return 0;
}
