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
    DerivedCache c;
    int calls = 0;

    auto v1 = c.get_or_compute<int>("foo", [&]() {
        ++calls;
        return std::pair<int, std::size_t>{42, 4};
    });
    assert(*v1 == 42);
    assert(calls == 1);

    // Second hit: cached, no recompute.
    auto v2 = c.get_or_compute<int>("foo", [&]() {
        ++calls;
        return std::pair<int, std::size_t>{99, 4};
    });
    assert(*v2 == 42);
    assert(calls == 1);
}

void test_invalidate_prefix() {
    DerivedCache c;
    c.get_or_compute<int>("tensors/a", [] { return std::pair<int, std::size_t>{1, 4}; });
    c.get_or_compute<int>("tensors/b", [] { return std::pair<int, std::size_t>{2, 4}; });
    c.get_or_compute<int>("surgery/x", [] { return std::pair<int, std::size_t>{3, 4}; });
    assert(c.size_bytes() == 12);

    c.invalidate("tensors/");
    assert(c.size_bytes() == 4);

    int calls = 0;
    auto v = c.get_or_compute<int>("tensors/a", [&]() {
        ++calls;
        return std::pair<int, std::size_t>{7, 4};
    });
    assert(*v == 7);
    assert(calls == 1);
}

void test_clear() {
    DerivedCache c;
    c.get_or_compute<int>("foo", [] { return std::pair<int, std::size_t>{1, 100}; });
    c.get_or_compute<int>("bar", [] { return std::pair<int, std::size_t>{2, 200}; });
    assert(c.size_bytes() == 300);
    c.clear();
    assert(c.size_bytes() == 0);
}

void test_lru_eviction() {
    // Soft cap = 100 bytes; insertions sum > 100 should trigger eviction
    // of the LRU entry.
    DerivedCache c(/*soft_cap_bytes=*/100);
    c.get_or_compute<int>("a", [] { return std::pair<int, std::size_t>{1, 60}; });
    c.get_or_compute<int>("b", [] { return std::pair<int, std::size_t>{2, 30}; });
    assert(c.size_bytes() == 90);

    // Insert "c" (40 bytes) — total would be 130. Eviction trims the
    // LRU ("a") to make room.
    c.get_or_compute<int>("c", [] { return std::pair<int, std::size_t>{3, 40}; });
    assert(c.size_bytes() <= 100);

    // "a" should be gone; refetching invokes the compute lambda.
    int calls = 0;
    c.get_or_compute<int>("a", [&]() {
        ++calls;
        return std::pair<int, std::size_t>{1, 10};
    });
    assert(calls == 1);
}

}  // namespace

int main() {
    test_compute_once();
    test_invalidate_prefix();
    test_clear();
    test_lru_eviction();
    return 0;
}
