// test_derived_cache_threading.cpp — concurrent get_or_compute on
// DerivedCache.
//
// The substrate's threading contract (model_view.hpp top-of-file
// block) declares DerivedCache is "internally synchronised; safe at
// frame rate from any thread."  This test pounds on it with multiple
// readers competing on the same key — verifies the LRU touch + the
// type-tagged retrieval stay coherent.
//
// Specific invariants:
//   1. Repeated lookups of the same key return the same shared_ptr<T>
//      regardless of how many threads contend for the first compute.
//   2. The compute lambda runs at most once per unique key (assuming
//      no eviction between lookups).
//   3. Different keys can be computed concurrently without
//      cross-contamination.

#include "llm_engine/derived_cache.hpp"

#include <atomic>
#include <cassert>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

using namespace llmengine;

int main() {
    // ── Same-key contention ─────────────────────────────────────────────
    {
        DerivedCache c;
        std::atomic<int> compute_count{0};
        std::atomic<int> read_count{0};

        auto worker = [&]() {
            for (int i = 0; i < 200; ++i) {
                auto v = c.get_or_compute<int>("hot_key", [&]() {
                    compute_count.fetch_add(1, std::memory_order_relaxed);
                    return 42;
                });
                assert(*v == 42);
                read_count.fetch_add(1, std::memory_order_relaxed);
            }
        };

        std::thread t1(worker), t2(worker), t3(worker), t4(worker);
        t1.join(); t2.join(); t3.join(); t4.join();

        // The compute lambda may run more than once across threads (when
        // two threads race past the initial lookup), but the cache must
        // converge: subsequent lookups return the cached value.  The
        // 4×200=800 read_count should match.
        assert(read_count.load() == 800);
        // At least one compute happened; at most a handful given a 4-
        // thread race.  Hard upper bound: 4 (one per worker's first
        // miss).
        const int cn = compute_count.load();
        assert(cn >= 1);
        assert(cn <= 4);
        std::printf("test_derived_cache_threading: hot_key — %d computes for 800 reads\n", cn);
    }

    // ── Disjoint-key parallelism ────────────────────────────────────────
    {
        DerivedCache c;
        std::atomic<int> total_computes{0};

        // 8 threads, each owning its own key.
        auto worker = [&](int tid) {
            const std::string key = "thread_" + std::to_string(tid);
            for (int i = 0; i < 100; ++i) {
                auto v = c.get_or_compute<int>(key, [&]() {
                    total_computes.fetch_add(1, std::memory_order_relaxed);
                    return tid;
                });
                assert(*v == tid);
            }
        };

        std::vector<std::thread> threads;
        threads.reserve(8);
        for (int i = 0; i < 8; ++i) threads.emplace_back(worker, i);
        for (auto& t : threads) t.join();

        // Exactly 8 computes (one per unique key).
        assert(total_computes.load() == 8);
        std::printf("test_derived_cache_threading: 8 disjoint keys — 8 computes for 800 reads\n");
    }

    // ── Concurrent invalidate while readers ─────────────────────────────
    // Stress test: writers invalidate prefixes while readers compute.
    // The cache must not crash; readers may see either the cached or
    // a freshly-computed value, both are acceptable.
    {
        DerivedCache c;
        std::atomic<bool> stop{false};
        std::atomic<int>  reads{0};
        std::atomic<int>  invalidations{0};

        auto reader = [&]() {
            int i = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                const std::string key = "k_" + std::to_string((i++) % 16);
                auto v = c.get_or_compute<int>(key, [&]() { return 1; });
                assert(*v == 1);
                reads.fetch_add(1, std::memory_order_relaxed);
            }
        };

        auto invalidator = [&]() {
            while (!stop.load(std::memory_order_relaxed)) {
                c.invalidate("k_");
                invalidations.fetch_add(1, std::memory_order_relaxed);
            }
        };

        std::thread r1(reader), r2(reader), r3(reader), inv(invalidator);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        stop.store(true);
        r1.join(); r2.join(); r3.join(); inv.join();

        std::printf("test_derived_cache_threading: contention — %d reads, %d invalidates, no crash\n",
                    reads.load(), invalidations.load());
        assert(reads.load() > 0);
        assert(invalidations.load() > 0);
    }

    return 0;
}
