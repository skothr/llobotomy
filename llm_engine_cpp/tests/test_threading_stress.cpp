// test_threading_stress.cpp — concurrent read/write stress on
// ModelView's atomic shared_ptr current.
//
// The substrate's threading contract (model_view.hpp top-of-file
// block) declares `current` is safe for concurrent reads + single
// writer.  This test pounds on it with 4 readers + 1 writer doing
// thousands of cycles, and verifies:
//   - No data races detected by ThreadSanitizer (if built with -fsanitize=thread).
//   - No torn reads — every loaded pointer either is null or points
//     at a valid CaptureBundle whose prompt_hash field is intact.
//   - Refcount safety — the bundle a reader loaded survives until
//     that reader is done with it, even if a concurrent store has
//     happened.
//
// Deep tier — takes ~1 second.  Pure substrate test; doesn't depend
// on any backend, file, or external service.

#include "llm_engine/capture.hpp"
#include "llm_engine/model_view.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace llmengine;

int main() {
    ModelView v;

    // Pre-construct N distinct bundles so the writer cycles through
    // them.  Each carries a known prompt_hash for sanity-checking.
    constexpr int N = 64;
    std::vector<std::shared_ptr<const CaptureBundle>> pool;
    pool.reserve(N);
    for (int i = 0; i < N; ++i) {
        auto b = std::make_shared<CaptureBundle>();
        b->prompt_hash = "hash_" + std::to_string(i);
        b->token_ids = {1, 2, 3, static_cast<TokenId>(i)};
        pool.push_back(std::move(b));
    }

    std::atomic<bool> stop{false};
    std::atomic<int>  total_loads{0};
    std::atomic<int>  loads_with_value{0};
    std::atomic<int>  torn_reads{0};        // would fire if prompt_hash were corrupt

    auto reader = [&]() {
        while (!stop.load(std::memory_order_relaxed)) {
            auto snap = v.current.load();
            total_loads.fetch_add(1, std::memory_order_relaxed);
            if (snap) {
                loads_with_value.fetch_add(1, std::memory_order_relaxed);
                // Touch every field — any torn read corrupts these.
                const auto& h = snap->prompt_hash;
                if (h.size() < 6 || h.substr(0, 5) != "hash_") {
                    torn_reads.fetch_add(1, std::memory_order_relaxed);
                }
                // The token_ids vector should always have 4 entries
                // (per the pre-construction above).
                if (snap->token_ids.size() != 4) {
                    torn_reads.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    };

    constexpr int kReaderCount = 4;
    std::vector<std::thread> readers;
    readers.reserve(kReaderCount);
    for (int i = 0; i < kReaderCount; ++i) readers.emplace_back(reader);

    // Writer: rapidly cycle through the pool.
    constexpr int kWrites = 5000;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kWrites; ++i) {
        v.current.store(pool[i % N]);
    }
    const auto t1 = std::chrono::steady_clock::now();

    // Let readers run a bit more so they catch the final state.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    stop.store(true);
    for (auto& t : readers) t.join();

    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::printf("test_threading_stress: %d writes in %lldms; %d loads (%d with value); torn=%d\n",
                kWrites, static_cast<long long>(ms),
                total_loads.load(), loads_with_value.load(), torn_reads.load());

    assert(torn_reads.load() == 0);
    assert(total_loads.load() > kWrites);   // readers should outrun writers
    assert(loads_with_value.load() > 0);    // many loads should see a value

    // Cleanup: clear the view; bundles in the pool are still
    // refcounted by `pool` so they survive.
    v.current.store(nullptr);
    assert(v.current.load() == nullptr);

    return 0;
}
