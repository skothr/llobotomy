#pragma once
//
// DerivedCache — memoised, byte-bounded LRU store for analyses derived
// from a ModelView (tensor stats, histograms, singular values, logit
// lens trajectories, …).
//
// The cache is type-erased so callers can store heterogeneous values
// without a closed-set variant. Values are held as shared_ptr<void>
// alongside a type_index for safe down-casting in get<T>.
//
// Threading: every operation is guarded by an internal mutex. Lookups
// happen on the UI thread up to ~60 Hz; misses trigger a compute that
// must be cheap (per-frame samplers are still meant to be O(1) reads;
// the first miss for an analysis is allowed to be slower).
//
// Bounded by a soft byte cap — when a new insertion would exceed it,
// LRU eviction proceeds until the new value fits. Tracking is by
// caller-reported byte size so the cache doesn't need to introspect
// the stored type.

#include <cstddef>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>

namespace llmengine {

class DerivedCache {
public:
    explicit DerivedCache(std::size_t soft_cap_bytes = 256ull * 1024 * 1024)
        : m_soft_cap(soft_cap_bytes) {}

    DerivedCache(const DerivedCache&)            = delete;
    DerivedCache& operator=(const DerivedCache&) = delete;

    // Read a cached value or compute + cache it. The compute lambda
    // returns a (value, byte_size) pair; byte_size is the caller's
    // estimate used for eviction. Misses on type mismatch throw — that's
    // a programming error, not a runtime miss.
    template <class T>
    std::shared_ptr<const T>
    get_or_compute(std::string_view key,
                   std::function<std::pair<T, std::size_t>()> compute) {
        std::unique_lock lk(m_mu);
        const std::string k(key);
        auto it = m_map.find(k);
        if (it != m_map.end()) {
            m_lru.splice(m_lru.begin(), m_lru, it->second->lru);
            if (it->second->type != std::type_index(typeid(T))) {
                throw std::runtime_error("DerivedCache: type mismatch on '" + k + "'");
            }
            return std::static_pointer_cast<const T>(it->second->value);
        }
        lk.unlock();
        // Compute outside the lock — compute() can be expensive and
        // shouldn't block other readers.
        auto [val, bytes] = compute();
        auto sp = std::make_shared<T>(std::move(val));

        lk.lock();
        // Another thread may have populated the same key while we were
        // computing. Last-write-wins is fine; both copies are valid.
        auto it2 = m_map.find(k);
        if (it2 != m_map.end()) {
            m_lru.splice(m_lru.begin(), m_lru, it2->second->lru);
            if (it2->second->type == std::type_index(typeid(T))) {
                return std::static_pointer_cast<const T>(it2->second->value);
            }
        }
        evict_to_fit_(bytes);
        m_lru.push_front(k);
        auto entry = std::make_unique<Entry>();
        entry->value = sp;
        entry->type  = std::type_index(typeid(T));
        entry->bytes = bytes;
        entry->lru   = m_lru.begin();
        m_bytes += bytes;
        m_map[k] = std::move(entry);
        return std::static_pointer_cast<const T>(m_map[k]->value);
    }

    // Drop every entry whose key starts with prefix. Used by mutators —
    // e.g. on surgery commit, invalidate("tensors/").
    void invalidate(std::string_view prefix);
    void clear();

    std::size_t size_bytes() const;

private:
    struct Entry {
        std::shared_ptr<void>      value;
        std::type_index            type{typeid(void)};
        std::size_t                bytes = 0;
        std::list<std::string>::iterator lru;
    };

    void evict_to_fit_(std::size_t incoming_bytes);

    mutable std::mutex                          m_mu;
    std::unordered_map<std::string, std::unique_ptr<Entry>> m_map;
    std::list<std::string>                      m_lru;       // front = most recent
    std::size_t                                 m_bytes = 0;
    std::size_t                                 m_soft_cap;
};

}  // namespace llmengine
