#include "llm_engine/derived_cache.hpp"

namespace llmengine {

void DerivedCache::invalidate(std::string_view prefix) {
    std::lock_guard lk(m_mu);
    for (auto it = m_map.begin(); it != m_map.end(); ) {
        if (it->first.rfind(prefix, 0) == 0) {
            m_bytes -= it->second->bytes;
            m_lru.erase(it->second->lru);
            it = m_map.erase(it);
        } else {
            ++it;
        }
    }
}

void DerivedCache::clear() {
    std::lock_guard lk(m_mu);
    m_map.clear();
    m_lru.clear();
    m_bytes = 0;
}

std::size_t DerivedCache::size_bytes() const {
    std::lock_guard lk(m_mu);
    return m_bytes;
}

void DerivedCache::evict_to_fit_(std::size_t incoming_bytes) {
    while (m_bytes + incoming_bytes > m_soft_cap && !m_lru.empty()) {
        const std::string& victim = m_lru.back();
        auto it = m_map.find(victim);
        if (it != m_map.end()) {
            m_bytes -= it->second->bytes;
            m_map.erase(it);
        }
        m_lru.pop_back();
    }
}

}  // namespace llmengine
