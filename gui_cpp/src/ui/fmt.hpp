#pragma once
//
// fmt — sentinel-aware string formatters for `Model::*` return values.
// Every workspace uses these so a `kNoFloat` / `kNoInt` / `kNoSize` from the
// backend renders consistently as "—" instead of e.g. "nan" or "-1".

#include "model/model.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>

namespace llob {

inline std::string FmtFloat(float v, const char* spec = "%.3f") {
    if (std::isnan(v)) return "—";
    char buf[32]; std::snprintf(buf, sizeof buf, spec, double(v));
    return buf;
}
inline std::string FmtInt(int v, const char* spec = "%d") {
    if (v == kNoInt) return "—";
    char buf[32]; std::snprintf(buf, sizeof buf, spec, v);
    return buf;
}
inline std::string FmtSize(std::int64_t v) {
    if (v == kNoSize) return "—";
    char buf[32];
    if      (v >= (std::int64_t(1) << 30)) std::snprintf(buf, sizeof buf, "%.1f GB", v / double(1ll << 30));
    else if (v >= (std::int64_t(1) << 20)) std::snprintf(buf, sizeof buf, "%.0f MB", v / double(1ll << 20));
    else if (v >= (std::int64_t(1) << 10)) std::snprintf(buf, sizeof buf, "%.0f KB", v / double(1ll << 10));
    else                                   std::snprintf(buf, sizeof buf, "%lld B", static_cast<long long>(v));
    return buf;
}
inline std::string FmtTokens(std::int64_t n) {
    if (n == kNoSize) return "—";
    char buf[32];
    if      (n >= 1'000'000'000ll) std::snprintf(buf, sizeof buf, "%.1fB",  n / 1e9);
    else if (n >= 1'000'000ll)     std::snprintf(buf, sizeof buf, "%.0fM",  n / 1e6);
    else if (n >= 1'000ll)         std::snprintf(buf, sizeof buf, "%.0fK",  n / 1e3);
    else                            std::snprintf(buf, sizeof buf, "%lld", static_cast<long long>(n));
    return buf;
}
inline std::string FmtPct(float v01, const char* spec = "%.0f%%") {
    if (std::isnan(v01)) return "—";
    char buf[16]; std::snprintf(buf, sizeof buf, spec, double(v01 * 100.0f));
    return buf;
}
inline std::string Or(std::string_view s, const char* fallback = "—") {
    return s.empty() ? std::string(fallback) : std::string(s);
}

}  // namespace llob
