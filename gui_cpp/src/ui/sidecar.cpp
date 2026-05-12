// Per-checkpoint sidecar JSON.  Schema (single line per array for grep-
// friendliness):
//
//   {
//     "checkpoint": "/abs/path/to/model.pt",
//     "ablated_heads":      ["4.3", "4.5", "2.2"],
//     "probed_heads":       ["4.3", "5.1"],
//     "ablated_components": ["4.W_Q"],
//     "probed_components":  [],
//     "skipped_layers":     [],
//     "expanded_layers":    [4]
//   }
//
// Hand-rolled minimal JSON read/write — no third-party dep.  Robust
// against added/removed schema fields (each line is parsed independently;
// unknown keys are ignored).  Strict about the shape it WRITES so the file
// stays grep-able.

#include "ui/sidecar.hpp"

#include "logger.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>

namespace llob {

namespace {

// Sidecar lives next to the checkpoint with the original extension
// stripped.  /foo/bar/model.pt → /foo/bar/model.llobotomy.json
std::filesystem::path Build(std::string_view ckpt) {
    if (ckpt.empty()) return {};
    std::filesystem::path p(ckpt);
    return p.parent_path() / (p.stem().string() + ".llobotomy.json");
}

// Strip whitespace + brackets from an array body.
std::string_view StripArray(std::string_view s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t' ||
                          s.front() == '[' || s.front() == ']')) s.remove_prefix(1);
    while (!s.empty() && (s.back()  == ' ' || s.back()  == '\t' ||
                          s.back()  == '[' || s.back()  == ']' ||
                          s.back()  == ',')) s.remove_suffix(1);
    return s;
}

template <typename Out>
void ParseStringArray(std::string_view body, Out& out) {
    body = StripArray(body);
    while (!body.empty()) {
        const auto comma = body.find(',');
        std::string_view tok = (comma == std::string_view::npos) ? body : body.substr(0, comma);
        // strip quotes + spaces
        while (!tok.empty() && (tok.front() == ' ' || tok.front() == '"')) tok.remove_prefix(1);
        while (!tok.empty() && (tok.back()  == ' ' || tok.back()  == '"')) tok.remove_suffix(1);
        if (!tok.empty()) out.insert(std::string(tok));
        if (comma == std::string_view::npos) break;
        body.remove_prefix(comma + 1);
    }
}

template <typename Out>
void ParseIntArray(std::string_view body, Out& out) {
    body = StripArray(body);
    while (!body.empty()) {
        const auto comma = body.find(',');
        std::string_view tok = (comma == std::string_view::npos) ? body : body.substr(0, comma);
        while (!tok.empty() && (tok.front() == ' ')) tok.remove_prefix(1);
        while (!tok.empty() && (tok.back()  == ' ')) tok.remove_suffix(1);
        if (!tok.empty()) {
            const std::string num(tok);
            char* end = nullptr;
            const long v = std::strtol(num.c_str(), &end, 10);
            if (end != num.c_str()) out.insert(static_cast<int>(v));
        }
        if (comma == std::string_view::npos) break;
        body.remove_prefix(comma + 1);
    }
}

// Locate the value (between the first '[' on the matching key line and
// the matching ']').  We restrict the search to the line for grep-
// friendliness AND to keep the parser shape-tolerant.
std::string_view FindArrayBody(std::string_view body, std::string_view key) {
    std::string needle = "\"" + std::string(key) + "\"";
    const auto kpos = body.find(needle);
    if (kpos == std::string_view::npos) return {};
    const auto lb = body.find('[', kpos);
    if (lb == std::string_view::npos) return {};
    const auto rb = body.find(']', lb);
    if (rb == std::string_view::npos) return {};
    return body.substr(lb, rb - lb + 1);
}

}  // namespace

std::filesystem::path SidecarPath(std::string_view ckpt) { return Build(ckpt); }

void SidecarLoad(AppState& s, std::string_view checkpointPath) {
    const auto p = Build(checkpointPath);
    if (p.empty()) return;
    std::ifstream in(p);
    if (!in) {
        LLOB_LOG_DEBUG("sidecar", "no sidecar at %s — fresh state", p.c_str());
        return;
    }
    std::stringstream buf; buf << in.rdbuf();
    const std::string body = buf.str();

    // Replace, don't append — the file is the new source of truth.
    s.ablatedHeads.clear();
    s.probedHeads.clear();
    s.ablatedComponents.clear();
    s.probedComponents.clear();
    s.skippedLayers.clear();
    s.expandedLayers.clear();

    ParseStringArray(FindArrayBody(body, "ablated_heads"),      s.ablatedHeads);
    ParseStringArray(FindArrayBody(body, "probed_heads"),       s.probedHeads);
    ParseStringArray(FindArrayBody(body, "ablated_components"), s.ablatedComponents);
    ParseStringArray(FindArrayBody(body, "probed_components"),  s.probedComponents);
    ParseIntArray   (FindArrayBody(body, "skipped_layers"),     s.skippedLayers);
    ParseIntArray   (FindArrayBody(body, "expanded_layers"),    s.expandedLayers);

    LLOB_LOG_INFO("sidecar", "loaded %s (%zuh ablate · %zuh probe · %zuc ablate · %zuc probe)",
                  p.c_str(),
                  s.ablatedHeads.size(),  s.probedHeads.size(),
                  s.ablatedComponents.size(), s.probedComponents.size());
}

namespace {

template <typename Set>
void EmitStringArray(std::ofstream& out, const char* key, const Set& items) {
    out << "  \"" << key << "\": [";
    bool first = true;
    for (const auto& s : items) {
        if (!first) out << ", ";
        out << "\"" << s << "\"";
        first = false;
    }
    out << "]";
}
template <typename Set>
void EmitIntArray(std::ofstream& out, const char* key, const Set& items) {
    out << "  \"" << key << "\": [";
    bool first = true;
    for (int v : items) {
        if (!first) out << ", ";
        out << v;
        first = false;
    }
    out << "]";
}

}  // namespace

bool SidecarSave(const AppState& s, std::string_view checkpointPath) {
    const auto p = Build(checkpointPath);
    if (p.empty()) return false;

    std::error_code ec;
    std::filesystem::create_directories(p.parent_path(), ec);   // best-effort

    const auto tmp = p.string() + ".tmp";
    std::ofstream out(tmp);
    if (!out) {
        LLOB_LOG_WARN("sidecar", "could not open %s for write", tmp.c_str());
        return false;
    }
    out << "{\n"
        << "  \"checkpoint\": \"" << checkpointPath << "\",\n";
    EmitStringArray(out, "ablated_heads",      s.ablatedHeads);      out << ",\n";
    EmitStringArray(out, "probed_heads",       s.probedHeads);       out << ",\n";
    EmitStringArray(out, "ablated_components", s.ablatedComponents); out << ",\n";
    EmitStringArray(out, "probed_components",  s.probedComponents);  out << ",\n";
    EmitIntArray   (out, "skipped_layers",     s.skippedLayers);     out << ",\n";
    EmitIntArray   (out, "expanded_layers",    s.expandedLayers);    out << "\n";
    out << "}\n";
    out.close();

    std::filesystem::rename(tmp, p, ec);
    if (ec) {
        LLOB_LOG_WARN("sidecar", "rename %s -> %s failed: %s",
                      tmp.c_str(), p.c_str(), ec.message().c_str());
        return false;
    }
    LLOB_LOG_INFO("sidecar", "wrote %s", p.c_str());
    return true;
}

}  // namespace llob
