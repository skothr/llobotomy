// Rich hover-tooltips for arch-map cells, with middle-click → pin so the
// content survives as a movable panel for sub-element inspection.
//
// Architecture:
//   - RenderXxxBody(m, ...) — pure rendering, no item-state queries.
//     Used by both BeginItemTooltip (transient hover) and Begin window
//     (persistent pinned panel).
//   - XxxTooltip wraps the body in BeginItemTooltip + middle-click pin.
//   - PinnedRegistry is a process-wide singleton; render queue is
//     drained once per frame from main.cpp via DrawPinnedPanels.
//
// Visual / data discipline:
//   - No "fill" on sparklines.  The line itself is the data;
//     a filled area underneath suggests an integral that doesn't exist.
//   - Every numeric value comes from a Model::* hook, plotted as-is.
//     No interpolation, no smoothing, no stylised projection.

#include "ui/tooltips.hpp"

#include "logger.hpp"
#include "style.hpp"
#include "ui/fmt.hpp"
#include "ui/widgets.hpp"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <string>
#include <vector>

namespace llob {

// ── Internal helpers ──────────────────────────────────────────────────────

namespace {

// Map a comp_ref like "W_Q" / "W_in_gate" / "norm1" to a canonical engine
// tensor name `blocks.<L>.<area>.<comp>.weight`.  When a backend uses a
// different naming scheme, override at the boundary by changing this
// helper rather than the call sites.
std::string TensorNameFor(int layer, std::string_view comp) {
    char buf[80];
    if      (comp == "norm1")     std::snprintf(buf, sizeof buf, "blocks.%d.norm1.weight",       layer);
    else if (comp == "norm2")     std::snprintf(buf, sizeof buf, "blocks.%d.norm2.weight",       layer);
    else if (comp == "W_Q" || comp == "W_K" || comp == "W_V" || comp == "W_O")
                                  std::snprintf(buf, sizeof buf, "blocks.%d.attn.%.*s.weight",
                                                layer, int(comp.size()), comp.data());
    else if (comp == "W_in_gate") std::snprintf(buf, sizeof buf, "blocks.%d.mlp.W_in_gate.weight",  layer);
    else if (comp == "W_in_up")   std::snprintf(buf, sizeof buf, "blocks.%d.mlp.W_in_up.weight",    layer);
    else if (comp == "W_out")     std::snprintf(buf, sizeof buf, "blocks.%d.mlp.W_out.weight",      layer);
    else                          std::snprintf(buf, sizeof buf, "blocks.%d.%.*s.weight",
                                                layer, int(comp.size()), comp.data());
    return buf;
}

std::string FmtShape(const std::vector<int>& shape) {
    std::string out = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) out += " x ";
        out += std::to_string(shape[i]);
    }
    out += "]";
    return out;
}

void Header(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    char buf[128]; std::vsnprintf(buf, sizeof buf, fmt, ap); va_end(ap);
    ImGui::PushStyleColor(ImGuiCol_Text, Sty().accent);
    ImGui::TextUnformatted(buf);
    ImGui::PopStyleColor();
}

void Muted(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    char buf[128]; std::vsnprintf(buf, sizeof buf, fmt, ap); va_end(ap);
    ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
    ImGui::TextUnformatted(buf);
    ImGui::PopStyleColor();
}

int LiveKindFor(std::string_view comp) {
    if (comp == "W_Q") return 0;
    if (comp == "W_K") return 1;
    if (comp == "W_V") return 2;
    return -1;
}

const char* LiveLabelFor(int kind) {
    switch (kind) {
        case 0: return "Q";
        case 1: return "K";
        case 2: return "V";
    }
    return "";
}

const char* BiasNameLong(HeadBias b) {
    switch (b) {
        case HeadBias::Diag:      return "diagonal (self-attention)";
        case HeadBias::Prev:      return "previous-token";
        case HeadBias::First:     return "first-token / BOS";
        case HeadBias::Broad:     return "broad / uniform";
        case HeadBias::Induction: return "induction";
    }
    return "?";
}

}  // namespace

// ── Render bodies (shared by tooltip + pinned panel) ─────────────────────

namespace {

void RenderComponentBody(Model& m, int layer, std::string_view comp_ref) {
    const std::string tname = TensorNameFor(layer, comp_ref);
    const auto meta  = m.getTensorMeta(tname);
    const auto stats = m.getTensorStats(tname);
    const auto bins  = m.getWeightHistogram(tname, 32);

    Header("L%02d · %.*s", layer, int(comp_ref.size()), comp_ref.data());
    Muted("%s", tname.c_str());

    // ── WEIGHTS — the matrix's own data ───────────────────────────
    ImGui::Separator();
    Muted("WEIGHTS");
    if (!meta.shape.empty()) {
        Muted("shape   %s", FmtShape(meta.shape).c_str());
        Muted("dtype   %s", meta.dtype.empty() ? "?" : meta.dtype.c_str());
        long long n_elem = 1;
        for (int d : meta.shape) n_elem *= d;
        if (n_elem > 0) Muted("params  %.2fM", double(n_elem) / 1e6);
    } else {
        Muted("// no tensor metadata");
    }
    if (!std::isnan(stats.frobenius)) Muted("‖·‖_F   %.3f", double(stats.frobenius));
    if (!std::isnan(stats.op_norm))   Muted("‖·‖_2   %.3f", double(stats.op_norm));
    if (stats.rank_eff != kNoInt)     Muted("rank    %d", stats.rank_eff);
    if (!bins.empty()) {
        Muted("value distribution (%d bins)", int(bins.size()));
        ActivationHistogram(bins, 260.0f, 48.0f, Sty().info);
    }

    // ── LIVE — what this matrix produces at the current token ────
    const int kind = LiveKindFor(comp_ref);
    if (kind >= 0) {
        ImGui::Separator();
        Muted("LIVE  %s vector @ active token (d_head=64)", LiveLabelFor(kind));
        // [DATA HOOK] Model::getActivation(layer, kind, n) — d_head
        // slice of the matrix's output at the current token.
        const auto vec = m.getActivation(layer, kind, 64);
        if (vec.empty()) {
            Muted("// no live activation data");
        } else {
            // No fill — the line is the data, a filled area would suggest
            // an integral / area-under-curve that has no meaning here.
            float lo = vec[0], hi = vec[0], sumsq = 0.0f;
            for (float v : vec) { lo = std::min(lo, v); hi = std::max(hi, v); sumsq += v * v; }
            SparkOpts so{};
            so.color    = Sty().accent;
            so.fill     = false;
            so.baseline = true;        // explicit y=0 reference
            so.width    = 260.0f;
            so.height   = 48.0f;
            so.min      = lo;
            so.max      = hi;
            Sparkline(vec, so);
            Muted("min %+.4f  max %+.4f  ‖·‖₂ %.4f  n=%zu",
                  double(lo), double(hi), double(std::sqrt(sumsq)), vec.size());
        }
    } else if (comp_ref == "W_O" || comp_ref == "W_out" ||
               comp_ref == "W_in_gate" || comp_ref == "W_in_up") {
        ImGui::Separator();
        Muted("LIVE  output activation");
        Muted("// hook needed: Model::getActivation kind for %.*s",
              int(comp_ref.size()), comp_ref.data());
    }
}

void RenderHeadBody(Model& m, int layer, int head) {
    const HeadBias bias = m.getHeadBias(layer, head);
    const float    norm = m.getHeadNorm(layer, head);
    const auto     pat  = m.getAttentionPattern(layer, head, 12, bias);
    const auto     hs   = m.getHeadStats(layer, head);

    Header("L%02d · h%d", layer, head);
    Muted("pattern  %s", BiasNameLong(bias));
    if (!std::isnan(norm)) Muted("‖head‖   %.3f", double(norm));

    if (!pat.empty()) {
        ImGui::Separator();
        Muted("attention pattern  (12-token downsample)");
        AttentionThumb(pat, 12, 110.0f, /*label=*/nullptr,
                       /*active=*/false, /*dim=*/false);
    }

    if (!hs.empty()) {
        ImGui::Separator();
        Muted("pattern statistics");
        for (const auto& row : hs) {
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
            ImGui::Text("%-20s", row.metric.c_str());
            ImGui::PopStyleColor();
            ImGui::SameLine(190.0f);
            ImGui::PushStyleColor(ImGuiCol_Text,
                row.tone && std::string(row.tone) == "warn" ? Sty().warn : Sty().text);
            ImGui::TextUnformatted(row.value_str.c_str());
            ImGui::PopStyleColor();
        }
    }
}

void RenderLayerBody(Model& m, int layer) {
    const auto live = m.getLiveActivations(layer);
    const auto rsum = m.getResidualSummary(layer);

    Header("block_%02d", layer);

    ImGui::Separator();
    Muted("live activations");
    Muted("attn_out_norm   %s", FmtFloat(live.attn_out_norm,   "%.3f").c_str());
    Muted("mlp_out_norm    %s", FmtFloat(live.mlp_out_norm,    "%.3f").c_str());
    Muted("resid_post_norm %s", FmtFloat(live.resid_post_norm, "%.3f").c_str());
    Muted("attn_entropy    %s nats", FmtFloat(live.attn_entropy_avg, "%.2f").c_str());
    if (live.dead_neurons != kNoInt && live.total_neurons != kNoInt && live.total_neurons > 0) {
        Muted("dead neurons    %d / %d  (%.1f%%)",
              live.dead_neurons, live.total_neurons,
              100.0 * double(live.dead_neurons) / double(live.total_neurons));
    }

    ImGui::Separator();
    Muted("residual summary");
    Muted("cos(L-1)        %s", FmtFloat(rsum.cos_prev, "%.4f").c_str());
    Muted("kurtosis        %s", FmtFloat(rsum.kurtosis, "%.2f").c_str());
    if (rsum.rank_eff != kNoInt && rsum.rank_full != kNoInt) {
        Muted("rank_eff        %d / %d", rsum.rank_eff, rsum.rank_full);
    }
}

}  // namespace

// ── Pin registry ──────────────────────────────────────────────────────────

namespace {

enum class PinKind : int { Component, Head, Layer };

struct Pin {
    int          id;
    PinKind      kind;
    int          layer;
    int          head;        // valid when Head
    std::string  comp_ref;    // valid when Component
    ImVec2       pos;
    bool         alive = true;
};

struct PinRegistry {
    std::vector<Pin> pins;
    int next_id = 1;
};

PinRegistry& Pins() { static PinRegistry r; return r; }

void TryPin(PinKind kind, int layer, int head, std::string_view comp_ref) {
    // Middle-click on the just-submitted item pins.  Detect via item
    // hover + middle-mouse press; ImGui doesn't fire IsItemClicked for
    // middle by default.
    if (!ImGui::IsItemHovered()) return;
    if (!ImGui::IsMouseClicked(ImGuiMouseButton_Middle)) return;

    Pin p;
    p.id       = Pins().next_id++;
    p.kind     = kind;
    p.layer    = layer;
    p.head     = head;
    p.comp_ref = std::string(comp_ref);
    p.pos      = ImGui::GetMousePos();
    Pins().pins.push_back(std::move(p));
    LLOB_LOG_DEBUG("pin", "pinned panel #%d (kind=%d, L=%d, h=%d, comp=%.*s)",
                   Pins().pins.back().id, int(kind), layer, head,
                   int(comp_ref.size()), comp_ref.data());
}

}  // namespace

// ── Tooltip helpers (transient hover) ─────────────────────────────────────

void ComponentTooltip(Model& m, int layer, std::string_view comp_ref) {
    if (ImGui::BeginItemTooltip()) {
        RenderComponentBody(m, layer, comp_ref);
        ImGui::EndTooltip();
    }
    TryPin(PinKind::Component, layer, -1, comp_ref);
}

void HeadTooltip(Model& m, int layer, int head) {
    if (ImGui::BeginItemTooltip()) {
        RenderHeadBody(m, layer, head);
        ImGui::EndTooltip();
    }
    TryPin(PinKind::Head, layer, head, {});
}

void LayerTooltip(Model& m, int layer) {
    if (ImGui::BeginItemTooltip()) {
        RenderLayerBody(m, layer);
        ImGui::EndTooltip();
    }
    TryPin(PinKind::Layer, layer, -1, {});
}

// ── Pinned-panel rendering ────────────────────────────────────────────────

void DrawPinnedPanels(Model& m) {
    auto& pins = Pins().pins;
    for (auto& p : pins) {
        if (!p.alive) continue;

        char wid[64];
        std::snprintf(wid, sizeof wid, "##pin_%d", p.id);
        char title[80];
        switch (p.kind) {
            case PinKind::Component: std::snprintf(title, sizeof title, "%s @ L%02d", p.comp_ref.c_str(), p.layer); break;
            case PinKind::Head:      std::snprintf(title, sizeof title, "h%d @ L%02d", p.head, p.layer); break;
            case PinKind::Layer:     std::snprintf(title, sizeof title, "block_%02d", p.layer); break;
        }
        ImGui::SetNextWindowPos(p.pos, ImGuiCond_Once);
        ImGui::SetNextWindowSize(ImVec2(320, 0), ImGuiCond_Once);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, Sty().bg_panel);
        ImGui::PushStyleColor(ImGuiCol_Border,   Sty().accent);
        ImGui::PushStyleVar  (ImGuiStyleVar_WindowBorderSize, 1.5f);

        bool open = true;
        if (ImGui::Begin(wid, &open,
                         ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoCollapse |
                         ImGuiWindowFlags_AlwaysAutoResize)) {
            // Custom titlebar — title + the same dock-id-style hint
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().accent);
            ImGui::TextUnformatted(title);
            ImGui::PopStyleColor();
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_dim);
            ImGui::Text("##pin_%d", p.id);
            ImGui::PopStyleColor();
            ImGui::Separator();

            switch (p.kind) {
                case PinKind::Component: RenderComponentBody(m, p.layer, p.comp_ref); break;
                case PinKind::Head:      RenderHeadBody     (m, p.layer, p.head);     break;
                case PinKind::Layer:     RenderLayerBody    (m, p.layer);             break;
            }
        }
        ImGui::End();
        ImGui::PopStyleVar();
        ImGui::PopStyleColor(2);

        if (!open) p.alive = false;
    }
    // Compact closed pins
    pins.erase(std::remove_if(pins.begin(), pins.end(),
                              [](const Pin& p){ return !p.alive; }),
               pins.end());
}

}  // namespace llob
