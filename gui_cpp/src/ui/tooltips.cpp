// Rich hover-tooltips for arch-map cells.  Each helper:
//   1. Bails immediately if the item isn't hovered (with the standard
//      ~0.4s delay so casual mouse-over doesn't flash a popup).
//   2. Opens BeginTooltip with tightened padding so the popup feels
//      compact next to the cell it's annotating.
//   3. Pulls data via Model::* — the same hooks the panels use, so a
//      real backend lights up tooltips automatically.

#include "ui/tooltips.hpp"

#include "style.hpp"
#include "ui/fmt.hpp"
#include "ui/widgets.hpp"

#include <imgui.h>

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <string>

namespace llob {

namespace {

// All tooltips open via ImGui::BeginItemTooltip(), which is the canonical
// pattern (= IsItemHovered with the internal DelayNormal + tooltip-routing
// flags + BeginTooltip).  Earlier IsItemHovered + BeginTooltip flashed a
// yellow item-highlight border for one frame when the mouse moved between
// two adjacent items — the destination item briefly entered the "hovered
// but no tooltip yet" state before the new tooltip won the redirect.

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

// "[768 x 768]" from a shape vector
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

}  // namespace

// ── Component tooltip ─────────────────────────────────────────────────────

namespace {

// W_Q → kind 0 (Q), W_K → kind 1 (K), W_V → kind 2 (V).  Anything else
// returns -1 (no live activation hook applies).
int LiveKindFor(std::string_view comp) {
    if (comp == "W_Q") return 0;
    if (comp == "W_K") return 1;
    if (comp == "W_V") return 2;
    return -1;
}

const char* LiveLabelFor(int kind) {
    switch (kind) {
        case 0: return "Q vector";
        case 1: return "K vector";
        case 2: return "V vector";
    }
    return "live";
}

}  // namespace

void ComponentTooltip(Model& m, int layer, std::string_view comp_ref) {
    if (!ImGui::BeginItemTooltip()) return;

    const std::string tname = TensorNameFor(layer, comp_ref);
    const auto meta  = m.getTensorMeta(tname);
    const auto stats = m.getTensorStats(tname);
    const auto bins  = m.getWeightHistogram(tname, 32);

    Header("L%02d · %.*s", layer, int(comp_ref.size()), comp_ref.data());
    Muted("%s", tname.c_str());

    // ── WEIGHTS — the matrix's own data ───────────────────────────────
    ImGui::Separator();
    Muted("WEIGHTS");
    if (!meta.shape.empty()) {
        Muted("shape   %s", FmtShape(meta.shape).c_str());
        Muted("dtype   %s", meta.dtype.empty() ? "?" : meta.dtype.c_str());
        long long n_elem = 1;
        for (int d : meta.shape) n_elem *= d;
        if (n_elem > 0) Muted("params  %.2fM", double(n_elem) / 1e6);
    } else {
        Muted("// no tensor metadata yet");
    }
    if (!std::isnan(stats.frobenius)) Muted("‖·‖_F   %.3f", double(stats.frobenius));
    if (!std::isnan(stats.op_norm))   Muted("‖·‖_2   %.3f", double(stats.op_norm));
    if (stats.rank_eff != kNoInt)     Muted("rank    %d", stats.rank_eff);
    if (!bins.empty()) {
        Muted("distribution");
        ActivationHistogram(bins, 240.0f, 48.0f, Sty().info);
    }

    // ── LIVE — what this matrix produces at the current token ────────
    // Only meaningful for the QKV projection matrices (right now); other
    // components (norms, MLP gates, output projections) render the
    // section header with a "no live data" hint so the tooltip layout
    // stays predictable.
    const int kind = LiveKindFor(comp_ref);
    if (kind >= 0) {
        ImGui::Separator();
        Muted("LIVE  %s @ active token", LiveLabelFor(kind));
        // [DATA HOOK] Model::getActivation(layer, kind, n) — pulls the
        // d_head slice of the matrix's output for the current token.
        // 64 is a defensive cap; real backend should respect d_head.
        const auto vec = m.getActivation(layer, kind, 64);
        if (vec.empty()) {
            Muted("// no live activation data");
        } else {
            SparkOpts so{};
            so.color  = Sty().accent;
            so.fill   = true;
            so.width  = 240.0f;
            so.height = 36.0f;
            Sparkline(vec, so);
            // L2 norm + min/max under the spark for a single-glance read
            float lo = vec[0], hi = vec[0], sumsq = 0.0f;
            for (float v : vec) { lo = std::min(lo, v); hi = std::max(hi, v); sumsq += v * v; }
            Muted("min %+.3f   max %+.3f   ‖·‖₂ %.3f",
                  double(lo), double(hi), double(std::sqrt(sumsq)));
        }
    } else if (comp_ref == "W_O" || comp_ref == "W_out" || comp_ref == "W_in_gate" || comp_ref == "W_in_up") {
        ImGui::Separator();
        Muted("LIVE  output activation");
        Muted("// engine hook not wired yet — needs Model::getActivation kind for this component");
    }

    ImGui::EndTooltip();
}

// ── Head tooltip ──────────────────────────────────────────────────────────
namespace {
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

void HeadTooltip(Model& m, int layer, int head) {
    if (!ImGui::BeginItemTooltip()) return;

    const HeadBias bias = m.getHeadBias(layer, head);
    const float    norm = m.getHeadNorm(layer, head);
    const auto     pat  = m.getAttentionPattern(layer, head, 12, bias);
    const auto     hs   = m.getHeadStats(layer, head);

    Header("L%02d · h%d", layer, head);
    Muted("pattern  %s", BiasNameLong(bias));
    if (!std::isnan(norm)) Muted("‖head‖   %.3f", double(norm));

    if (!pat.empty()) {
        ImGui::Separator();
        // Reuse the existing thumbnail widget — same colour ramp as the
        // head browser so visual recall is direct.
        AttentionThumb(pat, 12, 96.0f, /*label=*/nullptr,
                       /*active=*/false, /*dim=*/false);
    }

    if (!hs.empty()) {
        ImGui::Separator();
        Muted("pattern statistics");
        for (const auto& row : hs) {
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
            ImGui::Text("%-18s", row.metric.c_str());
            ImGui::PopStyleColor();
            ImGui::SameLine(180.0f);
            ImGui::PushStyleColor(ImGuiCol_Text,
                row.tone && std::string(row.tone) == "warn" ? Sty().warn : Sty().text);
            ImGui::TextUnformatted(row.value_str.c_str());
            ImGui::PopStyleColor();
        }
    }

    ImGui::EndTooltip();
}

// ── Layer tooltip ─────────────────────────────────────────────────────────
void LayerTooltip(Model& m, int layer) {
    if (!ImGui::BeginItemTooltip()) return;

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
        Muted("dead neurons    %d / %d (%.1f%%)",
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

    ImGui::EndTooltip();
}

}  // namespace llob
