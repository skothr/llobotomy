// Raw tensors workspace — state_dict | tensor view | tensor stats | diff.
// HANDOFF §3.8 — does NOT respect showRaw; it IS the raw view.
//
// All values come from the Model interface; nothing in this file is
// hardcoded.  The data hooks each section needs are documented inline as
// `// [DATA HOOK]` comments naming the Model::* method that supplies them.

#include "workspaces/workspaces.hpp"

#include "appstate.hpp"
#include "model/model.hpp"
#include "style.hpp"
#include "ui/chrome.hpp"
#include "ui/fmt.hpp"
#include "ui/widgets.hpp"

#include <imgui.h>

#include <cstdio>
#include <span>
#include <string>
#include <vector>

namespace llob {

namespace {

void DrawStateDict(AppState& s, Model& m) {
    // [DATA HOOK] Model::getStateDict() — list of every tensor in the
    // checkpoint with name + dtype + shape + size.  Engine source:
    // safetensors header / pytorch_model.bin index.
    const auto entries = m.getStateDict();
    char flag[32]; std::snprintf(flag, sizeof flag, "%zu entries", entries.size());
    DrawTitleBar("state_dict", "≡", flag, "sd");
    if (!ImGui::BeginChild("##sd_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    static char filter[64] = {};
    ImGui::SetNextItemWidth(-1);
    ImGui::InputTextWithHint("##f", "filter… e.g. blocks.8.attn", filter, sizeof filter);
    if (entries.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
        ImGui::TextUnformatted("// no state_dict — model not loaded");
        ImGui::PopStyleColor();
        ImGui::EndChild();
        return;
    }
    for (const auto& t : entries) {
        const bool sel = (s.activeTensor == t.name);
        ImGui::PushStyleColor(ImGuiCol_Header, Sty().accent_bg_strong);
        ImGui::PushStyleColor(ImGuiCol_Text, sel ? Sty().accent : Sty().text);
        if (ImGui::Selectable(t.name.c_str(), sel)) s.setActiveTensor(t.name);
        ImGui::PopStyleColor(2);
        ImGui::SameLine(ImGui::GetContentRegionMax().x - 36);
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().info);
        ImGui::TextUnformatted(t.dtype.empty() ? "—" : t.dtype.c_str());
        ImGui::PopStyleColor();
    }
    ImGui::EndChild();
}

void DrawTensorView(const AppState& s, Model& m) {
    // [DATA HOOK] Model::getTensorMeta(name) — shape + dtype + size for the
    // selected tensor.  Drives the title bar flag string.
    const auto meta = m.getTensorMeta(s.activeTensor);
    char shape_str[64];
    if (meta.shape.size() == 2)
        std::snprintf(shape_str, sizeof shape_str, "%s [%d, %d] · %s",
                       meta.dtype.c_str(), meta.shape[0], meta.shape[1],
                       FmtSize(meta.size_bytes).c_str());
    else if (meta.shape.size() == 1)
        std::snprintf(shape_str, sizeof shape_str, "%s [%d] · %s",
                       meta.dtype.c_str(), meta.shape[0],
                       FmtSize(meta.size_bytes).c_str());
    else
        std::snprintf(shape_str, sizeof shape_str, "%s · %s",
                       meta.dtype.c_str(), FmtSize(meta.size_bytes).c_str());

    DrawTitleBar(s.activeTensor.c_str(), "0x", shape_str, "tensor-main");
    if (!ImGui::BeginChild("##tv_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    if (auto sec = BeginSection("Slice scrubber", false, "3D → 2D")) {
        static float ax0 = 0, ax1 = 0;
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
        ImGui::TextUnformatted("axis 0 [d_in]"); ImGui::PopStyleColor();
        ImSliderF("##ax0", ax0, 0,
                  meta.shape.size() > 0 ? float(meta.shape[0] - 1) : 1.0f,
                  "%.0f", 280);
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
        ImGui::TextUnformatted("axis 1 [head]"); ImGui::PopStyleColor();
        ImSliderF("##ax1", ax1, 0,
                  meta.shape.size() > 1 ? float(meta.shape[1] - 1) : 1.0f,
                  "%.0f", 280);
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
        ImGui::TextUnformatted("colormap"); ImGui::PopStyleColor();
        static int cm = 0;
        const char* cms[] = { "diverge", "heat", "grayscale", "viridis" };
        for (int i = 0; i < 4; ++i) { if (i) ImGui::SameLine(); if (ImGui::SmallButton(cms[i])) cm = i; }
        (void)cm;  // colormap selection is UI-state; rendering uses diverge for now.
    }
    if (auto sec = BeginSection("2D slice", true)) {
        // [DATA HOOK] Model::getTensorSlice2D(name, axis0, axis1, rows, cols)
        // — a (rows × cols) window into a higher-dim tensor; engine paging.
        const auto data = m.getTensorSlice2D(s.activeTensor, 0, 0, 32, 64);
        if (data.empty() || data[0].empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
            ImGui::TextUnformatted("// no slice data");
            ImGui::PopStyleColor();
        } else {
            HeatmapOpts ho{}; ho.width = ImGui::GetContentRegionAvail().x - 8;
            ho.height = 120; ho.minV = -1; ho.maxV = 1;
            TensorHeatmap(data, ho, [](float v) { return DivergeColor(v * 2.0f - 1.0f); });
        }
        // Stats footer comes from getTensorStats below — keep them aligned.
        const auto st = m.getTensorStats(s.activeTensor);
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
        ImGui::Text("min: %s   max: %s   μ: %s   σ: %s   ‖·‖₂: %s",
                     FmtFloat(st.min,       "%+.3f").c_str(),
                     FmtFloat(st.max,       "%+.3f").c_str(),
                     FmtFloat(st.mean,      "%+.3f").c_str(),
                     FmtFloat(st.std,       "%.3f").c_str(),
                     FmtFloat(st.frobenius, "%.2f").c_str());
        ImGui::PopStyleColor();
    }
    ImGui::EndChild();
}

void DrawTensorStats(const std::string& name, Model& m) {
    DrawTitleBar("tensor_stats", "∑", nullptr, "t-stats");
    if (!ImGui::BeginChild("##ts_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    if (auto sec = BeginSection("Shape & dtype", true)) {
        const auto meta = m.getTensorMeta(name);
        char shape[40] = "—";
        if (!meta.shape.empty()) {
            char* p = shape; *p++ = '[';
            for (std::size_t i = 0; i < meta.shape.size(); ++i)
                p += std::snprintf(p, sizeof shape - (p - shape),
                                    i + 1 == meta.shape.size() ? "%d" : "%d, ",
                                    meta.shape[i]);
            std::snprintf(p, sizeof shape - (p - shape), "]");
        }
        char stride[40] = "—";
        if (!meta.stride.empty()) {
            char* p = stride; *p++ = '(';
            for (std::size_t i = 0; i < meta.stride.size(); ++i)
                p += std::snprintf(p, sizeof stride - (p - stride),
                                    i + 1 == meta.stride.size() ? "%d" : "%d, ",
                                    meta.stride[i]);
            std::snprintf(p, sizeof stride - (p - stride), ")");
        }
        KV({
            { "shape",       shape,                                "accent" },
            { "dtype",       Or(meta.dtype),                        "" },
            { "stride",      stride,                                "" },
            { "contiguous",  meta.contiguous ? "yes" : "no",        meta.contiguous ? "good" : "warn" },
            { "device",      Or(meta.device),                       "" },
            { "size",        FmtSize(meta.size_bytes),              "" },
        });
    }
    if (auto sec = BeginSection("Singular values", false, "top 16")) {
        // [DATA HOOK] Model::getSingularValues(name, k) — top-k singular
        // values of the tensor (pre-computed during checkpoint indexing
        // since SVD on every frame is expensive).
        const auto sv = m.getSingularValues(name, 16);
        if (sv.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
            ImGui::TextUnformatted("// no SVD data");
            ImGui::PopStyleColor();
        } else {
            SparkOpts so{}; so.color = Sty().accent; so.fill = true;
            so.width = 250; so.height = 48;
            Sparkline(std::span{sv}, so);
        }
    }
    if (auto sec = BeginSection("Norms")) {
        // [DATA HOOK] Model::getTensorStats(name) — Frobenius / op /
        // infinity norms + effective rank.
        const auto st = m.getTensorStats(name);
        char rank[24];
        if (st.rank_eff == kNoInt || st.rank_full == kNoInt)
            std::snprintf(rank, sizeof rank, "—");
        else
            std::snprintf(rank, sizeof rank, "%d / %d", st.rank_eff, st.rank_full);
        KV({
            { "‖·‖_F",      FmtFloat(st.frobenius, "%.2e"), "" },
            { "‖·‖_2 (op)", FmtFloat(st.op_norm,   "%.2f"), "" },
            { "‖·‖_∞",      FmtFloat(st.inf_norm,  "%.3f"), "" },
            { "rank (eff)", rank,                            "" },
        });
    }
    ImGui::EndChild();
}

void DrawDiffView(const std::string& name, Model& m) {
    DrawTitleBar("diff_view", "≠", nullptr, "diff-view");
    if (!ImGui::BeginChild("##dv_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    if (auto sec = BeginSection("ΔW vs base checkpoint", true)) {
        // [DATA HOOK] Model::getDiffStats(name) — Frobenius norm of the
        // delta + cosine vs the base checkpoint + top singular values of
        // the delta.
        const auto ds = m.getDiffStats(name);
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted); ImGui::TextUnformatted("‖ΔW‖_F"); ImGui::PopStyleColor();
        ImGui::SameLine(120);
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted); ImGui::TextUnformatted("cos(W, W₀)"); ImGui::PopStyleColor();
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().warn);
        ImGui::TextUnformatted(FmtFloat(ds.frobenius_norm, "%.3f").c_str()); ImGui::PopStyleColor();
        ImGui::SameLine(120);
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().good);
        ImGui::TextUnformatted(FmtFloat(ds.cosine, "%.4f").c_str()); ImGui::PopStyleColor();

        // [DATA HOOK] Model::getDiffSlice2D(name, rows, cols) — 2D delta
        // for visual inspection.  Diverging colormap centered on zero.
        const auto data = m.getDiffSlice2D(name, 16, 64);
        if (data.empty() || data[0].empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
            ImGui::TextUnformatted("// no delta — base checkpoint not set");
            ImGui::PopStyleColor();
        } else {
            HeatmapOpts ho{}; ho.width = ImGui::GetContentRegionAvail().x - 8;
            ho.height = 70; ho.minV = -0.2f; ho.maxV = 0.2f;
            TensorHeatmap(data, ho, [](float v) { return DivergeColor(v * 6.0f - 1.0f); });
        }
    }
    ImGui::EndChild();
}

}  // namespace

void DrawRawTensorsWorkspace(AppState& s, Model& m) {
    const float W = ImGui::GetContentRegionAvail().x, H = ImGui::GetContentRegionAvail().y;
    const float gap = 1.0f;
    const float lw = 320.0f, rw = 280.0f;
    const float cw = std::max(200.0f, W - lw - rw - 2 * gap);
    const float bot_h = std::min(220.0f, H * 0.30f);
    const float top_h = H - bot_h - gap;

    ImGui::BeginChild("##rt_sd", { lw, H }, ImGuiChildFlags_Borders);
    DrawStateDict(s, m); ImGui::EndChild(); ImGui::SameLine(0, gap);

    ImGui::BeginChild("##rt_center", { cw, H });
    ImGui::BeginChild("##rt_view", { cw, top_h }, ImGuiChildFlags_Borders);
    DrawTensorView(s, m); ImGui::EndChild();
    ImGui::BeginChild("##rt_diff", { cw, bot_h }, ImGuiChildFlags_Borders);
    DrawDiffView(s.activeTensor, m); ImGui::EndChild();
    ImGui::EndChild(); ImGui::SameLine(0, gap);

    ImGui::BeginChild("##rt_stats", { rw, H }, ImGuiChildFlags_Borders);
    DrawTensorStats(s.activeTensor, m); ImGui::EndChild();
}

}  // namespace llob
