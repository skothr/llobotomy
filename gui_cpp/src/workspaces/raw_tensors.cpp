// Raw tensors workspace — state_dict | tensor view | tensor stats | diff.
// HANDOFF §3.8 — does NOT respect showRaw; it IS the raw view.

#include "workspaces/workspaces.hpp"

#include "appstate.hpp"
#include "model/model.hpp"
#include "style.hpp"
#include "ui/chrome.hpp"
#include "ui/widgets.hpp"

#include <imgui.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

namespace llob {

namespace {

const std::array<const char*, 12>& TensorList() {
    static const std::array<const char*, 12> T = {
        "embed.weight", "pos_embed.freqs",
        "blocks.8.attn.W_Q.weight", "blocks.8.attn.W_K.weight",
        "blocks.8.attn.W_V.weight", "blocks.8.attn.W_O.weight",
        "blocks.8.attn.b_Q",
        "blocks.8.mlp.W_in.weight", "blocks.8.mlp.W_out.weight",
        "blocks.8.norm1.weight",
        "final_norm.weight", "unembed.weight",
    };
    return T;
}

void DrawStateDict(AppState& s) {
    DrawTitleBar("state_dict", "≡", "402 entries", "sd");
    if (!ImGui::BeginChild("##sd_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    static char filter[64] = {};
    ImGui::SetNextItemWidth(-1);
    ImGui::InputTextWithHint("##f", "filter… e.g. blocks.8.attn", filter, sizeof filter);
    for (auto* name : TensorList()) {
        const bool sel = (s.activeTensor == name);
        ImGui::PushStyleColor(ImGuiCol_Header, Sty().accent_bg_strong);
        ImGui::PushStyleColor(ImGuiCol_Text, sel ? Sty().accent : Sty().text);
        if (ImGui::Selectable(name, sel)) s.setActiveTensor(name);
        ImGui::PopStyleColor(2);
        ImGui::SameLine(ImGui::GetContentRegionMax().x - 36);
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().info);
        ImGui::TextUnformatted("fp16"); ImGui::PopStyleColor();
    }
    ImGui::EndChild();
}

void DrawTensorView(const AppState& s) {
    char shape[64];
    std::snprintf(shape, sizeof shape, "fp16 [%d, %d] · 1.18 MB", s.model.dModel, s.model.dModel);
    DrawTitleBar(s.activeTensor.c_str(), "0x", shape, "tensor-main");
    if (!ImGui::BeginChild("##tv_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    if (auto sec = BeginSection("Slice scrubber", false, "3D → 2D")) {
        static float ax0 = 384, ax1 = 3;
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
        ImGui::TextUnformatted("axis 0 [d_in]"); ImGui::PopStyleColor();
        ImSliderF("##ax0", ax0, 0, float(s.model.dModel * 2), "%.0f", 280);
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
        ImGui::TextUnformatted("axis 1 [head]"); ImGui::PopStyleColor();
        ImSliderF("##ax1", ax1, 0, float(s.model.nHeads - 1), "%.0f", 280);
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
        ImGui::TextUnformatted("colormap"); ImGui::PopStyleColor();
        static int cm = 0;
        const char* cms[] = { "diverge", "heat", "grayscale", "viridis" };
        for (int i = 0; i < 4; ++i) { if (i) ImGui::SameLine(); if (ImGui::SmallButton(cms[i])) cm = i; }
        EndSection(sec);
    }
    if (auto sec = BeginSection("2D slice", true, "[768 × 64]")) {
        std::vector<std::vector<float>> data(32, std::vector<float>(64));
        for (int i = 0; i < 32; ++i) for (int j = 0; j < 64; ++j)
            data[i][j] = std::sin(i * 0.4f + j * 0.3f) * std::cos(i * 0.2f - j * 0.15f);
        HeatmapOpts ho{}; ho.width = ImGui::GetContentRegionAvail().x - 8; ho.height = 120;
        ho.minV = -1; ho.maxV = 1;
        TensorHeatmap(data, ho, [](float v) { return DivergeColor(v * 2.0f - 1.0f); });
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
        ImGui::TextUnformatted("min: -0.412   max: +0.408   μ: +0.001   σ: 0.142   ‖·‖₂: 27.84");
        ImGui::PopStyleColor();
        EndSection(sec);
    }
    ImGui::EndChild();
}

void DrawTensorStats(const ModelInfo& mi) {
    DrawTitleBar("tensor_stats", "∑", nullptr, "t-stats");
    if (!ImGui::BeginChild("##ts_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    char shape[32]; std::snprintf(shape, sizeof shape, "[%d, %d]", mi.dModel, mi.dModel);
    if (auto sec = BeginSection("Shape & dtype", true)) {
        KV({
            { "shape",       shape,        "accent" },
            { "dtype",       "fp16",       "" },
            { "stride",      "(768, 1)",   "" },
            { "contiguous",  "yes",        "good" },
            { "device",      "cuda:0",     "" },
            { "size",        "1,179,648 B","" },
        });
        EndSection(sec);
    }
    if (auto sec = BeginSection("Singular values", false, "top 16")) {
        const float d[] = {24,18,14,11,9,7,6,5,4.2f,3.6f,3.1f,2.7f,2.4f,2.1f,1.9f,1.7f};
        SparkOpts so{}; so.color = Sty().accent; so.fill = true; so.width = 250; so.height = 48;
        Sparkline(d, so);
        EndSection(sec);
    }
    if (auto sec = BeginSection("Norms")) {
        KV({
            { "‖·‖_F",     "9.41e2",    "" },
            { "‖·‖_2 (op)","24.82",     "" },
            { "‖·‖_∞",     "0.412",     "" },
            { "rank (eff)","512 / 768", "" },
        });
        EndSection(sec);
    }
    ImGui::EndChild();
}

void DrawDiffView() {
    DrawTitleBar("diff_view", "≠", nullptr, "diff-view");
    if (!ImGui::BeginChild("##dv_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    if (auto sec = BeginSection("ΔW vs base checkpoint", true)) {
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted); ImGui::TextUnformatted("‖ΔW‖_F"); ImGui::PopStyleColor();
        ImGui::SameLine(120);
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted); ImGui::TextUnformatted("cos(W, W₀)"); ImGui::PopStyleColor();
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().warn);  ImGui::TextUnformatted("0.084"); ImGui::PopStyleColor();
        ImGui::SameLine(120);
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().good);  ImGui::TextUnformatted("0.9912"); ImGui::PopStyleColor();
        std::vector<std::vector<float>> data(16, std::vector<float>(64));
        for (int i = 0; i < 16; ++i) for (int j = 0; j < 64; ++j)
            data[i][j] = std::sin(i * 0.5f + j * 0.4f) * 0.1f * std::exp(-std::abs(i - 8.0f) * 0.2f);
        HeatmapOpts ho{}; ho.width = ImGui::GetContentRegionAvail().x - 8; ho.height = 70; ho.minV = -0.2f; ho.maxV = 0.2f;
        TensorHeatmap(data, ho, [](float v) { return DivergeColor(v * 6.0f - 1.0f); });
        EndSection(sec);
    }
    ImGui::EndChild();
}

}  // namespace

void DrawRawTensorsWorkspace(AppState& s, Model& /*m*/) {
    const float W = ImGui::GetContentRegionAvail().x, H = ImGui::GetContentRegionAvail().y;
    const float gap = 1.0f;
    const float lw = 320.0f, rw = 280.0f;
    const float cw = std::max(200.0f, W - lw - rw - 2 * gap);
    const float bot_h = std::min(220.0f, H * 0.30f);
    const float top_h = H - bot_h - gap;

    ImGui::BeginChild("##rt_sd", { lw, H }, ImGuiChildFlags_Borders);
    DrawStateDict(s); ImGui::EndChild(); ImGui::SameLine(0, gap);

    ImGui::BeginChild("##rt_center", { cw, H });
    ImGui::BeginChild("##rt_view", { cw, top_h }, ImGuiChildFlags_Borders);
    DrawTensorView(s); ImGui::EndChild();
    ImGui::BeginChild("##rt_diff", { cw, bot_h }, ImGuiChildFlags_Borders);
    DrawDiffView(); ImGui::EndChild();
    ImGui::EndChild(); ImGui::SameLine(0, gap);

    ImGui::BeginChild("##rt_stats", { rw, H }, ImGuiChildFlags_Borders);
    DrawTensorStats(s.model); ImGui::EndChild();
}

}  // namespace llob
