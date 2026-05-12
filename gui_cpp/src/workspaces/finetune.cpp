// Finetune workspace — LoRA config | freeze grid + delta heatmap | eval diff.
// HANDOFF §3.6.

#include "workspaces/workspaces.hpp"

#include "appstate.hpp"
#include "model/model.hpp"
#include "style.hpp"
#include "ui/chrome.hpp"
#include "ui/widgets.hpp"

#include <imgui.h>

#include <cmath>
#include <cstdio>
#include <unordered_set>
#include <vector>

namespace llob {

namespace {

void DrawLoRAConfig() {
    DrawTitleBar("lora_config", "◆", nullptr, "lora");
    if (!ImGui::BeginChild("##lc_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    if (auto sec = BeginSection("LoRA · adapter", true)) {
        KV({
            { "rank r",          "16",                       "accent" },
            { "α",               "32",                       "" },
            { "dropout",         "0.05",                     "" },
            { "target",          "q_proj, v_proj, o_proj",   "info" },
            { "trainable params","4.72M (0.6%)",             "good" },
            { "method",          "LoRA + RSLoRA scale",      "" },
        });
        EndSection(sec);
    }
    if (auto sec = BeginSection("Optimizer", true)) {
        KV({
            { "name",          "AdamW8bit",   "" },
            { "lr",            "2.0e-4",      "accent" },
            { "β1, β2",        "0.9, 0.95",   "" },
            { "warmup",        "100 step",    "" },
            { "schedule",      "cosine",      "" },
            { "weight_decay",  "0.01",        "" },
        });
        EndSection(sec);
    }
    if (auto sec = BeginSection("Data", true)) {
        KV({
            { "dataset", "sft/instruction_v3", "" },
            { "n_train", "128k",               "accent" },
            { "n_eval",  "2.0k",               "" },
            { "ctx_len", "4096",               "" },
            { "packing", "on",                 "good" },
        });
        EndSection(sec);
    }
    ImGui::EndChild();
}

void DrawLayerFreeze(const ModelInfo& mi, std::unordered_set<int>& frozen) {
    DrawTitleBar("layer_freeze", "❄", "click to toggle", "freeze");
    if (!ImGui::BeginChild("##lf_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
    ImGui::TextUnformatted("// ❄ frozen   ◉ trainable (LoRA)   ⚠ trainable (full)");
    ImGui::PopStyleColor();

    const int per_row = 6;
    const float w = (ImGui::GetContentRegionAvail().x - (per_row - 1) * 4) / float(per_row);
    for (int L = 0; L < mi.nLayers; ++L) {
        ImGui::PushID(L);
        const bool fr = frozen.contains(L);
        ImGui::PushStyleColor(ImGuiCol_Button,        fr ? Sty().bg_input : Sty().accent_bg);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, fr ? Sty().bg_input_hover : Sty().accent_bg_strong);
        ImGui::PushStyleColor(ImGuiCol_Text,          fr ? Sty().text_muted : Sty().accent);
        char lbl[32]; std::snprintf(lbl, sizeof lbl, "L%02d  %s", L, fr ? "❄" : "◉");
        if (ImGui::Button(lbl, ImVec2(w, 36))) {
            if (fr) frozen.erase(L); else frozen.insert(L);
        }
        ImGui::PopStyleColor(3);
        ImGui::PopID();
        if ((L + 1) % per_row != 0) ImGui::SameLine();
    }
    ImGui::NewLine();

    char fb[32]; std::snprintf(fb, sizeof fb, "%zu / %d layers", frozen.size(), mi.nLayers);
    char tb[32]; std::snprintf(tb, sizeof tb, "%zu layers", std::size_t(mi.nLayers) - frozen.size());
    KV({
        { "frozen",       fb,             "info" },
        { "trainable",    tb,             "accent" },
        { "Δ params",     "4.72M",        "good" },
        { "GPU mem est.", "38.4 GB / 80 GB","good" },
    }, true);
    ImGui::EndChild();
}

void DrawEvalDiff() {
    DrawTitleBar("eval_diff", "≠", nullptr, "evaldiff");
    if (!ImGui::BeginChild("##ed_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    if (auto sec = BeginSection("base vs ft · MMLU", true)) {
        KV({ {"base","0.412",""}, {"ft (step 4k)","0.448","good"}, {"Δ","+0.036","good"} });
        EndSection(sec);
    }
    if (auto sec = BeginSection("loss curves", true)) {
        const float d[] = {2.81f,2.62f,2.44f,2.31f,2.18f,2.06f,1.98f,1.91f,1.86f,1.82f,1.79f,1.78f};
        SparkOpts so{}; so.color = Sty().accent; so.fill = true; so.width = 280; so.height = 48;
        Sparkline(d, so);
        EndSection(sec);
    }
    if (auto sec = BeginSection("A/B sample", true)) {
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
        ImGui::TextUnformatted("PROMPT"); ImGui::PopStyleColor();
        ImGui::TextUnformatted("\"Explain the residual stream\"");
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_dim);
        ImGui::TextUnformatted("BASE →"); ImGui::PopStyleColor();
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
        ImGui::TextUnformatted("\"The residual stream is a connection that...\"");
        ImGui::PopStyleColor();
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().accent);
        ImGui::TextUnformatted("FT →"); ImGui::PopStyleColor();
        ImGui::TextUnformatted("\"The residual stream is the additive backbone...\"");
        EndSection(sec);
    }
    ImGui::EndChild();
}

void DrawDiffRun(const ModelInfo& mi, const std::unordered_set<int>& frozen) {
    DrawTitleBar("diff_run", "⇄", nullptr, "diff");
    if (!ImGui::BeginChild("##dr_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    if (auto sec = BeginSection("parameter delta heatmap · ‖ΔW‖ per layer × component", true)) {
        const int rows = mi.nLayers, cols = 12;
        std::vector<std::vector<float>> data(rows, std::vector<float>(cols));
        for (int L = 0; L < rows; ++L) for (int c = 0; c < cols; ++c) {
            data[L][c] = frozen.contains(L) ? 0.0f
                       : std::abs(std::sin(L * 0.7f + c * 1.3f)) * (1.0f - L * 0.02f);
        }
        HeatmapOpts ho{}; ho.width = ImGui::GetContentRegionAvail().x - 8; ho.height = 120;
        TensorHeatmap(data, ho);
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
        ImGui::TextUnformatted("Q  K  V  O  g_in  g_up  g_out  dn  rn1  rn2  b1  b2");
        ImGui::PopStyleColor();
        EndSection(sec);
    }
    ImGui::EndChild();
}

}  // namespace

void DrawFineTuneWorkspace(AppState& s, Model& /*m*/) {
    static std::unordered_set<int> frozen{0, 1, 2, 3};

    const float W = ImGui::GetContentRegionAvail().x, H = ImGui::GetContentRegionAvail().y;
    const float gap = 1.0f;
    const float lw = 300.0f, rw = 320.0f;
    const float cw = std::max(200.0f, W - lw - rw - 2 * gap);
    const float bot_h = std::min(220.0f, H * 0.32f);
    const float top_h = H - bot_h - gap;

    ImGui::BeginChild("##ft_left", { lw, H }, ImGuiChildFlags_Borders);
    DrawLoRAConfig(); ImGui::EndChild(); ImGui::SameLine(0, gap);

    ImGui::BeginChild("##ft_center", { cw, H });
    ImGui::BeginChild("##ft_freeze", { cw, top_h }, ImGuiChildFlags_Borders);
    DrawLayerFreeze(s.model, frozen); ImGui::EndChild();
    ImGui::BeginChild("##ft_diff",   { cw, bot_h }, ImGuiChildFlags_Borders);
    DrawDiffRun(s.model, frozen); ImGui::EndChild();
    ImGui::EndChild(); ImGui::SameLine(0, gap);

    ImGui::BeginChild("##ft_right", { rw, H }, ImGuiChildFlags_Borders);
    DrawEvalDiff(); ImGui::EndChild();
}

}  // namespace llob
