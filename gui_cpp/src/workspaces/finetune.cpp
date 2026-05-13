// Finetune workspace — LoRA config | freeze grid + delta heatmap | eval diff.
// HANDOFF §3.6.
//
// All values come from the Model interface; nothing in this file is
// hardcoded.  The data hooks each section needs are documented inline as
// `// [DATA HOOK]` comments naming the Model::* method that supplies them.

#include "workspaces/workspaces.hpp"

#include "appstate.hpp"
#include "llm_engine/model.hpp"
#include "style.hpp"
#include "ui/chrome.hpp"
#include "ui/fmt.hpp"
#include "ui/widgets.hpp"

#include <imgui.h>
#include <imgui_internal.h>           // DockBuilder*

#include <cstdio>
#include <string>
#include <unordered_set>
#include <vector>

namespace llob {

namespace {

void DrawLoRAConfig(Model& m) {
    DrawTitleBar("lora_config", "◆", nullptr, "lora");
    if (!ImGui::BeginChild("##lc_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }

    // [DATA HOOK] Model::getLoRAConfig() — adapter config.  Engine source:
    // PEFT-style LoraConfig dataclass attached to the active adapter.
    if (auto sec = BeginSection("LoRA · adapter", true)) {
        const auto c = m.getLoRAConfig();
        char tp[40]; std::snprintf(tp, sizeof tp, "%s  (%.1f%%)",
                                    FmtFloat(c.trainable_M, "%.2fM").c_str(),
                                    double(c.trainable_pct * 100.0));
        const std::string tp_s = std::isnan(c.trainable_M) ? "—" : tp;
        KV({
            { "rank r",          FmtInt(c.rank),                "accent" },
            { "α",               FmtFloat(c.alpha,    "%.0f"),  "" },
            { "dropout",         FmtFloat(c.dropout,  "%.2f"),  "" },
            { "target",          Or(c.target),                  "info" },
            { "trainable params",tp_s,                          "good" },
            { "method",          Or(c.method),                  "" },
        });
    }
    if (auto sec = BeginSection("Optimizer", true)) {
        // [DATA HOOK] Model::getOptimizerConfig() — name + lr + betas +
        // schedule etc.  Engine source: torch.optim instance config dict.
        const auto o = m.getOptimizerConfig();
        char betas[24];
        std::snprintf(betas, sizeof betas, "%.2f, %.2f", double(o.beta1), double(o.beta2));
        const std::string bs = (std::isnan(o.beta1) || std::isnan(o.beta2)) ? "—" : betas;
        char wm[16];   std::snprintf(wm, sizeof wm, "%s step", FmtInt(o.warmup_steps).c_str());
        KV({
            { "name",         Or(o.name),                            "" },
            { "lr",           FmtFloat(o.lr, "%.1e"),                "accent" },
            { "β1, β2",       bs,                                    "" },
            { "warmup",       o.warmup_steps == kNoInt ? "—" : wm,   "" },
            { "schedule",     Or(o.schedule),                        "" },
            { "weight_decay", FmtFloat(o.weight_decay, "%.2f"),      "" },
        });
    }
    if (auto sec = BeginSection("Data", true)) {
        // [DATA HOOK] Model::getDataConfig() — dataset name + sample
        // counts (in K) + ctx_len + packing.
        const auto d = m.getDataConfig();
        char nt[16]; std::snprintf(nt, sizeof nt, "%dk", d.n_train_K);
        char ne[16]; std::snprintf(ne, sizeof ne, "%dk", d.n_eval_K);
        KV({
            { "dataset", Or(d.dataset),                          "" },
            { "n_train", d.n_train_K == kNoInt ? "—" : nt,       "accent" },
            { "n_eval",  d.n_eval_K  == kNoInt ? "—" : ne,       "" },
            { "ctx_len", FmtInt(d.ctx_len),                      "" },
            { "packing", d.packing ? "on" : "off",               d.packing ? "good" : "" },
        });
    }
    ImGui::EndChild();
}

void DrawLayerFreeze(const ModelInfo& mi, std::unordered_set<int>& frozen) {
    // The frozen-layer set is interactive UI state (user clicks tiles to
    // toggle).  When the engine is wired, the toggle would call a
    // hypothetical Model::setLayerFrozen(layer, bool) — for now it's UI
    // state pushed into AppState (or the static set used here).
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
            // [DATA HOOK] Model::setLayerFrozen(L, !fr) when wired.
            if (fr) frozen.erase(L); else frozen.insert(L);
        }
        ImGui::PopStyleColor(3);
        ImGui::PopID();
        if ((L + 1) % per_row != 0) ImGui::SameLine();
    }
    ImGui::NewLine();

    char fb[32]; std::snprintf(fb, sizeof fb, "%zu / %d layers", frozen.size(), mi.nLayers);
    char tb[32]; std::snprintf(tb, sizeof tb, "%zu layers",
                                std::size_t(mi.nLayers) - frozen.size());
    KV({
        { "frozen",       fb,    "info" },
        { "trainable",    tb,    "accent" },
        { "Δ params",     "—",   "" },         // [DATA HOOK] from LoRAConfig.trainable_M
        { "GPU mem est.", "—",   "" },         // [DATA HOOK] from EngineMetrics
    }, true);
    ImGui::EndChild();
}

void DrawEvalDiff(Model& m) {
    DrawTitleBar("eval_diff", "≠", nullptr, "evaldiff");
    if (!ImGui::BeginChild("##ed_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    if (auto sec = BeginSection("base vs ft", true)) {
        // [DATA HOOK] Model::getEvalDiff(benchmark) — base / ft / Δ for the
        // named benchmark.  Engine source: lm-eval-harness or in-house eval.
        const auto e = m.getEvalDiff("MMLU");
        char dlt[16]; std::snprintf(dlt, sizeof dlt, "%+.3f", double(e.delta));
        KV({
            { e.benchmark.empty() ? "benchmark" : e.benchmark.c_str(),
                                       FmtFloat(e.base, "%.3f"),  "" },
            { "ft (latest)",           FmtFloat(e.ft,   "%.3f"),  "good" },
            { "Δ",                     std::isnan(e.delta) ? "—" : dlt, "good" },
        });
    }
    if (auto sec = BeginSection("loss curves", true)) {
        // [DATA HOOK] Model::getEvalLossCurve() — finetune loss progression
        // (downsampled for the sparkline).
        const auto d = m.getEvalLossCurve();
        if (d.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
            ImGui::TextUnformatted("// no curve");
            ImGui::PopStyleColor();
        } else {
            SparkOpts so{}; so.color = Sty().accent; so.fill = true;
            so.width = 280; so.height = 48;
            Sparkline(d, so);
        }
    }
    if (auto sec = BeginSection("A/B sample", true)) {
        // [DATA HOOK] Model::getABSample() — current sample prompt + base
        // model + ft model responses, for side-by-side comparison.
        const auto ab = m.getABSample();
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
        ImGui::TextUnformatted("PROMPT"); ImGui::PopStyleColor();
        ImGui::TextUnformatted(Or(ab.prompt).c_str());
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_dim);
        ImGui::TextUnformatted("BASE →"); ImGui::PopStyleColor();
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
        ImGui::TextUnformatted(Or(ab.base_response).c_str());
        ImGui::PopStyleColor();
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().accent);
        ImGui::TextUnformatted("FT →"); ImGui::PopStyleColor();
        ImGui::TextUnformatted(Or(ab.ft_response).c_str());
    }
    ImGui::EndChild();
}

void DrawDiffRun(const ModelInfo& mi, const std::unordered_set<int>& frozen, Model& m) {
    DrawTitleBar("diff_run", "⇄", nullptr, "diff");
    if (!ImGui::BeginChild("##dr_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    if (auto sec = BeginSection("parameter delta heatmap · ‖ΔW‖ per layer × component", true)) {
        // [DATA HOOK] Model::getDeltaWHeatmap(numLayers, numComponents) —
        // per-layer × per-component ‖ΔW‖ matrix.  Engine source: diff
        // between the LoRA-merged adapter and the base checkpoint.
        // [DATA HOOK] Model::getDeltaWComponentNames() — column labels.
        auto data = m.getDeltaWHeatmap(mi.nLayers, 12);
        const auto names = m.getDeltaWComponentNames();
        // Apply the UI-side freeze mask (engine doesn't know about UI
        // intent yet — this is what the user has TENTATIVELY frozen).
        for (int L = 0; L < int(data.size()); ++L)
            if (frozen.contains(L))
                std::fill(data[L].begin(), data[L].end(), 0.0f);

        if (data.empty() || data[0].empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
            ImGui::TextUnformatted("// no ΔW data — train a finetune first");
            ImGui::PopStyleColor();
        } else {
            HeatmapOpts ho{}; ho.width = ImGui::GetContentRegionAvail().x - 8; ho.height = 120;
            TensorHeatmap(data, ho);
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
            std::string row;
            for (auto& n : names) { row += n; row += "  "; }
            if (row.empty()) row = "// no component names";
            ImGui::TextUnformatted(row.c_str());
            ImGui::PopStyleColor();
        }
    }
    ImGui::EndChild();
}

}  // namespace

// Layout: lora_config | center(layer_freeze / diff_run) | eval_diff
void BuildFineTuneLayout(ImGuiID dock_id) {
    ImGui::DockBuilderRemoveNode(dock_id);
    ImGui::DockBuilderAddNode(dock_id, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dock_id, ImGui::GetMainViewport()->Size);

    ImGuiID lora_n, rest1, rest2, eval_n, freeze_n, diff_n;
    ImGui::DockBuilderSplitNode(dock_id, ImGuiDir_Left,  0.20f, &lora_n, &rest1);
    ImGui::DockBuilderSplitNode(rest1,   ImGuiDir_Right, 0.24f, &eval_n, &rest2);
    ImGui::DockBuilderSplitNode(rest2,   ImGuiDir_Down,  0.32f, &diff_n, &freeze_n);

    ImGui::DockBuilderDockWindow("ft.lora_config",  lora_n);
    ImGui::DockBuilderDockWindow("ft.layer_freeze", freeze_n);
    ImGui::DockBuilderDockWindow("ft.diff_run",     diff_n);
    ImGui::DockBuilderDockWindow("ft.eval_diff",    eval_n);
    ImGui::DockBuilderFinish(dock_id);
}

void SubmitFineTunePanels(AppState& s, Model& m) {
    if (!s.hasModel()) {
        if (ImGui::Begin("ft.lora_config", nullptr, ImGuiWindowFlags_NoTitleBar)) {
            EmptyStatePlaceholder("// no model loaded — open a checkpoint to configure finetuning");
        }
        ImGui::End();
        return;
    }
    static std::unordered_set<int> frozen{0, 1, 2, 3};
    if (ImGui::Begin("ft.lora_config",  nullptr, ImGuiWindowFlags_NoTitleBar)) DrawLoRAConfig(m);
    ImGui::End();
    if (ImGui::Begin("ft.layer_freeze", nullptr, ImGuiWindowFlags_NoTitleBar)) DrawLayerFreeze(s.model, frozen);
    ImGui::End();
    if (ImGui::Begin("ft.diff_run",     nullptr, ImGuiWindowFlags_NoTitleBar)) DrawDiffRun(s.model, frozen, m);
    ImGui::End();
    if (ImGui::Begin("ft.eval_diff",    nullptr, ImGuiWindowFlags_NoTitleBar)) DrawEvalDiff(m);
    ImGui::End();
}

}  // namespace llob
