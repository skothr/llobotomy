// Training workspace — run summary + control on top, loss/lr + grad_flow,
// per-layer loss sparkline grid at bottom.  HANDOFF §3.5.
//
// All values come from the Model interface; nothing in this file is
// hardcoded.  The data hooks each section needs are documented inline as
// `// [DATA HOOK]` comments naming the Model::* method that supplies them.

#include "workspaces/workspaces.hpp"

#include "appstate.hpp"
#include "model/model.hpp"
#include "logger.hpp"
#include "style.hpp"
#include "ui/chrome.hpp"
#include "ui/colormap.hpp"
#include "ui/fmt.hpp"
#include "ui/widgets.hpp"

#include <imgui.h>
#include <imgui_internal.h>           // DockBuilder*

#include <algorithm>
#include <cstdio>
#include <span>
#include <vector>

namespace llob {

namespace {

void DrawRunSummary(Model& m) {
    // [DATA HOOK] Model::getTrainingState() — current step / total /
    // running flag.  [DATA HOOK] Model::getTrainingMetrics() — list of
    // metric cards (LOSS, VAL LOSS, LR, ...) with display strings.
    const auto st    = m.getTrainingState();
    const auto cards = m.getTrainingMetrics();
    char flag[40];
    std::snprintf(flag, sizeof flag, "step %d / %s",
                   st.step,
                   st.total_steps == kNoInt ? "—" : (FmtInt(st.total_steps) + std::string()).c_str());
    DrawTitleBar("run_summary", st.running ? "●" : "⏸", flag, "run-sum");
    if (!ImGui::BeginChild("##rs_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    if (cards.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
        ImGui::TextUnformatted("// no training metrics available");
        ImGui::PopStyleColor();
        ImGui::EndChild();
        return;
    }
    const int n = int(cards.size());
    const float w = (ImGui::GetContentRegionAvail().x - (n - 1) * 12) / float(n);
    for (int i = 0; i < n; ++i) {
        ImGui::BeginGroup();
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
        ImGui::TextUnformatted(cards[i].label.c_str()); ImGui::PopStyleColor();
        ImGui::PushStyleColor(ImGuiCol_Text,
            cards[i].tone && *cards[i].tone ? ToneColor(cards[i].tone) : Sty().text_bright);
        ImGui::TextUnformatted(cards[i].value_str.c_str()); ImGui::PopStyleColor();
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
        ImGui::TextUnformatted(cards[i].sub_str.c_str()); ImGui::PopStyleColor();
        ImGui::EndGroup();
        if (i < n - 1) ImGui::SameLine(0, 12);
        (void)w;
    }
    ImGui::EndChild();
}

void DrawControl(AppState&, Model& m) {
    DrawTitleBar("control", "⏵", nullptr, "ctrl");
    if (!ImGui::BeginChild("##ctrl_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    const auto st = m.getTrainingState();
    if (ImGui::Button(st.running ? "|| pause" : "> resume")) {
        // [DATA HOOK] Model::pauseTraining() / resumeTraining() — gate the
        // training loop's tick.  Engine emits its own log line.
        if (st.running) { m.pauseTraining();  LLOB_LOG_INFO("train", "paused");  }
        else            { m.resumeTraining(); LLOB_LOG_INFO("train", "resumed"); }
    }
    ImGui::SameLine();
    if (ImGui::Button("⏵ step")) {
        // [DATA HOOK] Model::stepTraining() — advance one batch and stop.
        m.stepTraining();
        LLOB_LOG_INFO("train", "stepped one batch");
    }
    ImGui::SameLine();
    if (ImGui::Button("(reset)")) {
        // [DATA HOOK] Model::resetTraining() — re-init from step 0.
        m.resetTraining();
        LLOB_LOG_INFO("train", "reset to step 0");
    }
    ImGui::SameLine();
    if (ImGui::Button("■ stop")) {
        // [DATA HOOK] Model::stopTraining() — request a clean shutdown.
        m.stopTraining();
        LLOB_LOG_INFO("train", "STOP requested");
    }
    // [DATA HOOK] Schedule + sync state belong on a future
    // TrainingScheduleConfig hook.  Placeholders here.
    KV({
        { "ckpt every", "—", "" },                // [DATA HOOK]
        { "eval every", "—", "" },                // [DATA HOOK]
        { "hf-sync",    "—", "" },                // [DATA HOOK]
        { "wandb",      "—", "" },                // [DATA HOOK]
    });
    ImGui::EndChild();
}

void DrawLossPlot(Model& m) {
    DrawTitleBar("loss / lr", "∿", nullptr, "loss-plot");
    if (!ImGui::BeginChild("##lp_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    if (auto sec = BeginSection("train_loss", true)) {
        // [DATA HOOK] Model::getTrainingLoss(maxSteps) — train + val loss
        // history (each at sample resolution).  Engine source: ring buffer
        // of the last N values from each scalar log.
        const auto curve = m.getTrainingLoss(80);
        if (curve.train.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
            ImGui::TextUnformatted("// no loss history yet");
            ImGui::PopStyleColor();
        } else {
            SparkOpts so{}; so.color = Sty().accent; so.fill = true;
            so.width = ImGui::GetContentRegionAvail().x - 8; so.height = 160;
            so.min = 0.2f; so.max = 3.5f;
            Sparkline(std::span{curve.train}, so);
            // val curve overlaid (smaller, no fill).
            // (Could overlay with separate Sparkline + custom AddPolyline.)
        }
    }
    ImGui::EndChild();
}

void DrawGradFlow(const ModelInfo& mi, Model& m) {
    DrawTitleBar("grad_flow", "↯", nullptr, "grad-flow");
    if (!ImGui::BeginChild("##gf_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    if (auto sec = BeginSection("‖∂L/∂W‖ per layer", true)) {
        // [DATA HOOK] Model::getGradFlowPerLayer() — per-layer gradient
        // norm at the most recent backward pass.  Length is whatever the
        // engine returns; we trim to nLayers.
        const auto v = m.getGradFlowPerLayer();
        if (v.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
            ImGui::TextUnformatted("// no gradient data");
            ImGui::PopStyleColor();
        } else {
            for (int L = 0; L < mi.nLayers; ++L) {
                const float g = L < int(v.size()) ? v[L] : 0.0f;
                ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
                ImGui::Text("L%02d", L); ImGui::PopStyleColor();
                ImGui::SameLine(40);
                Bar(g, 140, 5, L < 3 ? Sty().bad : Sty().accent);
                ImGui::SameLine();
                ImGui::TextUnformatted(FmtFloat(g, "%.3f").c_str());
            }
        }
    }
    ImGui::EndChild();
}

void DrawLayerLoss(const ModelInfo& mi, Model& m) {
    DrawTitleBar("layerwise_loss", "▦", nullptr, "layer-loss");
    if (!ImGui::BeginChild("##ll_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    if (auto sec = BeginSection("per-layer loss sparkline grid", true)) {
        // [DATA HOOK] Model::getPerLayerLoss(maxSteps) — 2D matrix
        // [layer][step] of layer-resolved loss values, useful for spotting
        // which layers are unstable mid-training.
        const auto rows = m.getPerLayerLoss(40);
        if (rows.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
            ImGui::TextUnformatted("// no per-layer loss data");
            ImGui::PopStyleColor();
        } else {
            const int per_row = 4;
            const float w = (ImGui::GetContentRegionAvail().x - (per_row - 1) * 8) / float(per_row);
            for (int L = 0; L < mi.nLayers; ++L) {
                if (L >= int(rows.size())) break;
                const auto& data = rows[L];
                if (data.empty()) continue;
                ImGui::BeginGroup();
                ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
                ImGui::Text("L%02d", L); ImGui::PopStyleColor();
                SparkOpts so{};
                so.color  = L < 4 ? Sty().bad : (L < 8 ? Sty().warn : Sty().accent);
                so.fill   = true;
                so.width  = w - 4;
                so.height = 28;
                Sparkline(std::span{data}, so);
                ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
                ImGui::Text("%.2f → %.2f", double(data.front()), double(data.back()));
                ImGui::PopStyleColor();
                ImGui::EndGroup();
                if ((L + 1) % per_row != 0) ImGui::SameLine();
            }
        }
    }
    ImGui::EndChild();
}

}  // namespace

// Layout: top(run | ctrl) / mid(loss | grad) / bot layer_loss (full width)
void BuildTrainingLayout(ImGuiID dock_id) {
    ImGui::DockBuilderRemoveNode(dock_id);
    ImGui::DockBuilderAddNode(dock_id, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dock_id, ImGui::GetMainViewport()->Size);

    ImGuiID top_n, rest1, mid_n, bot_n, run_n, ctrl_n, loss_n, grad_n;
    ImGui::DockBuilderSplitNode(dock_id, ImGuiDir_Down,  0.78f, &rest1, &top_n);
    ImGui::DockBuilderSplitNode(rest1,  ImGuiDir_Down,   0.50f, &bot_n, &mid_n);
    ImGui::DockBuilderSplitNode(top_n,  ImGuiDir_Right,  0.20f, &ctrl_n, &run_n);
    ImGui::DockBuilderSplitNode(mid_n,  ImGuiDir_Right,  0.20f, &grad_n, &loss_n);

    ImGui::DockBuilderDockWindow("train.run",        run_n);
    ImGui::DockBuilderDockWindow("train.ctrl",       ctrl_n);
    ImGui::DockBuilderDockWindow("train.loss",       loss_n);
    ImGui::DockBuilderDockWindow("train.grad",       grad_n);
    ImGui::DockBuilderDockWindow("train.layer_loss", bot_n);
    ImGui::DockBuilderFinish(dock_id);
}

void SubmitTrainingPanels(AppState& s, Model& m) {
    if (!s.hasModel()) {
        if (ImGui::Begin("train.run", nullptr, ImGuiWindowFlags_NoTitleBar)) {
            EmptyStatePlaceholder("// no model loaded — open a checkpoint to begin training");
        }
        ImGui::End();
        return;
    }
    if (ImGui::Begin("train.run",        nullptr, ImGuiWindowFlags_NoTitleBar)) DrawRunSummary(m);
    ImGui::End();
    if (ImGui::Begin("train.ctrl",       nullptr, ImGuiWindowFlags_NoTitleBar)) DrawControl(s, m);
    ImGui::End();
    if (ImGui::Begin("train.loss",       nullptr, ImGuiWindowFlags_NoTitleBar)) DrawLossPlot(m);
    ImGui::End();
    if (ImGui::Begin("train.grad",       nullptr, ImGuiWindowFlags_NoTitleBar)) DrawGradFlow(s.model, m);
    ImGui::End();
    if (ImGui::Begin("train.layer_loss", nullptr, ImGuiWindowFlags_NoTitleBar)) DrawLayerLoss(s.model, m);
    ImGui::End();
}

}  // namespace llob
