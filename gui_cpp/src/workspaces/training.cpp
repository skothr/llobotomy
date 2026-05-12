// Training workspace — run summary + control on top, loss/lr + grad_flow,
// per-layer loss sparkline grid at bottom.  HANDOFF §3.5.

#include "workspaces/workspaces.hpp"

#include "appstate.hpp"
#include "model/model.hpp"
#include "style.hpp"
#include "ui/chrome.hpp"
#include "ui/colormap.hpp"
#include "ui/widgets.hpp"

#include <imgui.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

namespace llob {

namespace {

void DrawRunSummary(int step) {
    char flag[40]; std::snprintf(flag, sizeof flag, "step %d / 100k", step);
    DrawTitleBar("run_summary", "●", flag, "run-sum");
    if (!ImGui::BeginChild("##rs_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    const struct { const char* l; const char* v; const char* sub; const char* tone; }
        cards[] = {
            { "LOSS",      "2.184",   "-0.012",     "good" },
            { "VAL LOSS",  "2.241",   "-0.008",     "good" },
            { "LR",        "3.0e-4",  "cosine",     ""     },
            { "TOK/SEC",   "184,210", "+1.2%",      "good" },
            { "GPU UTIL",  "94.2%",   "A100×8",     "good" },
            { "GRAD NORM", "1.48",    "clipped",    "warn" },
            { "EPOCH",     "0.40",    "/3.00",      ""     },
        };
    const float w = (ImGui::GetContentRegionAvail().x - 6 * 12) / 7.0f;
    for (int i = 0; i < 7; ++i) {
        ImGui::BeginGroup();
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
        ImGui::TextUnformatted(cards[i].l); ImGui::PopStyleColor();
        ImGui::PushStyleColor(ImGuiCol_Text, cards[i].tone[0] ? ToneColor(cards[i].tone) : Sty().text_bright);
        ImGui::TextUnformatted(cards[i].v);
        ImGui::PopStyleColor();
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
        ImGui::TextUnformatted(cards[i].sub); ImGui::PopStyleColor();
        ImGui::EndGroup();
        if (i < 6) { ImGui::SameLine(0, 12); }
        (void)w;
    }
    ImGui::EndChild();
}

void DrawControl(AppState& s, bool& running, int& step) {
    DrawTitleBar("control", "⏵", nullptr, "ctrl");
    if (!ImGui::BeginChild("##ctrl_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    if (ImGui::Button(running ? "|| pause" : "> resume"))  { running = !running; s.pushLog("train", running ? "resumed" : "paused"); }
    ImGui::SameLine();
    if (ImGui::Button("⏵ step"))                            { ++step; s.pushLog("train", "stepped one batch"); }
    ImGui::SameLine();
    if (ImGui::Button("(reset)"))                           { step = 0; s.pushLog("train", "reset to step 0"); }
    ImGui::SameLine();
    if (ImGui::Button("■ stop"))                            { running = false; s.pushLog("train", "STOP requested"); }
    KV({
        { "ckpt every", "1000 step", "" },
        { "eval every", "500 step",  "" },
        { "hf-sync",    "on",        "good" },
        { "wandb",      "live",      "good" },
    });
    ImGui::EndChild();
}

void DrawLossPlot() {
    DrawTitleBar("loss / lr", "∿", nullptr, "loss-plot");
    if (!ImGui::BeginChild("##lp_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    if (auto sec = BeginSection("train_loss · 24,180 steps", true)) {
        std::vector<float> tr(80), va(80);
        for (int i = 0; i < 80; ++i) {
            tr[i] = 3.2f * std::exp(-i * 0.05f) + 0.3f + std::sin(i * 0.4f) * 0.05f;
            va[i] = tr[i] + 0.06f + std::sin(i * 0.3f) * 0.02f;
        }
        SparkOpts so{}; so.color = Sty().accent; so.fill = true;
        so.width = ImGui::GetContentRegionAvail().x - 8; so.height = 160;
        so.min = 0.2f; so.max = 3.5f;
        Sparkline(tr, so);
        EndSection(sec);
    }
    ImGui::EndChild();
}

void DrawGradFlow(const ModelInfo& mi) {
    DrawTitleBar("grad_flow", "↯", nullptr, "grad-flow");
    if (!ImGui::BeginChild("##gf_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    if (auto sec = BeginSection("‖∂L/∂W‖ per layer", true)) {
        for (int L = 0; L < mi.nLayers; ++L) {
            const float v = std::exp(-L * 0.02f) * (0.6f + std::sin(float(L)) * 0.1f);
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
            ImGui::Text("L%02d", L); ImGui::PopStyleColor();
            ImGui::SameLine(40);
            Bar(v, 140, 5, L < 3 ? Sty().bad : Sty().accent);
            ImGui::SameLine();
            ImGui::Text("%.3f", v);
        }
        EndSection(sec);
    }
    ImGui::EndChild();
}

void DrawLayerLoss(const ModelInfo& mi) {
    DrawTitleBar("layerwise_loss", "▦", nullptr, "layer-loss");
    if (!ImGui::BeginChild("##ll_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    if (auto sec = BeginSection("per-layer loss sparkline grid", true)) {
        const int per_row = 4;
        const float w = (ImGui::GetContentRegionAvail().x - (per_row - 1) * 8) / float(per_row);
        for (int L = 0; L < mi.nLayers; ++L) {
            std::vector<float> data(40);
            for (int i = 0; i < 40; ++i)
                data[i] = (3.0f - L * 0.05f) * std::exp(-i * 0.06f) + (0.4f - L * 0.005f);
            ImGui::BeginGroup();
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
            ImGui::Text("L%02d", L); ImGui::PopStyleColor();
            SparkOpts so{}; so.color = L < 4 ? Sty().bad : (L < 8 ? Sty().warn : Sty().accent);
            so.fill = true; so.width = w - 4; so.height = 28;
            Sparkline(data, so);
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
            ImGui::Text("%.2f → %.2f", data.front(), data.back()); ImGui::PopStyleColor();
            ImGui::EndGroup();
            if ((L + 1) % per_row != 0) ImGui::SameLine();
        }
        EndSection(sec);
    }
    ImGui::EndChild();
}

}  // namespace

void DrawTrainingWorkspace(AppState& s, Model& /*m*/) {
    static bool running = true;
    static int  step    = 24180;
    static auto last    = std::chrono::steady_clock::now();
    if (running) {
        const auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last).count() >= 200) {
            ++step; last = now;
        }
    }

    const float W = ImGui::GetContentRegionAvail().x, H = ImGui::GetContentRegionAvail().y;
    const float gap = 1.0f;
    const float right_w = 320.0f, left_w = std::max(200.0f, W - right_w - gap);
    const float top_h = 180.0f, mid_h = std::max(160.0f, H - top_h - 200.0f - 2 * gap);
    const float bot_h = H - top_h - mid_h - 2 * gap;

    ImGui::BeginChild("##tr_top_left",  { left_w, top_h }, ImGuiChildFlags_Borders);
    DrawRunSummary(step); ImGui::EndChild(); ImGui::SameLine(0, gap);
    ImGui::BeginChild("##tr_top_right", { right_w, top_h }, ImGuiChildFlags_Borders);
    DrawControl(s, running, step); ImGui::EndChild();

    ImGui::BeginChild("##tr_mid_left",  { left_w, mid_h }, ImGuiChildFlags_Borders);
    DrawLossPlot(); ImGui::EndChild(); ImGui::SameLine(0, gap);
    ImGui::BeginChild("##tr_mid_right", { right_w, mid_h }, ImGuiChildFlags_Borders);
    DrawGradFlow(s.model); ImGui::EndChild();

    ImGui::BeginChild("##tr_bot",       { W, bot_h }, ImGuiChildFlags_Borders);
    DrawLayerLoss(s.model); ImGui::EndChild();
}

}  // namespace llob
