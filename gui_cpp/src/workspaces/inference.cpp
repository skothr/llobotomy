// Inference workspace — residual_flow | (forward_pass + token_strip) |
// (probe panel + probe controls) | [raw tensor], split per HANDOFF §3.2.

#include "workspaces/workspaces.hpp"

#include "appstate.hpp"
#include "model/model.hpp"
#include "style.hpp"
#include "ui/chrome.hpp"
#include "ui/widgets.hpp"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace llob {

namespace {

void DrawResidualFlow(AppState& s, Model&) {
    DrawTitleBar("residual_flow", "≡", "float", "resflow");
    if (!ImGui::BeginChild("##rf_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }

    // Top: token info + LIVE pill.
    char tok[64];
    if (s.activeToken < int(s.sampleTokens.size())) {
        std::snprintf(tok, sizeof tok, "Token: \"%s\" [pos %d]",
                      s.sampleTokens[s.activeToken].c_str(), s.activeToken);
    } else {
        std::snprintf(tok, sizeof tok, "Token: ?? [pos %d]", s.activeToken);
    }
    ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
    ImGui::TextUnformatted(tok);
    ImGui::PopStyleColor();
    ImGui::SameLine(); ImGui::Dummy(ImVec2(8, 0)); ImGui::SameLine();
    Pill(s.running ? "RUN" : "LIVE", s.running ? "good" : "accent");

    // Body: layer-rows with attn / mlp contribution bars.
    const float w = ImGui::GetContentRegionAvail().x;
    const float h = std::max(160.0f, ImGui::GetContentRegionAvail().y - 120.0f);
    const ImVec2 p0 = ImGui::GetCursorScreenPos();
    const ImVec2 p1 = { p0.x + w, p0.y + h };
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p0, p1, Sty().bg_input);
    dl->AddRect      (p0, p1, Sty().border);

    const int n = s.model.nLayers;
    const float row = (h - 16.0f) / float(std::max(1, n));
    const float laneX = p0.x + w * 0.5f;
    dl->AddLine({laneX, p0.y + 8}, {laneX, p1.y - 8}, Sty().accent_dim, 2.0f);
    AddDashedLine(dl, {p0.x + 50, p0.y + 8}, {p0.x + 50, p1.y - 8},  Sty().info, 2, 3, 1);
    AddDashedLine(dl, {p1.x - 50, p0.y + 8}, {p1.x - 50, p1.y - 8},  Sty().warn, 2, 3, 1);
    dl->AddText({p0.x + 36, p0.y + 0}, Sty().info,  "attn");
    dl->AddText({laneX - 12, p0.y + 0}, Sty().accent, "resid");
    dl->AddText({p1.x - 60, p0.y + 0}, Sty().warn,   "mlp");

    Mulberry32 rng(7);
    for (int i = 0; i < n; ++i) {
        const float yc = p0.y + 16 + i * row + row * 0.5f;
        const bool active = (i == s.activeLayer);
        if (active) {
            dl->AddRectFilled({p0.x, yc - row * 0.5f + 1}, {p1.x, yc + row * 0.5f - 1},
                              Sty().accent_bg);
        }
        const float att = 0.2f + rng.next() * 0.7f;
        const float mlp = 0.15f + rng.next() * 0.7f;
        dl->AddRectFilled({p0.x + 56,        yc - 1}, {p0.x + 56 + 20 * att, yc + 1}, Sty().info);
        dl->AddRectFilled({p1.x - 76,        yc - 1}, {p1.x - 76 + 20 * mlp, yc + 1}, Sty().warn);
        dl->AddCircle({p0.x + 50, yc}, 4, Sty().info, 0, 1);
        dl->AddCircle({p1.x - 50, yc}, 4, Sty().warn, 0, 1);
        char ll[8]; std::snprintf(ll, sizeof ll, "L%02d", i);
        dl->AddText({laneX + 8, yc - 6},
                    active ? Sty().accent : Sty().text_muted, ll);

        // Hit-test the row.
        ImGui::PushID(i);
        ImGui::SetCursorScreenPos({p0.x, yc - row * 0.5f});
        ImGui::InvisibleButton("rrow", { w, row });
        if (ImGui::IsItemClicked()) s.setActiveLayer(i);
        ImGui::PopID();
    }
    ImGui::SetCursorScreenPos({p0.x, p1.y + 4});
    ImGui::Dummy(ImVec2(0, 0));

    // Bottom KV summary.
    char L[16]; std::snprintf(L, sizeof L, "L%02d", s.activeLayer);
    KV({
        { "Sel layer",  L,        "accent" },
        { "||attn_out||", "0.421",  "accent" },
        { "||mlp_out||",  "0.388",  "warn" },
        { "||resid||",    "14.832", "" },
        { "cos(prev)",    "0.991",  "good" },
    }, true);

    ImGui::EndChild();
}

void DrawForwardPass(AppState& s, Model&) {
    char flag[16]; std::snprintf(flag, sizeof flag, "%s", s.running ? "running…" : "idle");
    DrawTitleBar("forward_pass", "▶", flag, "fwd", [&] {
        if (ImGui::SmallButton(s.running ? "|| pause" : "> run")) {
            s.running = !s.running;
            s.pushLog("run", s.running ? "starting forward sweep" : "paused forward sweep");
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("step")) { s.setActiveToken(s.activeToken + 1); s.pushLog("run", "step → next token"); }
        ImGui::SameLine();
        if (ImGui::SmallButton("(reset)")) { s.activeToken = 0; s.stepIdx = 0; s.running = false; s.pushLog("run", "reset to pos 0"); }
    });
    if (!ImGui::BeginChild("##fwd_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }

    char b[24]; std::snprintf(b, sizeof b, "%zu tok", s.sampleTokens.size());
    if (auto sec = BeginSection("Prompt", false, b)) {
        for (std::size_t i = 0; i < s.sampleTokens.size(); ++i) {
            ImGui::PushID(int(i));
            const bool act = (int(i) == s.activeToken);
            ImGui::PushStyleColor(ImGuiCol_Button,        act ? Sty().accent_bg_strong : Sty().bg_input);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, Sty().bg_input_hover);
            ImGui::PushStyleColor(ImGuiCol_Text,          act ? Sty().accent : Sty().text);
            char chip[64]; std::snprintf(chip, sizeof chip, "%s %zu", s.sampleTokens[i].c_str(), i);
            if (ImGui::SmallButton(chip)) s.setActiveToken(int(i));
            ImGui::PopStyleColor(3);
            ImGui::PopID();
            if ((i + 1) % 8 != 0) ImGui::SameLine();
        }
        ImGui::NewLine();
        EndSection(sec);
    }

    if (auto sec = BeginSection("Per-token logit-lens trajectory", false, "layer × top-1")) {
        if (ImGui::BeginTable("##lens", 7, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInner)) {
            ImGui::TableSetupColumn("L",       ImGuiTableColumnFlags_WidthFixed,  32);
            ImGui::TableSetupColumn("top-1",   ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("prob",    ImGuiTableColumnFlags_WidthFixed,  56);
            ImGui::TableSetupColumn("2nd",     ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("prob",    ImGuiTableColumnFlags_WidthFixed,  56);
            ImGui::TableSetupColumn("entropy", ImGuiTableColumnFlags_WidthFixed,  60);
            ImGui::TableSetupColumn("shift",   ImGuiTableColumnFlags_WidthFixed,  80);
            ImGui::TableHeadersRow();

            const struct Row { const char* L; const char* t1; float p1; const char* t2; float p2; float ent; }
                rows[] = {
                    {"00", "the",     0.04f, "a",       0.03f, 4.82f},
                    {"01", "the",     0.06f, "of",      0.04f, 4.61f},
                    {"02", "the",     0.09f, "a",       0.05f, 4.32f},
                    {"03", "to",      0.11f, "the",     0.07f, 4.04f},
                    {"04", "to",      0.18f, "on",      0.09f, 3.71f},
                    {"05", "to",      0.28f, "towards", 0.07f, 3.21f},
            };
            for (auto& r : rows) {
                ImGui::PushID(r.L);
                const int Li = std::atoi(r.L);
                if (ImGui::TableNextColumn()) {
                    ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
                    ImGui::TextUnformatted(r.L); ImGui::PopStyleColor();
                }
                if (ImGui::TableNextColumn()) {
                    ImGui::PushStyleColor(ImGuiCol_Text, Sty().accent);
                    ImGui::Text("\"%s\"", r.t1); ImGui::PopStyleColor();
                }
                if (ImGui::TableNextColumn()) ImGui::Text("%.3f", r.p1);
                if (ImGui::TableNextColumn()) {
                    ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
                    ImGui::Text("\"%s\"", r.t2); ImGui::PopStyleColor();
                }
                if (ImGui::TableNextColumn()) ImGui::Text("%.3f", r.p2);
                if (ImGui::TableNextColumn()) ImGui::Text("%.2f",  r.ent);
                if (ImGui::TableNextColumn()) Bar(1.0f - r.ent / 5.0f, 70, 6, Sty().accent);

                ImGui::SameLine();
                if (ImGui::IsItemClicked()) s.setActiveLayer(Li);
                ImGui::PopID();
            }
            ImGui::EndTable();
        }
        EndSection(sec);
    }

    if (auto sec = BeginSection("Output logits · next-token distribution")) {
        const LogitItem items[] = {
            { "to",       0.92f,  +0.04f, true  },
            { "towards",  0.022f, -0.01f, false },
            { "on",       0.018f,  0.0f,  false },
            { "at",       0.011f, -0.002f,false },
            { "for",      0.008f, +0.001f,false },
            { "through",  0.005f,  0.0f,  false },
        };
        LogitBars(std::span{items, std::size(items)});
        EndSection(sec);
    }

    ImGui::EndChild();
}

void DrawProbePanel(AppState& s, Model& m) {
    char title[32]; std::snprintf(title, sizeof title, "L%02d_probe", s.activeLayer);
    DrawTitleBar(title, "◈", nullptr, "probe");
    if (!ImGui::BeginChild("##probe_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }

    char id[16];   std::snprintf(id,   sizeof id,   "L%02d", s.activeLayer);
    char shape[40];std::snprintf(shape,sizeof shape,"[%zu, %d]", s.sampleTokens.size(), s.model.dModel);
    if (auto sec = BeginSection("Layer summary", true)) {
        KV({
            { "layer.id",  id,    "accent" },
            { "resid_pre", shape, "" },
            { "attn_out",  shape, "" },
            { "mlp_out",   shape, "" },
            { "||attn||",  "0.421", "accent" },
            { "||mlp||",   "0.388", "warn" },
            { "cos(L-1)",  "0.9821","good" },
            { "kurtosis",  "4.32",  "" },
        });
        EndSection(sec);
    }
    if (auto sec = BeginSection("resid_post · histogram", false, "μ=0.001 σ=0.847")) {
        const auto bins = m.getWeightHistogram(s.activeLayer, "resid_post", 40);
        const LensAnnotation an[] = {
            { 0.50f, "μ",         Sty().accent },
            { 0.78f, "top-1 dir", Sty().warn   },
        };
        ActivationHistogram(bins, ImGui::GetContentRegionAvail().x - 8, 84, Sty().info,
                            std::span{an, std::size(an)});
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
        ImGui::TextUnformatted("min -3.21 · max +3.04 · zero-frac 0.024");
        ImGui::PopStyleColor();
        EndSection(sec);
    }
    char hb[8]; std::snprintf(hb, sizeof hb, "%dh", s.model.nHeads);
    if (auto sec = BeginSection("Per-head attention norms", false, hb)) {
        for (int h = 0; h < s.model.nHeads; ++h) {
            ImGui::PushID(h);
            char hk[16]; std::snprintf(hk, sizeof hk, "%d.%d", s.activeLayer, h);
            const bool ab = s.ablatedHeads.contains(hk);
            const bool pr = s.probedHeads.contains(hk);
            const float v = m.getHeadNorm(s.activeLayer, h);
            ImGui::PushStyleColor(ImGuiCol_Button,        h == s.activeHead ? Sty().accent_bg : Sty().bg_input);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, Sty().bg_input_hover);
            ImGui::PushStyleColor(ImGuiCol_Text,          ab ? Sty().bad : (pr ? Sty().accent : Sty().text));
            char hl[16]; std::snprintf(hl, sizeof hl, "h%d  %.2f", h, v * 1.5f);
            if (ImGui::Button(hl, ImVec2(80, 0))) s.setActiveHead(h);
            ImGui::PopStyleColor(3);
            ImGui::PopID();
            if ((h + 1) % 4 != 0) ImGui::SameLine();
        }
        ImGui::NewLine();
        EndSection(sec);
    }
    ImGui::EndChild();
}

void DrawTokenStrip(AppState& s, Model&) {
    DrawTitleBar("token_strip", "▦", nullptr, "tokstrip");
    if (!ImGui::BeginChild("##ts_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    const std::size_t n = s.sampleTokens.size();
    std::vector<std::string_view> tk; tk.reserve(n);
    for (auto& t : s.sampleTokens) tk.emplace_back(t);
    std::vector<float> loss(n);
    for (std::size_t i = 0; i < n; ++i) loss[i] = std::abs(std::sin(i * 0.7f + s.activeLayer * 0.3f));
    if (auto sec = BeginSection("cross-entropy loss per token", false, "L?")) {
        TokenGutter(tk, loss, HeatColor);
        EndSection(sec);
    }
    std::vector<float> diff(n);
    for (std::size_t i = 0; i < n; ++i) diff[i] = std::sin(i * 0.7f) * 0.5f + 0.5f;
    if (auto sec = BeginSection("surprisal Δ vs base run", false, "diff")) {
        TokenGutter(tk, diff, [](float v) { return DivergeColor(v * 2.0f - 1.0f); });
        EndSection(sec);
    }
    ImGui::EndChild();
}

void DrawProbeControls(AppState& s, Model&) {
    DrawTitleBar("probe_controls", "◎", nullptr, "probe-ctrl");
    if (!ImGui::BeginChild("##pc_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    char nbuf[24]; std::snprintf(nbuf, sizeof nbuf, "%zu",
                                  s.probedHeads.size() + s.probedComponents.size());
    if (auto sec = BeginSection("Active probes", true, nbuf)) {
        if (ImGui::BeginTable("##probes", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInner)) {
            ImGui::TableSetupColumn("type"); ImGui::TableSetupColumn("name"); ImGui::TableSetupColumn("acc", ImGuiTableColumnFlags_WidthFixed, 60);
            ImGui::TableHeadersRow();
            ImGui::TableNextRow(); ImGui::TableNextColumn(); Pill("L", "accent", true); ImGui::TableNextColumn(); ImGui::Text("linear/refusal_dir"); ImGui::TableNextColumn(); ImGui::PushStyleColor(ImGuiCol_Text, Sty().good); ImGui::Text("0.91"); ImGui::PopStyleColor();
            ImGui::TableNextRow(); ImGui::TableNextColumn(); Pill("P", "accent");       ImGui::TableNextColumn(); ImGui::Text("logistic/sentiment"); ImGui::TableNextColumn(); ImGui::PushStyleColor(ImGuiCol_Text, Sty().good); ImGui::Text("0.84"); ImGui::PopStyleColor();
            ImGui::TableNextRow(); ImGui::TableNextColumn(); Pill("S", "warn");         ImGui::TableNextColumn(); ImGui::Text("SAE/feature_2381");   ImGui::TableNextColumn(); ImGui::Text("0.62");
            ImGui::EndTable();
        }
        EndSection(sec);
    }
    if (auto sec = BeginSection("Steering vector", true, "ON")) {
        KV({
            { "source", "\"refusal\" prompts (n=128)", "" },
            { "layer",  "L08.resid_post", "accent" },
            { "α",      "+1.40", "warn" },
            { "cos sim","0.873", "good" },
        });
        static float alpha = 1.4f;
        ImSliderF("##alpha", alpha, -3.0f, 3.0f, "%.2f", ImGui::GetContentRegionAvail().x - 8);
        EndSection(sec);
    }
    ImGui::EndChild();
}

}  // namespace

void DrawInferenceWorkspace(AppState& s, Model& m) {
    const float W = ImGui::GetContentRegionAvail().x;
    const float H = ImGui::GetContentRegionAvail().y;
    const float gap = 1.0f;
    const bool  raw = s.showRaw;
    const float left_w   = 320.0f;
    const float right_w  = 360.0f;
    const float rraw_w   = raw ? 170.0f : 0.0f;
    const float center_w = std::max(200.0f, W - left_w - right_w - rraw_w - 3 * gap);
    const float bot_h    = std::min(220.0f, H * 0.30f);
    const float top_h    = H - bot_h - gap;

    ImGui::BeginChild("##inf_left", { left_w, H }, ImGuiChildFlags_Borders);
    DrawResidualFlow(s, m);
    ImGui::EndChild(); ImGui::SameLine(0, gap);

    ImGui::BeginChild("##inf_center", { center_w, H });
    ImGui::BeginChild("##inf_fwd",   { center_w, top_h }, ImGuiChildFlags_Borders);
    DrawForwardPass(s, m); ImGui::EndChild();
    ImGui::BeginChild("##inf_strip", { center_w, bot_h }, ImGuiChildFlags_Borders);
    DrawTokenStrip(s, m);  ImGui::EndChild();
    ImGui::EndChild(); ImGui::SameLine(0, gap);

    ImGui::BeginChild("##inf_right", { right_w, H });
    ImGui::BeginChild("##inf_probe", { right_w, top_h }, ImGuiChildFlags_Borders);
    DrawProbePanel(s, m); ImGui::EndChild();
    ImGui::BeginChild("##inf_pctrl", { right_w, bot_h }, ImGuiChildFlags_Borders);
    DrawProbeControls(s, m); ImGui::EndChild();
    ImGui::EndChild();

    if (raw) {
        ImGui::SameLine(0, gap);
        ImGui::BeginChild("##inf_raw", { rraw_w, H }, ImGuiChildFlags_Borders);
        DrawTitleBar("raw_tensor", "0x", "fp16", "raw");
        if (ImGui::BeginChild("##raw_body", ImVec2(0, 0))) {
            const auto buf = m.getActivation(s.activeLayer, 0, 256);
            HexView(buf, 0, 4, 28, HexMode::Fp16);
            ImGui::EndChild();
        }
        ImGui::EndChild();
    }
}

}  // namespace llob
