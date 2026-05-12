// Inference workspace — residual flow | (forward_pass + token_strip) |
// (probe panel + probe controls) | [raw tensor], split per HANDOFF §3.2.
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
#include "ui/fmt.hpp"
#include "ui/widgets.hpp"

#include <imgui.h>
#include <imgui_internal.h>           // DockBuilder*

#include <algorithm>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

namespace llob {

namespace {

void DrawResidualFlow(AppState& s, Model& m) {
    DrawTitleBar("residual_flow", "≡", "float", "resflow");
    if (!ImGui::BeginChild("##rf_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }

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

    for (int i = 0; i < n; ++i) {
        const float yc = p0.y + 16 + i * row + row * 0.5f;
        const bool active = (i == s.activeLayer);
        if (active) {
            dl->AddRectFilled({p0.x, yc - row * 0.5f + 1}, {p1.x, yc + row * 0.5f - 1},
                              Sty().accent_bg);
        }
        // [DATA HOOK] Model::getResidualContribution(layer) — per-layer
        // {attn, mlp} norm contributions for this token's forward pass.
        const auto c = m.getResidualContribution(i);
        const float att = std::isnan(c.attn) ? 0.0f : c.attn;
        const float mlp = std::isnan(c.mlp)  ? 0.0f : c.mlp;
        dl->AddRectFilled({p0.x + 56, yc - 1}, {p0.x + 56 + 20 * att, yc + 1}, Sty().info);
        dl->AddRectFilled({p1.x - 76, yc - 1}, {p1.x - 76 + 20 * mlp, yc + 1}, Sty().warn);
        dl->AddCircle({p0.x + 50, yc}, 4, Sty().info, 0, 1);
        dl->AddCircle({p1.x - 50, yc}, 4, Sty().warn, 0, 1);
        char ll[8]; std::snprintf(ll, sizeof ll, "L%02d", i);
        dl->AddText({laneX + 8, yc - 6},
                    active ? Sty().accent : Sty().text_muted, ll);

        ImGui::PushID(i);
        ImGui::SetCursorScreenPos({p0.x, yc - row * 0.5f});
        ImGui::InvisibleButton("rrow", { w, row });
        if (ImGui::IsItemClicked()) s.setActiveLayer(i);
        ImGui::PopID();
    }
    ImGui::SetCursorScreenPos({p0.x, p1.y + 4});
    ImGui::Dummy(ImVec2(0, 0));

    // [DATA HOOK] Model::getResidualSummary(layer) — bottom KV summary for
    // the active layer (||attn_out||, ||mlp_out||, ||resid||, cos(prev)).
    const auto sum = m.getResidualSummary(s.activeLayer);
    char L[16]; std::snprintf(L, sizeof L, "L%02d", s.activeLayer);
    const auto vAtt   = FmtFloat(sum.attn_out_norm, "%.3f");
    const auto vMlp   = FmtFloat(sum.mlp_out_norm,  "%.3f");
    const auto vResid = FmtFloat(sum.resid_norm,    "%.3f");
    const auto vCos   = FmtFloat(sum.cos_prev,      "%.3f");
    KV({
        { "Sel layer",    L,      "accent" },
        { "||attn_out||", vAtt,   "accent" },
        { "||mlp_out||",  vMlp,   "warn" },
        { "||resid||",    vResid, "" },
        { "cos(prev)",    vCos,   "good" },
    }, true);

    ImGui::EndChild();
}

void DrawForwardPass(AppState& s, Model& m) {
    char flag[16]; std::snprintf(flag, sizeof flag, "%s", s.running ? "running…" : "idle");
    DrawTitleBar("forward_pass", "▶", flag, "fwd", [&] {
        if (ImGui::SmallButton(s.running ? "|| pause" : "> run")) {
            s.running = !s.running;
            LLOB_LOG_INFO("run", "%s", s.running ? "starting forward sweep" : "paused forward sweep");
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("step")) { s.setActiveToken(s.activeToken + 1); LLOB_LOG_INFO("run", "step → next token"); }
        ImGui::SameLine();
        if (ImGui::SmallButton("(reset)")) { s.activeToken = 0; s.stepIdx = 0; s.running = false; LLOB_LOG_INFO("run", "reset to pos 0"); }
    });
    if (!ImGui::BeginChild("##fwd_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }

    char b[24]; std::snprintf(b, sizeof b, "%zu tok", s.sampleTokens.size());
    if (auto sec = BeginSection("Prompt", false, b)) {
        // Source: AppState.sampleTokens.  In a real run this comes from the
        // tokenizer (Tokenizer::encode on the input string).
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
    }

    if (auto sec = BeginSection("Per-token logit-lens trajectory", false, "layer × top-1")) {
        // [DATA HOOK] Model::getLogitLensTrajectory(token, kLayers) — per-
        // layer top-1 / top-2 token + their probabilities + entropy.  When
        // `is_resolved` is true the row is highlighted as the layer where
        // the prediction stabilises.
        const auto rows = m.getLogitLensTrajectory(s.activeToken, s.model.nLayers);
        if (rows.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
            ImGui::TextUnformatted("// no logit-lens trajectory available");
            ImGui::PopStyleColor();
        } else if (ImGui::BeginTable("##lens", 7, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInner)) {
            ImGui::TableSetupColumn("L",       ImGuiTableColumnFlags_WidthFixed,  32);
            ImGui::TableSetupColumn("top-1",   ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("prob",    ImGuiTableColumnFlags_WidthFixed,  56);
            ImGui::TableSetupColumn("2nd",     ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("prob",    ImGuiTableColumnFlags_WidthFixed,  56);
            ImGui::TableSetupColumn("entropy", ImGuiTableColumnFlags_WidthFixed,  60);
            ImGui::TableSetupColumn("shift",   ImGuiTableColumnFlags_WidthFixed,  80);
            ImGui::TableHeadersRow();
            for (const auto& r : rows) {
                ImGui::PushID(r.layer);
                ImGui::TableNextRow(r.is_resolved ? ImGuiTableRowFlags_Headers : 0);
                if (ImGui::TableNextColumn()) {
                    ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
                    char L[4]; std::snprintf(L, sizeof L, "%02d", r.layer);
                    ImGui::TextUnformatted(L); ImGui::PopStyleColor();
                }
                if (ImGui::TableNextColumn()) {
                    ImGui::PushStyleColor(ImGuiCol_Text, Sty().accent);
                    ImGui::Text("\"%s\"", r.top1.c_str()); ImGui::PopStyleColor();
                }
                if (ImGui::TableNextColumn()) ImGui::TextUnformatted(FmtFloat(r.p1, "%.3f").c_str());
                if (ImGui::TableNextColumn()) {
                    ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
                    ImGui::Text("\"%s\"", r.top2.c_str()); ImGui::PopStyleColor();
                }
                if (ImGui::TableNextColumn()) ImGui::TextUnformatted(FmtFloat(r.p2, "%.3f").c_str());
                if (ImGui::TableNextColumn()) ImGui::TextUnformatted(FmtFloat(r.entropy, "%.2f").c_str());
                if (ImGui::TableNextColumn()) Bar(1.0f - r.entropy / 5.0f, 70, 6, Sty().accent);
                if (ImGui::IsItemClicked()) s.setActiveLayer(r.layer);
                ImGui::PopID();
            }
            ImGui::EndTable();
        }
    }

    if (auto sec = BeginSection("Output logits · next-token distribution")) {
        // [DATA HOOK] Model::getOutputLogits(k) — top-k tokens with prob
        // and (optional) baseline-vs-current delta.  `selected` marks the
        // model's current pick for visual emphasis.
        const auto items = m.getOutputLogits(8);
        if (items.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
            ImGui::TextUnformatted("// no output distribution available");
            ImGui::PopStyleColor();
        } else {
            std::vector<LogitItem> view; view.reserve(items.size());
            for (const auto& it : items) {
                view.push_back({ it.token, it.prob, it.delta, it.selected });
            }
            LogitBars(std::span{view});
        }
    }

    ImGui::EndChild();
}

void DrawProbePanel(AppState& s, Model& m) {
    char title[32]; std::snprintf(title, sizeof title, "L%02d_probe", s.activeLayer);
    DrawTitleBar(title, "◈", nullptr, "probe");
    if (!ImGui::BeginChild("##probe_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }

    if (auto sec = BeginSection("Layer summary", true)) {
        // [DATA HOOK] Model::getResidualSummary(layer) — same struct as
        // residual_flow's bottom KV; aggregated forward-pass norms for the
        // selected layer.
        const auto sum = m.getResidualSummary(s.activeLayer);
        char id[16];   std::snprintf(id,   sizeof id,   "L%02d", s.activeLayer);
        char shape[40];std::snprintf(shape,sizeof shape,"[%zu, %d]",
                                       s.sampleTokens.size(), s.model.dModel);
        char rank[24]; std::snprintf(rank, sizeof rank, "%s / %s",
                                      FmtInt(sum.rank_eff).c_str(),
                                      FmtInt(sum.rank_full).c_str());
        KV({
            { "layer.id",  id,    "accent" },
            { "resid_pre", shape, "" },
            { "attn_out",  shape, "" },
            { "mlp_out",   shape, "" },
            { "||attn||",  FmtFloat(sum.attn_out_norm, "%.3f"), "accent" },
            { "||mlp||",   FmtFloat(sum.mlp_out_norm,  "%.3f"), "warn"   },
            { "cos(L-1)",  FmtFloat(sum.cos_prev,      "%.4f"), "good"   },
            { "kurtosis",  FmtFloat(sum.kurtosis,      "%.2f"), "" },
            { "rank_eff",  rank,                                "" },
        });
    }
    if (auto sec = BeginSection("resid_post · histogram", false, "μ=? σ=?")) {
        // [DATA HOOK] Model::getWeightHistogram(name, bins) — distribution
        // of residual_post values for the selected layer.  Engine names the
        // tensor "blocks.<L>.resid_post" or similar.
        char name[64]; std::snprintf(name, sizeof name, "blocks.%d.resid_post", s.activeLayer);
        const auto bins = m.getWeightHistogram(name, 40);
        const LensAnnotation an[] = {
            { 0.50f, "μ",         Sty().accent },
            { 0.78f, "top-1 dir", Sty().warn   },
        };
        ActivationHistogram(bins, ImGui::GetContentRegionAvail().x - 8, 84, Sty().info,
                            std::span{an, std::size(an)});
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
        ImGui::TextUnformatted("[engine: provide min/max/zero-frac in TensorStats]");
        ImGui::PopStyleColor();
    }
    char hb[8]; std::snprintf(hb, sizeof hb, "%dh", s.model.nHeads);
    if (auto sec = BeginSection("Per-head attention norms", false, hb)) {
        for (int h = 0; h < s.model.nHeads; ++h) {
            ImGui::PushID(h);
            char hk[16]; std::snprintf(hk, sizeof hk, "%d.%d", s.activeLayer, h);
            const bool ab = s.ablatedHeads.contains(hk);
            const bool pr = s.probedHeads.contains(hk);
            // [DATA HOOK] Model::getHeadNorm(layer, head) — magnitude of the
            // head's contribution at the active layer (also drives arch-map
            // tinting).
            const float v = m.getHeadNorm(s.activeLayer, h);
            ImGui::PushStyleColor(ImGuiCol_Button,        h == s.activeHead ? Sty().accent_bg : Sty().bg_input);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, Sty().bg_input_hover);
            ImGui::PushStyleColor(ImGuiCol_Text,          ab ? Sty().bad : (pr ? Sty().accent : Sty().text));
            char hl[24];
            if (std::isnan(v)) std::snprintf(hl, sizeof hl, "h%d  —", h);
            else                std::snprintf(hl, sizeof hl, "h%d  %.2f", h, v * 1.5f);
            if (ImGui::Button(hl, ImVec2(80, 0))) s.setActiveHead(h);
            ImGui::PopStyleColor(3);
            ImGui::PopID();
            if ((h + 1) % 4 != 0) ImGui::SameLine();
        }
        ImGui::NewLine();
    }
    if (auto sec = BeginSection("MLP feature activations", false, "top 8")) {
        // [DATA HOOK] Model::getMlpFeatures(layer, k) — top-k MLP feature
        // indices and their signed activations for this token.
        const auto feats = m.getMlpFeatures(s.activeLayer, 8);
        if (feats.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
            ImGui::TextUnformatted("// no MLP feature activations available");
            ImGui::PopStyleColor();
        } else if (ImGui::BeginTable("##feat", 3, ImGuiTableFlags_BordersInner)) {
            ImGui::TableSetupColumn("idx",     ImGuiTableColumnFlags_WidthFixed, 46);
            ImGui::TableSetupColumn("preview");
            ImGui::TableSetupColumn("value",   ImGuiTableColumnFlags_WidthFixed, 60);
            ImGui::TableHeadersRow();
            float maxv = 0.001f; for (const auto& f : feats) maxv = std::max(maxv, std::abs(f.value));
            for (const auto& f : feats) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn(); ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
                ImGui::Text("f%d", f.idx); ImGui::PopStyleColor();
                ImGui::TableNextColumn(); Bar(std::abs(f.value) / maxv, 140, 6, Sty().warn);
                ImGui::TableNextColumn(); ImGui::Text("%+.2f", double(f.value));
            }
            ImGui::EndTable();
        }
    }
    ImGui::EndChild();
}

void DrawTokenStrip(AppState& s, Model& m) {
    DrawTitleBar("token_strip", "▦", nullptr, "tokstrip");
    if (!ImGui::BeginChild("##ts_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    const std::size_t n = s.sampleTokens.size();
    std::vector<std::string_view> tk; tk.reserve(n);
    for (auto& t : s.sampleTokens) tk.emplace_back(t);

    if (auto sec = BeginSection("cross-entropy loss per token", false, "L?")) {
        // [DATA HOOK] Model::getTokenLossPerToken(layer) — per-token loss
        // at the active layer.  Engine source: lm_head logits vs. the
        // teacher-forced next-token target.
        const auto loss = m.getTokenLossPerToken(s.activeLayer);
        if (loss.empty() || tk.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
            ImGui::TextUnformatted("// no per-token loss available");
            ImGui::PopStyleColor();
        } else {
            TokenGutter(tk, std::span{loss}, HeatColor);
        }
    }
    if (auto sec = BeginSection("surprisal Δ vs base run", false, "diff")) {
        // [DATA HOOK] Model::getSurprisalDelta() — per-token surprisal
        // delta vs the baseline (un-ablated) forward pass.  Engine should
        // run two passes (ablated + base) and subtract.
        const auto diff = m.getSurprisalDelta();
        if (diff.empty() || tk.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
            ImGui::TextUnformatted("// no surprisal delta available (need baseline)");
            ImGui::PopStyleColor();
        } else {
            TokenGutter(tk, std::span{diff}, [](float v) { return DivergeColor(v * 2.0f - 1.0f); });
        }
    }
    ImGui::EndChild();
}

void DrawProbeControls(AppState& s, Model& m) {
    DrawTitleBar("probe_controls", "◎", nullptr, "probe-ctrl");
    if (!ImGui::BeginChild("##pc_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    char nbuf[24]; std::snprintf(nbuf, sizeof nbuf, "%zu",
                                  s.probedHeads.size() + s.probedComponents.size());
    if (auto sec = BeginSection("Active probes", true, nbuf)) {
        // [DATA HOOK] Model::getActiveProbes() — probes currently attached
        // to the model (engine-side, not the UI's pending probedHeads /
        // probedComponents sets which represent user intent).
        const auto probes = m.getActiveProbes();
        if (probes.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
            ImGui::TextUnformatted("// no active probes");
            ImGui::PopStyleColor();
        } else if (ImGui::BeginTable("##probes", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInner)) {
            ImGui::TableSetupColumn("type"); ImGui::TableSetupColumn("name");
            ImGui::TableSetupColumn("acc", ImGuiTableColumnFlags_WidthFixed, 60);
            ImGui::TableHeadersRow();
            for (const auto& p : probes) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn(); Pill(p.type.c_str(), p.type_tone.c_str(), p.type_solid);
                ImGui::TableNextColumn(); ImGui::Text("%s", p.name.c_str());
                ImGui::TableNextColumn();
                ImGui::PushStyleColor(ImGuiCol_Text, Sty().good);
                ImGui::TextUnformatted(FmtFloat(p.accuracy, "%.2f").c_str());
                ImGui::PopStyleColor();
            }
            ImGui::EndTable();
        }
    }
    if (auto sec = BeginSection("Steering vector", true, "ON")) {
        // [DATA HOOK] Model::getSteering() — current steering-vector config.
        // Engine writes here when a steering vector is loaded (typically
        // via `add_hook` on the residual stream).
        const auto st = m.getSteering();
        if (!st.active) {
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
            ImGui::TextUnformatted("// no steering vector loaded");
            ImGui::PopStyleColor();
        } else {
            char alpha[16]; std::snprintf(alpha, sizeof alpha, "%+.2f", double(st.alpha));
            KV({
                { "source", Or(st.source), "" },
                { "layer",  Or(st.layer),  "accent" },
                { "α",      alpha,         "warn" },
                { "cos sim",FmtFloat(st.cos_sim, "%.3f"), "good" },
            });
            // The slider here is purely UI intent — committed back to the
            // engine via [DATA HOOK] Model::setSteering(alpha) (TBD).
            static float alpha_ui = st.alpha;
            ImSliderF("##alpha", alpha_ui, -3.0f, 3.0f, "%.2f",
                      ImGui::GetContentRegionAvail().x - 8);
        }
    }
    ImGui::EndChild();
}

}  // namespace

// Layout: resid | center(fwd / strip) | right(probe / pctrl + raw tabbed)
void BuildInferenceLayout(ImGuiID dock_id) {
    ImGui::DockBuilderRemoveNode(dock_id);
    ImGui::DockBuilderAddNode(dock_id, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dock_id, ImGui::GetMainViewport()->Size);

    ImGuiID resid_n, rest1, rest2, right_n, probe_n, pctrl_n, fwd_n, strip_n;
    ImGui::DockBuilderSplitNode(dock_id, ImGuiDir_Left,  0.20f, &resid_n, &rest1);
    ImGui::DockBuilderSplitNode(rest1,   ImGuiDir_Right, 0.24f, &right_n, &rest2);
    ImGui::DockBuilderSplitNode(right_n, ImGuiDir_Down,  0.45f, &pctrl_n, &probe_n);
    ImGui::DockBuilderSplitNode(rest2,   ImGuiDir_Down,  0.30f, &strip_n, &fwd_n);

    ImGui::DockBuilderDockWindow("inf.resid", resid_n);
    ImGui::DockBuilderDockWindow("inf.fwd",   fwd_n);
    ImGui::DockBuilderDockWindow("inf.strip", strip_n);
    ImGui::DockBuilderDockWindow("inf.probe", probe_n);
    ImGui::DockBuilderDockWindow("inf.pctrl", pctrl_n);
    ImGui::DockBuilderDockWindow("inf.raw",   pctrl_n);     // tabbed with pctrl
    ImGui::DockBuilderFinish(dock_id);
}

void SubmitInferencePanels(AppState& s, Model& m) {
    if (!s.hasModel()) {
        if (ImGui::Begin("inf.resid", nullptr, ImGuiWindowFlags_NoTitleBar)) {
            EmptyStatePlaceholder("// no model loaded — open a checkpoint via File ▸ Open");
        }
        ImGui::End();
        return;
    }
    if (ImGui::Begin("inf.resid", nullptr, ImGuiWindowFlags_NoTitleBar)) DrawResidualFlow (s, m);
    ImGui::End();
    if (ImGui::Begin("inf.fwd",   nullptr, ImGuiWindowFlags_NoTitleBar)) DrawForwardPass  (s, m);
    ImGui::End();
    if (ImGui::Begin("inf.strip", nullptr, ImGuiWindowFlags_NoTitleBar)) DrawTokenStrip   (s, m);
    ImGui::End();
    if (ImGui::Begin("inf.probe", nullptr, ImGuiWindowFlags_NoTitleBar)) DrawProbePanel   (s, m);
    ImGui::End();
    if (ImGui::Begin("inf.pctrl", nullptr, ImGuiWindowFlags_NoTitleBar)) DrawProbeControls(s, m);
    ImGui::End();
    if (s.showRaw) {
        if (ImGui::Begin("inf.raw", nullptr, ImGuiWindowFlags_NoTitleBar)) {
            DrawTitleBar("raw_tensor", "0x", "fp16", "raw");
            // Activations are bounded by d_model — even for big models
            // they fit comfortably in memory, so a single getActivation
            // call still suffices.  Wrap in HexViewVirtual anyway so the
            // ImGuiListClipper skips offscreen rows when d_model is large
            // (e.g. 8192 → 2048 rows of 4 cols, mostly clipped).
            constexpr int kCols = 4;
            const int total = s.model.dModel;
            if (total <= 0) {
                ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
                ImGui::TextUnformatted("// no activation data");
                ImGui::PopStyleColor();
            } else {
                HexViewVirtual(total / kCols, kCols, /*baseAddr=*/0, HexMode::Fp16,
                    [&](int first, int n) {
                        // [DATA HOOK] Model::getActivation(layer, kind, n)
                        // currently fetches from offset 0.  Real engine
                        // should support `(layer, kind, offset, n)` so the
                        // page is true mid-tensor; until then we fetch the
                        // full slice and slice it CPU-side.
                        const auto full = m.getActivation(s.activeLayer, 0, total);
                        const int  off  = first * kCols;
                        const int  end  = std::min(off + n * kCols, int(full.size()));
                        return std::vector<float>(
                            full.begin() + std::min(off, int(full.size())),
                            full.begin() + std::max(off, end));
                    });
            }
        }
        ImGui::End();
    }
}

}  // namespace llob
