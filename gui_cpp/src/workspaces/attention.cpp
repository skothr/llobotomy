// Attention workspace — head browser | full attention heatmap + qkv inspector,
// stats + ablation queue at bottom.  See HANDOFF §3.3.
//
// All values come from the Model interface; nothing in this file is
// hardcoded.  The data hooks each section needs are documented inline as
// `// [DATA HOOK]` comments naming the Model::* method that supplies them.

#include "workspaces/workspaces.hpp"

#include "appstate.hpp"
#include "model/model.hpp"
#include "style.hpp"
#include "ui/chrome.hpp"
#include "ui/colormap.hpp"
#include "ui/fmt.hpp"
#include "ui/widgets.hpp"

#include <imgui.h>

#include <cstdio>
#include <string>
#include <vector>

namespace llob {

namespace {

const char* BiasName(HeadBias b) {
    switch (b) {
        case HeadBias::Diag:      return "diag";
        case HeadBias::Prev:      return "prev";
        case HeadBias::First:     return "first";
        case HeadBias::Broad:     return "broad";
        case HeadBias::Induction: return "induction";
    }
    return "?";
}

void DrawHeadBrowser(AppState& s, Model& m) {
    DrawTitleBar("head_browser", "⊞", nullptr, "hb");
    if (!ImGui::BeginChild("##hb_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }

    ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
    ImGui::TextUnformatted("layer"); ImGui::PopStyleColor(); ImGui::SameLine();
    ImGui::SetNextItemWidth(70);
    int L = s.activeLayer;
    if (ImGui::InputInt("##L", &L, 1, 1)) s.setActiveLayer(L);
    ImGui::SameLine(); if (ImGui::SmallButton("◀")) s.setActiveLayer(s.activeLayer - 1);
    ImGui::SameLine(); if (ImGui::SmallButton("▶")) s.setActiveLayer(s.activeLayer + 1);
    ImGui::Separator();

    constexpr int kThumbN  = 12;
    constexpr float kThumb = 62;
    const int per_row = 3;
    for (int h = 0; h < s.model.nHeads; ++h) {
        ImGui::PushID(h);
        // [DATA HOOK] Model::getHeadBias(layer, head) — cosmetic pattern label
        // ("induction"/"prev"/...) for the thumbnail; also drives color hints.
        const HeadBias b = m.getHeadBias(s.activeLayer, h);
        // [DATA HOOK] Model::getAttentionPattern(layer, head, seqLen, bias)
        // — N×N causal attention matrix (here at downsampled N=12).
        const auto data  = m.getAttentionPattern(s.activeLayer, h, kThumbN, b);
        char lbl[64]; std::snprintf(lbl, sizeof lbl, "L%d.h%d · %s",
                                     s.activeLayer, h, BiasName(b));
        char hk[16];  std::snprintf(hk,  sizeof hk,  "%d.%d", s.activeLayer, h);
        if (AttentionThumb(data, kThumbN, kThumb, lbl,
                           h == s.activeHead, s.ablatedHeads.contains(hk))) {
            s.setActiveHead(h);
        }
        ImGui::PopID();
        if ((h + 1) % per_row != 0) ImGui::SameLine();
    }
    ImGui::EndChild();
}

void DrawAttnMain(AppState& s, Model& m) {
    char title[32]; std::snprintf(title, sizeof title, "attn[L%02d.h%d]",
                                   s.activeLayer, s.activeHead);
    const HeadBias bias = m.getHeadBias(s.activeLayer, s.activeHead);
    char flag[32];  std::snprintf(flag, sizeof flag, "pattern: %s", BiasName(bias));

    DrawTitleBar(title, "▦", flag, "attn-main", [&] {
        char hk[16]; std::snprintf(hk, sizeof hk, "%d.%d", s.activeLayer, s.activeHead);
        const bool ab = s.ablatedHeads.contains(hk);
        ImGui::PushStyleColor(ImGuiCol_Button, ab ? Sty().bad_bg : Sty().bg_titlebar);
        if (ImGui::SmallButton(ab ? "X restore" : "X ablate")) s.toggleAblate(hk);
        ImGui::PopStyleColor();
        ImGui::SameLine();
        if (ImGui::SmallButton("(pin)")) {}
    });
    if (!ImGui::BeginChild("##attn_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }

    // [DATA HOOK] Model::getAttentionPattern(layer, head, seqLen, bias)
    // — full-resolution pattern for the heatmap; same source as the thumbs,
    //   queried at the live sequence length.
    const auto data = m.getAttentionPattern(
        s.activeLayer, s.activeHead, int(s.sampleTokens.size()), bias);
    std::vector<std::string_view> tks; tks.reserve(s.sampleTokens.size());
    for (auto& t : s.sampleTokens) tks.emplace_back(t);
    AttnHeatmapOpts o{}; o.cellSize = 20; o.maxCells = 24;
    o.selected = s.activeToken; o.causal = true;
    int clicked = AttentionHeatmap(data, tks, o);
    if (clicked >= 0) s.setActiveToken(clicked);
    ImGui::EndChild();
}

void DrawQKV(AppState& s, Model& m) {
    DrawTitleBar("qkv_inspector", "⌬", nullptr, "qkv");
    if (!ImGui::BeginChild("##qkv_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }

    char tt[80];
    if (s.activeToken < int(s.sampleTokens.size()))
        std::snprintf(tt, sizeof tt, "token: \"%s\" [pos %d]",
                       s.sampleTokens[s.activeToken].c_str(), s.activeToken);
    else
        std::snprintf(tt, sizeof tt, "token: ?? [pos %d]", s.activeToken);

    if (auto sec = BeginSection(tt, true)) {
        // [DATA HOOK] Model::getQKVStats(layer, head, token) — Q/K/V vector
        // norms + selected-token attention fractions ({to_bos, to_self,
        // to_prev}) for the chosen attention head + token.
        const auto q = m.getQKVStats(s.activeLayer, s.activeHead, s.activeToken);
        const std::string vq    = FmtFloat(q.q_norm, "%.3f");
        const std::string vk    = FmtFloat(q.k_norm, "%.3f");
        const std::string vv    = FmtFloat(q.v_norm, "%.3f");
        const std::string vbos  = FmtFloat(q.attn_to_bos,  "%.2f");
        const std::string vself = FmtFloat(q.attn_to_self, "%.2f");
        const std::string vprev = FmtFloat(q.attn_to_prev, "%.2f");
        KV({
            { "||q||",       vq,    "accent" },
            { "||k||",       vk,    "info"   },
            { "||v||",       vv,    "warn"   },
            { "attn → bos",  vbos,  "" },
            { "attn → self", vself, "accent" },
            { "attn → prev", vprev, "accent" },
        });
    }

    // [DATA HOOK] Model::getActivation(layer, kind, n) — Q/K/V slice vectors
    // (each of length d_head).  `kind` indexes the tensor: 0=q, 1=k, 2=v.
    const auto qv = m.getActivation(s.activeLayer * 100 + s.activeHead, 0, 64);
    const auto kv = m.getActivation(s.activeLayer * 100 + s.activeHead, 1, 64);
    const auto vv = m.getActivation(s.activeLayer * 100 + s.activeHead, 2, 64);
    if (auto sec = BeginSection("Q vector (slice)", false, "d=64")) MiniGrid(qv, 16, 14);
    if (auto sec = BeginSection("K vector (slice)", false, "d=64")) MiniGrid(kv, 16, 14);
    if (auto sec = BeginSection("V vector (slice)", false, "d=64")) MiniGrid(vv, 16, 14);
    ImGui::EndChild();
}

void DrawHeadStats(AppState& s, Model& m) {
    DrawTitleBar("head_stats", "∑", nullptr, "hs");
    if (!ImGui::BeginChild("##hs_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    if (auto sec = BeginSection("Pattern statistics", true)) {
        // [DATA HOOK] Model::getHeadStats(layer, head) — pre-computed pattern
        // statistics for the active head: entropy / token, diagonal weight,
        // previous-token bias, BOS attention, induction score, ...
        const auto rows = m.getHeadStats(s.activeLayer, s.activeHead);
        if (rows.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
            ImGui::TextUnformatted("// no head stats available");
            ImGui::PopStyleColor();
        } else if (ImGui::BeginTable("##stats", 3, ImGuiTableFlags_BordersInner)) {
            ImGui::TableSetupColumn("metric");
            ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthFixed, 80);
            ImGui::TableSetupColumn("vs corpus");
            ImGui::TableHeadersRow();
            for (const auto& r : rows) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn(); ImGui::TextUnformatted(r.metric.c_str());
                ImGui::TableNextColumn();
                ImGui::PushStyleColor(ImGuiCol_Text,
                    r.tone && *r.tone ? ToneColor(r.tone) : Sty().text);
                ImGui::TextUnformatted(r.value_str.c_str());
                ImGui::PopStyleColor();
                ImGui::TableNextColumn();
                Bar(r.vs_corpus, 100, 6, Sty().accent);
            }
            ImGui::EndTable();
        }
    }
    ImGui::EndChild();
}

void DrawAblQueue(AppState& s, Model& m) {
    char nb[16]; std::snprintf(nb, sizeof nb, "%zu", s.ablatedHeads.size());
    DrawTitleBar("ablation_queue", "X", nullptr, "abl");
    if (!ImGui::BeginChild("##abl_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    if (auto sec = BeginSection("Active ablations", true, nb)) {
        // Source of truth for the ablation set is AppState (UI-mutated).
        // The engine subscribes via toggleAblate's pushLog hook and applies
        // the effect on the next forward pass.
        if (s.ablatedHeads.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
            ImGui::TextUnformatted("// no ablations active");
            ImGui::PopStyleColor();
        } else {
            std::vector<std::string> snapshot(s.ablatedHeads.begin(), s.ablatedHeads.end());
            for (const auto& h : snapshot) {
                ImGui::PushID(h.c_str());
                ImGui::PushStyleColor(ImGuiCol_Text, Sty().bad);
                char lbl[24]; std::snprintf(lbl, sizeof lbl, "L%s.h%s",
                    h.substr(0, h.find('.')).c_str(),
                    h.substr(h.find('.') + 1).c_str());
                ImGui::TextUnformatted(lbl); ImGui::PopStyleColor();
                ImGui::SameLine(); ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
                ImGui::TextUnformatted("zero"); ImGui::PopStyleColor();
                ImGui::SameLine();
                if (ImGui::SmallButton("X")) s.toggleAblate(h);
                ImGui::PopID();
            }
        }
    }
    if (auto sec = BeginSection("Patch source", true)) {
        // [DATA HOOK] Model::getPatchSource() — describes the current
        // activation-patching source: a separate prompt whose activations
        // are spliced into the current run at `target_pos.component`.
        const auto p = m.getPatchSource();
        const std::string tp = FmtInt(p.target_pos);
        char eff[24]; std::snprintf(eff, sizeof eff, "%+.2f nats", double(p.effect_nats));
        const std::string effs = std::isnan(p.effect_nats) ? std::string("—") : eff;
        KV({
            { "source prompt", Or(p.source_prompt), "" },
            { "target pos",    tp,                  "accent" },
            { "component",     Or(p.component),     "info" },
            { "effect",        effs,                "warn" },
        });
    }
    ImGui::EndChild();
}

}  // namespace

void DrawAttentionWorkspace(AppState& s, Model& m) {
    if (!s.hasModel()) {
        EmptyStatePlaceholder("// no model loaded — open a checkpoint via File ▸ Open");
        return;
    }
    const float W = ImGui::GetContentRegionAvail().x;
    const float H = ImGui::GetContentRegionAvail().y;
    const float gap = 1.0f;
    const float left_w  = 260.0f;
    const float right_w = 320.0f;
    const float center_w= std::max(200.0f, W - left_w - right_w - 2 * gap);
    const float bot_h   = std::min(240.0f, H * 0.32f);
    const float top_h   = H - bot_h - gap;

    ImGui::BeginChild("##at_left", { left_w, H }, ImGuiChildFlags_Borders);
    DrawHeadBrowser(s, m); ImGui::EndChild(); ImGui::SameLine(0, gap);

    ImGui::BeginChild("##at_center", { center_w, H });
    ImGui::BeginChild("##at_main", { center_w, top_h }, ImGuiChildFlags_Borders);
    DrawAttnMain(s, m); ImGui::EndChild();
    ImGui::BeginChild("##at_stats", { center_w, bot_h }, ImGuiChildFlags_Borders);
    DrawHeadStats(s, m); ImGui::EndChild();
    ImGui::EndChild(); ImGui::SameLine(0, gap);

    ImGui::BeginChild("##at_right", { right_w, H });
    ImGui::BeginChild("##at_qkv", { right_w, top_h }, ImGuiChildFlags_Borders);
    DrawQKV(s, m); ImGui::EndChild();
    ImGui::BeginChild("##at_abl", { right_w, bot_h }, ImGuiChildFlags_Borders);
    DrawAblQueue(s, m); ImGui::EndChild();
    ImGui::EndChild();
}

}  // namespace llob
