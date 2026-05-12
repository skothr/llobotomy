// Attention workspace — head browser | full attention heatmap + qkv inspector,
// stats + ablation queue at bottom.  See HANDOFF §3.3.

#include "workspaces/workspaces.hpp"

#include "appstate.hpp"
#include "model/model.hpp"
#include "style.hpp"
#include "ui/chrome.hpp"
#include "ui/widgets.hpp"

#include <imgui.h>

#include <cstdio>
#include <string>
#include <vector>

namespace llob {

namespace {

HeadBias BiasFor(int L, int H) {
    static const HeadBias bm[] = { HeadBias::Diag, HeadBias::Prev, HeadBias::First,
                                    HeadBias::Broad, HeadBias::Induction };
    return bm[(L + H) % 5];
}
const char* BiasName(HeadBias b) {
    switch (b) {
        case HeadBias::Diag: return "diag";
        case HeadBias::Prev: return "prev";
        case HeadBias::First:return "first";
        case HeadBias::Broad:return "broad";
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
    ImGui::SameLine();
    if (ImGui::SmallButton("◀")) s.setActiveLayer(s.activeLayer - 1);
    ImGui::SameLine();
    if (ImGui::SmallButton("▶")) s.setActiveLayer(s.activeLayer + 1);

    ImGui::Separator();
    int per_row = 3;
    for (int h = 0; h < s.model.nHeads; ++h) {
        ImGui::PushID(h);
        const HeadBias b = BiasFor(s.activeLayer, h);
        const auto data  = m.getAttentionPattern(s.activeLayer, h, 12, b);
        char lbl[64]; std::snprintf(lbl, sizeof lbl, "L%d.h%d · %s", s.activeLayer, h, BiasName(b));
        char hk[16];  std::snprintf(hk,  sizeof hk,  "%d.%d", s.activeLayer, h);
        if (AttentionThumb(data, 12, 62, lbl, h == s.activeHead, s.ablatedHeads.contains(hk))) {
            s.setActiveHead(h);
        }
        ImGui::PopID();
        if ((h + 1) % per_row != 0) ImGui::SameLine();
    }
    ImGui::EndChild();
}

void DrawAttnMain(AppState& s, Model& m) {
    char title[32]; std::snprintf(title, sizeof title, "attn[L%02d.h%d]", s.activeLayer, s.activeHead);
    const HeadBias bias = BiasFor(s.activeLayer, s.activeHead);
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
    const auto data = m.getAttentionPattern(s.activeLayer, s.activeHead, 24, bias);
    std::vector<std::string_view> tks; tks.reserve(s.sampleTokens.size());
    for (auto& t : s.sampleTokens) tks.emplace_back(t);
    AttnHeatmapOpts o{}; o.cellSize = 20; o.maxCells = 24; o.selected = s.activeToken; o.causal = true;
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
        std::snprintf(tt, sizeof tt, "token: ??");
    if (auto sec = BeginSection(tt, true)) {
        KV({
            { "||q||",       "0.421", "accent" },
            { "||k||",       "0.388", "info"   },
            { "||v||",       "0.623", "warn"   },
            { "attn → bos",  "0.04",  "" },
            { "attn → self", "0.31",  "accent" },
            { "attn → prev", "0.42",  "accent" },
        });
        EndSection(sec);
    }
    auto qkv = m.getActivation(s.activeLayer * 100 + s.activeHead, 0, 64);
    if (auto sec = BeginSection("Q vector (slice)", false, "d=64")) {
        MiniGrid(qkv, 16, 14); EndSection(sec);
    }
    auto kv = m.getActivation(s.activeLayer * 100 + s.activeHead + 7, 1, 64);
    if (auto sec = BeginSection("K vector (slice)", false, "d=64")) {
        MiniGrid(kv, 16, 14); EndSection(sec);
    }
    auto vv = m.getActivation(s.activeLayer * 100 + s.activeHead + 13, 2, 64);
    if (auto sec = BeginSection("V vector (slice)", false, "d=64")) {
        MiniGrid(vv, 16, 14); EndSection(sec);
    }
    ImGui::EndChild();
}

void DrawHeadStats(AppState&, Model&) {
    DrawTitleBar("head_stats", "∑", nullptr, "hs");
    if (!ImGui::BeginChild("##hs_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    if (auto sec = BeginSection("Pattern statistics", true)) {
        const struct { const char* n; const char* v; float bar; ImU32 col; } rows[] = {
            { "entropy / token",       "1.14 nats",    0.45f, 0 },
            { "diagonal weight",       "0.62",         0.62f, 0 },
            { "previous-token bias",   "0.41",         0.41f, 0 },
            { "BOS attention",         "0.07",         0.07f, 0 },
            { "induction score",       "0.78",         0.78f, 0 },
        };
        if (ImGui::BeginTable("##stats", 3, ImGuiTableFlags_BordersInner)) {
            ImGui::TableSetupColumn("metric");
            ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthFixed, 80);
            ImGui::TableSetupColumn("vs corpus");
            ImGui::TableHeadersRow();
            for (auto& r : rows) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn(); ImGui::TextUnformatted(r.n);
                ImGui::TableNextColumn(); ImGui::TextUnformatted(r.v);
                ImGui::TableNextColumn(); Bar(r.bar, 100, 6, Sty().accent);
            }
            ImGui::EndTable();
        }
        EndSection(sec);
    }
    ImGui::EndChild();
}

void DrawAblQueue(AppState& s, Model&) {
    char nb[16]; std::snprintf(nb, sizeof nb, "%zu", s.ablatedHeads.size());
    DrawTitleBar("ablation_queue", "X", nullptr, "abl");
    if (!ImGui::BeginChild("##abl_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    if (auto sec = BeginSection("Active ablations", true, nb)) {
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
        EndSection(sec);
    }
    if (auto sec = BeginSection("Patch source", true)) {
        char tp[16]; std::snprintf(tp, sizeof tp, "%d", s.activeToken);
        KV({
            { "source prompt", "\"The ship sailed...\"", "" },
            { "target pos",    tp,           "accent" },
            { "component",     "attn_out",   "info" },
            { "effect",        "+0.31 nats", "warn" },
        });
        EndSection(sec);
    }
    ImGui::EndChild();
}

}  // namespace

void DrawAttentionWorkspace(AppState& s, Model& m) {
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
