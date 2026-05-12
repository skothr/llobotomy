// Architecture workspace — outline tree | arch map | inspector | [raw hex],
// with param-distribution + ops in the bottom row.
//
// The arch map is a hand-drawn `ImDrawList` view of every transformer block
// as a horizontal lane.  Each block has three display states (skipped /
// collapsed / expanded) and every clickable subregion is an `InvisibleButton`
// for hit-testing per the interaction contract in HANDOFF.md §6.

#include "workspaces/workspaces.hpp"

#include "appstate.hpp"
#include "model/model.hpp"
#include "style.hpp"
#include "ui/chrome.hpp"
#include "ui/colormap.hpp"
#include "ui/fmt.hpp"
#include "ui/widgets.hpp"

#include <imgui.h>
#include <imgui_internal.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace llob {

namespace {

constexpr float kBlockH_Skipped   = 22.0f;
constexpr float kBlockH_Collapsed = 32.0f;
constexpr float kBlockH_Expanded  = 132.0f;
constexpr float kLayerGap         = 4.0f;

struct Cols {
    struct C { float x, w; };
    C label, norm1, qkv, heads, wo, add1, norm2, gateup, silu, wout, add2;
    int  nHeads;
    float headCellW = 0, headCellGap = 1;
    float blockBodyW = 0, blockW = 0;
};

Cols ComputeCols(const ModelInfo& mi) {
    Cols c{};
    c.nHeads      = mi.nHeads;
    c.headCellW   = std::max(8.0f, std::min(22.0f, std::floor(280.0f / float(mi.nHeads))));
    c.headCellGap = 1.0f;

    const float dScale = std::max(1.0f, std::log2(float(mi.dModel) / 384.0f) * 0.18f + 1.0f);
    auto cw = [dScale](float base) { return std::round(base * dScale); };
    const float pad_l = 16.0f;
    c.label  = { pad_l,           64 };
    c.norm1  = { pad_l + 76,      cw(64) };
    c.qkv    = { 0,               cw(56) };
    c.heads  = { 0,               c.nHeads * (c.headCellW + c.headCellGap) };
    c.wo     = { 0,               cw(56) };
    c.add1   = { 0,               18 };
    c.norm2  = { 0,               cw(64) };
    c.gateup = { 0,               cw(72) };
    c.silu   = { 0,               22 };
    c.wout   = { 0,               cw(72) };
    c.add2   = { 0,               18 };

    float x = c.norm1.x + c.norm1.w + 14;
    c.qkv.x    = x; x += c.qkv.w    + 12;
    c.heads.x  = x; x += c.heads.w  + 10;
    c.wo.x     = x; x += c.wo.w     + 8;
    c.add1.x   = x; x += c.add1.w   + 14;
    c.norm2.x  = x; x += c.norm2.w  + 10;
    c.gateup.x = x; x += c.gateup.w + 4;
    c.silu.x   = x; x += c.silu.w   + 4;
    c.wout.x   = x; x += c.wout.w   + 14;
    c.add2.x   = x; x += c.add2.w   + 16;
    c.blockBodyW = x;
    c.blockW     = x;
    return c;
}

bool LayerHasKey(const std::unordered_set<std::string>& set, int L, std::string_view comp = {}) {
    char buf[64];
    if (comp.empty()) {
        std::snprintf(buf, sizeof buf, "%d", L);
        for (const auto& k : set) if (k.starts_with(std::string(buf) + ".")) return true;
        return false;
    }
    std::snprintf(buf, sizeof buf, "%d.%.40s", L, std::string(comp).c_str());
    return set.contains(buf);
}

int CountForLayer(const std::unordered_set<std::string>& set, int L) {
    char prefix[16]; std::snprintf(prefix, sizeof prefix, "%d.", L);
    int n = 0; for (const auto& k : set) if (k.starts_with(prefix)) ++n;
    return n;
}

// ── Outline tree ───────────────────────────────────────────────────────────
void DrawOutline(AppState& s, Model& m) {
    (void)m;
    DrawTitleBar("model_outline", "≡", nullptr, "outline");
    if (!ImGui::BeginChild("##outline_body", ImVec2(0, 0), ImGuiChildFlags_None,
                           ImGuiWindowFlags_HorizontalScrollbar)) {
        ImGui::EndChild(); return;
    }

    ImGui::PushStyleColor(ImGuiCol_Text, Sty().accent);
    ImGui::Text("v model");
    ImGui::PopStyleColor();
    ImGui::Indent(12.0f);
    ImGui::PushStyleColor(ImGuiCol_Text, Sty().magenta);
    ImGui::Text("tok_embed [%d, %d]", s.model.vocab, s.model.dModel);
    ImGui::PopStyleColor();
    ImGui::PushStyleColor(ImGuiCol_Text, Sty().yellow);
    ImGui::Text("pos_embed (RoPE)");
    ImGui::PopStyleColor();
    ImGui::Text("v blocks [%d]", s.model.nLayers);

    for (int L = 0; L < s.model.nLayers; ++L) {
        ImGui::PushID(L);
        const bool active   = (L == s.activeLayer);
        const bool expanded = s.expandedLayers.contains(L);
        const bool skipped  = s.skippedLayers.contains(L);

        char label[48];
        std::snprintf(label, sizeof label, "%s block_%02d   ·%dh%s",
                      skipped ? "x" : (expanded ? "v" : ">"),
                      L, s.model.nHeads, skipped ? "  SKIP" : "");

        TreeRowFlags f{};
        f.indent_px  = 4;
        f.active     = active;
        f.strikethru = skipped;
        f.fg         = skipped ? Sty().bad : (active ? Sty().accent : Sty().text_muted);
        if (TreeRow(label, label, f))     s.setActiveLayer(L);
        if (WasRightClicked())            s.toggleLayerSkip(L);

        // Chevron click — also expand/collapse.
        if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Middle)) {
            if (!skipped) s.toggleLayerExpand(L);
        }

        if (expanded && !skipped) {
            auto compRow = [&](const char* lbl, std::string_view ref, ImU32 color) {
                const bool ab = LayerHasKey(s.ablatedComponents, L, ref);
                const bool pr = LayerHasKey(s.probedComponents,  L, ref);
                std::string display = std::string("  · ") + lbl;
                if (pr) display += "   o";
                if (ab) display += "   x";
                TreeRowFlags ff{};
                ff.indent_px = 36;
                ff.fg        = ab ? Sty().bad : color;
                ff.strikethru= ab;
                if (TreeRow(lbl, display.c_str(), ff)) s.toggleProbeComp(L, ref);
                if (WasRightClicked())                s.toggleAblateComp(L, ref);
            };

            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_dim);
            ImGui::Indent(20.0f); ImGui::TextUnformatted("v attn"); ImGui::Unindent(20.0f);
            ImGui::PopStyleColor();
            compRow("rmsnorm_1",  "norm1", Sty().text_muted);
            compRow("W_Q [d->d]", "W_Q",   Sty().info);
            compRow("W_K [d->d]", "W_K",   Sty().info);
            compRow("W_V [d->d]", "W_V",   Sty().info);
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_dim);
            ImGui::Indent(28.0f);
            ImGui::Text("v heads x%d", s.model.nHeads);
            ImGui::Unindent(28.0f);
            ImGui::PopStyleColor();

            ImGui::Indent(36.0f);
            for (int h = 0; h < s.model.nHeads; ++h) {
                ImGui::PushID(h);
                char hk[16]; std::snprintf(hk, sizeof hk, "%d.%d", L, h);
                const bool ab = s.ablatedHeads.contains(hk);
                const bool pr = s.probedHeads.contains(hk);
                const ImU32 fg = ab ? Sty().bad : (pr ? Sty().accent : Sty().info);
                ImGui::PushStyleColor(ImGuiCol_Text,        fg);
                ImGui::PushStyleColor(ImGuiCol_Button,      ab ? Sty().bad_bg : (pr ? Sty().accent_bg : Sty().bg_input));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ab ? Sty().bad_bg : Sty().bg_input_hover);
                char hl[8]; std::snprintf(hl, sizeof hl, "h%d", h);
                if (ImGui::SmallButton(hl)) s.toggleProbe(L, h);
                if (WasRightClicked())      s.toggleAblate(hk);
                ImGui::PopStyleColor(3);
                ImGui::PopID();
                if (h + 1 < s.model.nHeads) ImGui::SameLine();
            }
            ImGui::Unindent(36.0f);

            compRow("W_O [d->d]",   "W_O",        Sty().info);
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_dim);
            ImGui::Indent(20.0f); ImGui::TextUnformatted("v mlp"); ImGui::Unindent(20.0f);
            ImGui::PopStyleColor();
            compRow("rmsnorm_2",    "norm2",      Sty().text_muted);
            compRow("W_in (gate)",  "W_in_gate",  Sty().warn);
            compRow("W_in (up)",    "W_in_up",    Sty().warn);
            compRow("W_out",        "W_out",      Sty().warn);
        }
        ImGui::PopID();
    }
    ImGui::Unindent(12.0f);
    ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted); ImGui::Text("final_rmsnorm");
    ImGui::PopStyleColor();
    ImGui::PushStyleColor(ImGuiCol_Text, Sty().good);
    ImGui::Text("lm_head [%d, %d]", s.model.dModel, s.model.vocab);
    ImGui::PopStyleColor();
    ImGui::EndChild();
}

// ── Architecture map (custom 2D) ───────────────────────────────────────────
void DrawArchMap(AppState& s, Model& m) {
    char zoom[24]; std::snprintf(zoom, sizeof zoom, "zoom %d%%", int(s.archZoom * 100));
    DrawTitleBar("architecture_map", "◫", zoom, "arch", [&] {
        if (ImGui::SmallButton("-"))    s.archZoom = std::max(0.4f, s.archZoom - 0.1f);
        ImGui::SameLine();
        char z[8]; std::snprintf(z, sizeof z, "%d%%", int(s.archZoom * 100));
        if (ImGui::SmallButton(z))      s.archZoom = 1.0f;
        ImGui::SameLine();
        if (ImGui::SmallButton("+"))    s.archZoom = std::min(2.0f, s.archZoom + 0.1f);
        ImGui::SameLine();
        if (ImGui::SmallButton("(+) fit")) s.archZoom = 1.0f;
    });

    if (!ImGui::BeginChild("##archmap_body", ImVec2(0, 0), ImGuiChildFlags_None,
                            ImGuiWindowFlags_HorizontalScrollbar)) {
        ImGui::EndChild(); return;
    }

    if (!s.hasModel()) {
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
        ImGui::Dummy(ImVec2(0, 30));
        ImGui::TextUnformatted("    // no model loaded — open a checkpoint via File ▸ Open");
        ImGui::PopStyleColor();
        ImGui::EndChild();
        return;
    }

    const Cols cols = ComputeCols(s.model);
    const float Z = s.archZoom;

    // Pre-compute per-block heights/y.
    std::vector<float> blockY(s.model.nLayers), blockH(s.model.nLayers);
    float y = 130.0f;
    for (int L = 0; L < s.model.nLayers; ++L) {
        blockY[L] = y;
        blockH[L] = s.skippedLayers.contains(L) ? kBlockH_Skipped
                  : s.expandedLayers.contains(L) ? kBlockH_Expanded
                  : kBlockH_Collapsed;
        y += blockH[L] + kLayerGap;
    }
    const float stackEndY = y;
    const float totalH    = stackEndY + 110.0f;
    const float W         = (cols.blockW + 80.0f);

    const ImVec2 origin = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    auto P = [&](float x, float yp) { return ImVec2(origin.x + x * Z, origin.y + yp * Z); };

    // Grid background
    for (float gx = 0; gx < W; gx += 20)  dl->AddLine(P(gx, 0), P(gx, totalH), Sty().separator, 0.4f);
    for (float gy = 0; gy < totalH; gy += 20) dl->AddLine(P(0, gy), P(W, gy),  Sty().separator, 0.4f);

    // Top: tok_embed + pos_embed nodes + summary line.
    auto archNode = [&](float x, float yp, float w, float h, ImU32 fg, ImU32 bg,
                        const char* label, const char* sub) {
        dl->AddRectFilled(P(x, yp), P(x + w, yp + h), bg);
        dl->AddRect      (P(x, yp), P(x + w, yp + h), fg);
        dl->AddText(P(x + 6, yp + 4),  fg, label);
        if (sub) dl->AddText(P(x + 6, yp + 16), Sty().text_muted, sub);
    };
    char vbuf[64]; std::snprintf(vbuf, sizeof vbuf, "[%d x %d]", s.model.vocab, s.model.dModel);
    archNode(cols.label.x + 50,        20, 260, 48, Sty().magenta, Sty().bg_panel_alt,
             "tok_embed", vbuf);
    char pbuf[64]; std::snprintf(pbuf, sizeof pbuf, "theta=10000 / d_head=%d", s.model.dHead);
    archNode(cols.label.x + 50 + 280,  20, 240, 48, Sty().yellow, Sty().bg_panel_alt,
             "pos_embed (RoPE)", pbuf);

    char sumbuf[160];
    const long long block_p = static_cast<long long>(s.model.dModel) * s.model.dModel * 4
                            + static_cast<long long>(s.model.dModel) * s.model.dMlp * 3;
    const long long total_p = static_cast<long long>(s.model.vocab) * s.model.dModel * 2
                            + static_cast<long long>(s.model.nLayers) * block_p;
    std::snprintf(sumbuf, sizeof sumbuf,
        "%dL · %dh · d_model=%d · d_mlp=%d · params/block %.2fM · total %.0fM",
        s.model.nLayers, s.model.nHeads, s.model.dModel, s.model.dMlp,
        block_p / 1.0e6, total_p / 1.0e6);
    dl->AddText(P(cols.label.x + 50, 86), Sty().text_muted, sumbuf);
    dl->AddText(P(cols.label.x + 50, 100), Sty().text_dim,
                "click > expand · shift+click > skip layer · L:probe O · R:ablate X");

    // Block stack
    for (int L = 0; L < s.model.nLayers; ++L) {
        ImGui::PushID(L);
        const float y0 = blockY[L], h = blockH[L];
        const bool isSkipped  = s.skippedLayers.contains(L);
        const bool isExpanded = s.expandedLayers.contains(L) && !isSkipped;
        const bool isActive   = (L == s.activeLayer);

        // Body background + border
        dl->AddRectFilled(P(cols.label.x, y0), P(cols.label.x + cols.blockW, y0 + h),
                          isSkipped ? IM_COL32(60, 40, 40, 100)
                          : isActive ? Sty().accent_bg : Sty().bg_window);
        if (isSkipped) {
            AddDashedLine(dl, P(cols.label.x, y0),
                          P(cols.label.x + cols.blockW, y0), Sty().bad, 4, 3, 1);
        }
        dl->AddRect(P(cols.label.x, y0), P(cols.label.x + cols.blockW, y0 + h),
                    isSkipped ? Sty().bad : (isActive ? Sty().accent : Sty().border_strong),
                    0, 0, isActive ? 1.5f : 1.0f);

        // Body click — collapsed→expand+focus, expanded→focus only.
        // SetNextItemAllowOverlap so the smaller InvisibleButtons we submit
        // afterwards (head cells, component cells, mini-head bar) properly
        // steal hover from the body when the cursor is on top of them — else
        // every click on a head/component would also fire body's handler.
        ImGui::SetNextItemAllowOverlap();
        ImGui::SetCursorScreenPos(P(cols.label.x, y0));
        ImGui::InvisibleButton("body", { (cols.blockW) * Z, h * Z });
        if (ImGui::IsItemClicked() && !isSkipped) {
            if (!isExpanded) s.toggleLayerExpand(L);
            s.setActiveLayer(L);
        }

        // Layer label cell (left rect)
        const float lbl_w = 62.0f;
        dl->AddRectFilled(P(cols.label.x + 4, y0 + 4),
                          P(cols.label.x + 4 + lbl_w, y0 + h - 4),
                          isActive ? Sty().accent_bg_strong : Sty().bg_titlebar);
        dl->AddRect      (P(cols.label.x + 4, y0 + 4),
                          P(cols.label.x + 4 + lbl_w, y0 + h - 4),
                          isActive ? Sty().accent : Sty().border);
        char lbl[32]; std::snprintf(lbl, sizeof lbl, "%s L%02d",
            isSkipped ? "x" : (isExpanded ? "v" : ">"), L);
        dl->AddText(P(cols.label.x + 10, y0 + 8),
                    isSkipped ? Sty().bad : (isActive ? Sty().accent : Sty().text), lbl);

        const int layerPr = CountForLayer(s.probedHeads, L)
                          + CountForLayer(s.probedComponents, L);
        const int layerAb = CountForLayer(s.ablatedHeads, L)
                          + CountForLayer(s.ablatedComponents, L);
        if (!isSkipped) {
            char pbuf2[16]; std::snprintf(pbuf2, sizeof pbuf2, "O %d", layerPr);
            char abuf2[16]; std::snprintf(abuf2, sizeof abuf2, "X %d", layerAb);
            if (layerPr > 0) dl->AddText(P(cols.label.x + 10, y0 + 20), Sty().accent, pbuf2);
            if (layerAb > 0) dl->AddText(P(cols.label.x + 36, y0 + 20), Sty().bad,    abuf2);
        } else {
            dl->AddText(P(cols.label.x + 10, y0 + 20), Sty().bad, "SKIPPED");
        }

        ImGui::SetCursorScreenPos(P(cols.label.x + 4, y0 + 4));
        ImGui::InvisibleButton("label", { lbl_w * Z, (h - 8) * Z });
        if (ImGui::IsItemClicked()) {
            if (ImGui::GetIO().KeyShift) s.toggleLayerSkip(L);
            else { s.toggleLayerExpand(L); s.setActiveLayer(L); }
        }
        if (WasRightClicked()) s.toggleLayerSkip(L);

        // Residual stream line
        const float residY = y0 + h - 6.0f;
        if (!isSkipped) {
            dl->AddLine(P(cols.label.x + 70, residY),
                        P(cols.label.x + cols.blockW - 4, residY),
                        WithAlpha(Sty().accent, 0.4f), 2.0f * Z);
        } else {
            AddDashedLine(dl, P(cols.label.x + 70, y0 + h * 0.5f),
                          P(cols.label.x + cols.blockW - 4, y0 + h * 0.5f),
                          Sty().bad, 3, 3, 1.2f);
        }

        if (!isSkipped && !isExpanded) {
            // Mini head bar
            dl->AddText(P(cols.label.x + 76, y0 + 8), Sty().text_muted, "attn");
            const float mhw  = std::max(3.0f, std::min(8.0f, std::floor(180.0f / float(s.model.nHeads))));
            const float mhg  = 1.0f, mx0 = cols.label.x + 76 + 28;
            for (int hd = 0; hd < s.model.nHeads; ++hd) {
                ImGui::PushID(hd);
                char hk[16]; std::snprintf(hk, sizeof hk, "%d.%d", L, hd);
                const bool ab = s.ablatedHeads.contains(hk);
                const bool pr = s.probedHeads.contains(hk);
                const float sig = m.getHeadNorm(L, hd);
                const ImU32 fill = ab ? Sty().bad
                                  : pr ? Sty().accent
                                  : WithAlpha(Sty().accent, 0.15f + sig * 0.35f);
                const ImU32 bd   = ab ? Sty().bad : (pr ? Sty().accent : Sty().info);
                const float hx = mx0 + hd * (mhw + mhg);
                dl->AddRectFilled(P(hx, y0 + 8),  P(hx + mhw, y0 + 22), fill);
                dl->AddRect      (P(hx, y0 + 8),  P(hx + mhw, y0 + 22), bd, 0, 0, 0.4f);
                ImGui::SetCursorScreenPos(P(hx, y0 + 8));
                ImGui::InvisibleButton("h", { mhw * Z, 14 * Z });
                if (ImGui::IsItemClicked()) s.toggleProbe(L, hd);
                if (WasRightClicked())      s.toggleAblate(hk);
                ImGui::PopID();
            }
            const float mlp_x = mx0 + mhw * s.model.nHeads + 18;
            dl->AddText(P(mlp_x, y0 + 8), Sty().warn, "mlp ▭");
        } else if (!isSkipped && isExpanded) {
            // Section labels
            dl->AddText(P(cols.qkv.x - 8, y0 + 4), Sty().info, "v ATTN");
            dl->AddText(P(cols.norm2.x - 8, y0 + 4), Sty().warn, "v MLP");

            // Component cells (centered around y0 + 32 + offset)
            auto drawComp = [&](const char* id, std::string_view ref, float x, float w,
                                const char* label, const char* sub, ImU32 fg, ImU32 bg,
                                float yShift) {
                ImGui::PushID(id);
                const float yMid = y0 + 32.0f + yShift;
                char absk[64]; std::snprintf(absk, sizeof absk, "%d.%.40s", L, std::string(ref).c_str());
                const bool ab = s.ablatedComponents.contains(absk);
                const bool pr = s.probedComponents.contains(absk);
                dl->AddRectFilled(P(x, yMid), P(x + w, yMid + 26), bg);
                dl->AddRect      (P(x, yMid), P(x + w, yMid + 26),
                                  ab ? Sty().bad : fg, 0, 0, 1.0f);
                if (ab) {
                    AddDashedLine(dl, P(x, yMid),       P(x + w, yMid),       Sty().bad, 3, 2, 1);
                    AddDashedLine(dl, P(x, yMid + 26),  P(x + w, yMid + 26),  Sty().bad, 3, 2, 1);
                }
                if (pr) dl->AddCircleFilled(P(x + w - 4, yMid + 3), 2.5f * Z, Sty().accent);
                dl->AddText(P(x + 6, yMid + 4),  fg, label);
                if (sub) dl->AddText(P(x + 6, yMid + 16), Sty().text_muted, sub);

                ImGui::SetCursorScreenPos(P(x, yMid));
                ImGui::InvisibleButton("c", { w * Z, 26 * Z });
                if (ImGui::IsItemClicked()) s.toggleProbeComp(L, ref);
                if (WasRightClicked())      s.toggleAblateComp(L, ref);
                ImGui::PopID();
            };

            char dd[32];  std::snprintf(dd, sizeof dd, "γ [%d]", s.model.dModel);
            drawComp("n1",  "norm1", cols.norm1.x, cols.norm1.w, "rmsnorm_1", dd,
                     Sty().text_muted, Sty().bg_panel_alt, 0);

            drawComp("WQ", "W_Q", cols.qkv.x, cols.qkv.w, "W_Q", "d->d",
                     Sty().info, WithAlpha(Sty().info, 0.10f), -26);
            drawComp("WK", "W_K", cols.qkv.x, cols.qkv.w, "W_K", "d->d",
                     Sty().info, WithAlpha(Sty().info, 0.10f), 0);
            drawComp("WV", "W_V", cols.qkv.x, cols.qkv.w, "W_V", "d->d",
                     Sty().info, WithAlpha(Sty().info, 0.10f), 26);

            // Head cells (clickable)
            for (int hd = 0; hd < s.model.nHeads; ++hd) {
                ImGui::PushID(1000 + hd);
                char hk[16]; std::snprintf(hk, sizeof hk, "%d.%d", L, hd);
                const bool ab = s.ablatedHeads.contains(hk);
                const bool pr = s.probedHeads.contains(hk);
                const float sig = m.getHeadNorm(L, hd);
                const float hx = cols.heads.x + hd * (cols.headCellW + cols.headCellGap);
                const float hy = y0 + 32 - 4;
                const ImU32 fill = ab ? Sty().bad_bg
                                  : WithAlpha(Sty().accent, 0.10f + sig * 0.35f);
                const ImU32 bd   = pr ? Sty().accent : (ab ? Sty().bad : Sty().info);
                dl->AddRectFilled(P(hx, hy), P(hx + cols.headCellW, hy + 34), fill);
                dl->AddRect      (P(hx, hy), P(hx + cols.headCellW, hy + 34), bd, 0, 0, pr ? 1.2f : 0.5f);
                if (ab) AddDashedLine(dl, P(hx, hy), P(hx + cols.headCellW, hy + 34), Sty().bad, 2, 2, 1);
                if (cols.headCellW >= 14) {
                    char hl[8]; std::snprintf(hl, sizeof hl, "h%d", hd);
                    dl->AddText(P(hx + cols.headCellW * 0.5f - 6, hy + 14),
                                ab ? Sty().bad : Sty().info, hl);
                }
                if (pr) dl->AddCircleFilled(P(hx + cols.headCellW - 2, hy + 2), 1.4f * Z, Sty().accent);
                ImGui::SetCursorScreenPos(P(hx, hy));
                ImGui::InvisibleButton("hd", { cols.headCellW * Z, 34 * Z });
                if (ImGui::IsItemClicked()) s.toggleProbe(L, hd);
                if (WasRightClicked())      s.toggleAblate(hk);
                ImGui::PopID();
            }

            drawComp("WO", "W_O", cols.wo.x, cols.wo.w, "W_O", "d->d",
                     Sty().info, WithAlpha(Sty().info, 0.10f), 0);

            // Add nodes
            dl->AddCircleFilled(P(cols.add1.x + cols.add1.w * 0.5f, residY), 6 * Z, Sty().bg_input);
            dl->AddCircle      (P(cols.add1.x + cols.add1.w * 0.5f, residY), 6 * Z, Sty().accent, 0, 1);
            dl->AddText        (P(cols.add1.x + cols.add1.w * 0.5f - 3, residY - 6), Sty().accent, "+");

            drawComp("n2", "norm2", cols.norm2.x, cols.norm2.w, "rmsnorm_2", dd,
                     Sty().text_muted, Sty().bg_panel_alt, 0);
            char gu[32]; std::snprintf(gu, sizeof gu, "d->%d", s.model.dMlp);
            drawComp("Wig", "W_in_gate", cols.gateup.x, cols.gateup.w, "W_in (gate)", gu,
                     Sty().warn, WithAlpha(Sty().warn, 0.10f), -15);
            drawComp("Wiu", "W_in_up",   cols.gateup.x, cols.gateup.w, "W_in (up)",   gu,
                     Sty().warn, WithAlpha(Sty().warn, 0.10f), 15);

            dl->AddCircleFilled(P(cols.silu.x + cols.silu.w * 0.5f, y0 + 45), 9 * Z, Sty().bg_input);
            dl->AddCircle      (P(cols.silu.x + cols.silu.w * 0.5f, y0 + 45), 9 * Z, Sty().warn, 0, 1);
            dl->AddText        (P(cols.silu.x + cols.silu.w * 0.5f - 3, y0 + 40), Sty().warn, "x");

            char gout[32]; std::snprintf(gout, sizeof gout, "%d->d", s.model.dMlp);
            drawComp("Wo", "W_out", cols.wout.x, cols.wout.w, "W_out", gout,
                     Sty().warn, WithAlpha(Sty().warn, 0.10f), 0);

            dl->AddCircleFilled(P(cols.add2.x + cols.add2.w * 0.5f, residY), 6 * Z, Sty().bg_input);
            dl->AddCircle      (P(cols.add2.x + cols.add2.w * 0.5f, residY), 6 * Z, Sty().accent, 0, 1);
            dl->AddText        (P(cols.add2.x + cols.add2.w * 0.5f - 3, residY - 6), Sty().accent, "+");
        }

        // Inter-block residual continuation
        if (L < s.model.nLayers - 1) {
            dl->AddLine(P(cols.label.x + cols.blockW - 4, isSkipped ? y0 + h * 0.5f : residY),
                        P(cols.label.x + cols.blockW + 12, isSkipped ? y0 + h * 0.5f : residY),
                        WithAlpha(Sty().accent, 0.55f), 1.5f * Z);
        }
        ImGui::PopID();
    }

    // Final norm + lm_head
    archNode(cols.label.x + 60, stackEndY + 16, 140, 36, Sty().text_muted, Sty().bg_panel_alt,
             "final_rmsnorm", "epsilon=1e-5");
    char lh[64]; std::snprintf(lh, sizeof lh, "[%d x %d] -> logits", s.model.dModel, s.model.vocab);
    archNode(cols.label.x + 220, stackEndY + 16, 260, 36, Sty().good, Sty().bg_panel_alt,
             "lm_head", lh);

    // Reserve scrolling area.
    ImGui::Dummy({W * Z, totalH * Z});

    // Mouse-wheel zoom
    if (ImGui::IsWindowHovered()) {
        const float wheel = ImGui::GetIO().MouseWheel;
        if (wheel != 0 && ImGui::GetIO().KeyCtrl) {
            s.archZoom = std::clamp(s.archZoom + wheel * 0.1f, 0.3f, 3.0f);
        }
    }
    ImGui::EndChild();
}

// ── Inspector ──────────────────────────────────────────────────────────────
void DrawInspector(AppState& s, Model& m) {
    DrawTitleBar("component_inspector", "◈", nullptr, "inspect");
    if (!ImGui::BeginChild("##insp_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }

    char lbuf[32];  std::snprintf(lbuf, sizeof lbuf, "block_%02d", s.activeLayer);
    char l1[32];    std::snprintf(l1,   sizeof l1,   "L%02d / %d", s.activeLayer, s.model.nLayers);
    char d1[16];    std::snprintf(d1,   sizeof d1,   "%d", s.model.dModel);
    char nh[16];    std::snprintf(nh,   sizeof nh,   "%d", s.model.nHeads);
    char dh[16];    std::snprintf(dh,   sizeof dh,   "%d", s.model.dHead);
    char dm[16];    std::snprintf(dm,   sizeof dm,   "%d", s.model.dMlp);

    if (auto sec = BeginSection(lbuf, true)) {
        // ModelInfo (act / norm / rope) is checkpoint-static config —
        // sourced from AppState.model which the engine populates on load.
        // [DATA HOOK] expose model-config strings (act_fn, norm_kind,
        // rope_theta) on ModelInfo when wiring a real backend.
        KV({
            { "layer",   l1, "accent" },
            { "d_model", d1, "" },
            { "n_heads", nh, "" },
            { "d_head",  dh, "" },
            { "d_mlp",   dm, "" },
            { "act",     "SiLU (SwiGLU)", "" },         // [DATA HOOK]
            { "norm",    "RMSNorm pre",   "" },         // [DATA HOOK]
            { "rope θ",  "10000.0",       "" },         // [DATA HOOK]
        });
    }

    if (auto sec = BeginSection("Parameter breakdown", false, "block")) {
        // [DATA HOOK] Model::getParamBreakdown(layer) — per-component
        // parameter counts (in millions) + percent-of-block.
        const auto rows = m.getParamBreakdown(s.activeLayer);
        if (rows.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
            ImGui::TextUnformatted("// no breakdown available");
            ImGui::PopStyleColor();
        } else {
            std::vector<KVRow> kv; kv.reserve(rows.size());
            std::vector<std::string> values; values.reserve(rows.size());
            for (const auto& r : rows) {
                char buf[40];
                if (r.params_M < 0.01f)
                    std::snprintf(buf, sizeof buf, "%.0fK  (%.2f%%)",
                                  double(r.params_M * 1000.0f), double(r.pct * 100.0f));
                else
                    std::snprintf(buf, sizeof buf, "%.2fM  (%.0f%%)",
                                  double(r.params_M), double(r.pct * 100.0f));
                values.emplace_back(buf);
            }
            for (std::size_t i = 0; i < rows.size(); ++i) {
                kv.push_back({ rows[i].component.c_str(), values[i], rows[i].tone });
            }
            // KV() takes initializer_list which can't take a runtime-sized
            // vector; render the rows manually instead.
            if (ImGui::BeginTable("##pbk", 2, ImGuiTableFlags_NoHostExtendX |
                                              ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("k", ImGuiTableColumnFlags_WidthFixed, 130);
                ImGui::TableSetupColumn("v", ImGuiTableColumnFlags_WidthStretch);
                for (std::size_t i = 0; i < rows.size(); ++i) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        rows[i].tone && *rows[i].tone ? ToneColor(rows[i].tone) : Sty().text_muted);
                    ImGui::TextUnformatted(rows[i].component.c_str());
                    ImGui::PopStyleColor();
                    ImGui::TableNextColumn();
                    const float cw = ImGui::GetContentRegionAvail().x;
                    const ImVec2 sz = ImGui::CalcTextSize(values[i].c_str());
                    if (sz.x < cw) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (cw - sz.x));
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        rows[i].tone && *rows[i].tone ? ToneColor(rows[i].tone) : Sty().text);
                    ImGui::TextUnformatted(values[i].c_str());
                    ImGui::PopStyleColor();
                }
                ImGui::EndTable();
            }
        }
    }

    if (auto sec = BeginSection("Live activations", false, "∂fwd")) {
        // [DATA HOOK] Model::getLiveActivations(layer) — per-layer live
        // activation norms + entropy + dead-neuron count for the active
        // layer.  Engine source: forward-hook output captured on the
        // current token.
        const auto a = m.getLiveActivations(s.activeLayer);
        char dn[32];
        if (a.dead_neurons == kNoInt || a.total_neurons == kNoInt)
            std::snprintf(dn, sizeof dn, "—");
        else
            std::snprintf(dn, sizeof dn, "%d / %d", a.dead_neurons, a.total_neurons);
        const std::string entropy = std::isnan(a.attn_entropy_avg)
            ? "—" : (FmtFloat(a.attn_entropy_avg, "%.2f") + " nats");
        KV({
            { "attn_out norm",    FmtFloat(a.attn_out_norm,   "%.2f"), "info" },
            { "mlp_out norm",     FmtFloat(a.mlp_out_norm,    "%.2f"), "warn" },
            { "resid_post norm",  FmtFloat(a.resid_post_norm, "%.2f"), "accent" },
            { "attn entropy avg", entropy,                             "" },
            { "dead neurons",     dn,                                   "warn" },
        });
    }
    ImGui::EndChild();
}

// ── Param distribution ────────────────────────────────────────────────────
void DrawDist(AppState& s, Model& m) {
    DrawTitleBar("param_distribution", "∿", nullptr, "dist");
    if (!ImGui::BeginChild("##dist_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    if (auto sec = BeginSection("weight magnitudes per component", true)) {
        const char* names[] = { "W_Q","W_K","W_V","W_O","W_in (gate)","W_in (up)","W_out","norm.γ" };
        const char* keys [] = { "attn.W_Q","attn.W_K","attn.W_V","attn.W_O",
                                 "mlp.W_in_gate","mlp.W_in_up","mlp.W_out","norm1" };
        const ImU32 cols[] = { Sty().info, Sty().info, Sty().info, Sty().info,
                                Sty().warn, Sty().warn, Sty().warn, Sty().text_muted };
        const float child_w = (ImGui::GetContentRegionAvail().x - 4 * 8) / 4.0f;
        for (int i = 0; i < 8; ++i) {
            ImGui::BeginGroup();
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
            ImGui::TextUnformatted(names[i]);
            ImGui::PopStyleColor();
            // [DATA HOOK] Model::getWeightHistogram(name, bins) — bin counts
            // for the named tensor's value distribution.  Tensor name is
            // built as `blocks.<L>.<component>.weight`.
            char tname[64];
            std::snprintf(tname, sizeof tname,
                          "blocks.%d.%s.weight", s.activeLayer, keys[i]);
            const auto bins = m.getWeightHistogram(tname, 30);
            if (bins.empty()) {
                ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_dim);
                ImGui::TextUnformatted("// no data");
                ImGui::PopStyleColor();
            } else {
                ActivationHistogram(bins, child_w - 4, 60, cols[i]);
            }
            ImGui::EndGroup();
            if (i % 4 != 3) ImGui::SameLine();
        }
    }
    ImGui::EndChild();
}

// ── Architecture ops ──────────────────────────────────────────────────────
void DrawOps(AppState& s, Model& m) {
    (void)m;
    DrawTitleBar("architecture_ops", "⚙", nullptr, "arch-ops");
    if (!ImGui::BeginChild("##ops_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    char ref[64];   std::snprintf(ref, sizeof ref, "blocks.%d.attn.W_Q", s.activeLayer);
    char shape[32]; std::snprintf(shape, sizeof shape, "[%d, %d]", s.model.dModel, s.model.dModel);
    char prk[32];   std::snprintf(prk, sizeof prk, "%d.W_Q", s.activeLayer);
    const bool prc = s.probedComponents.contains(prk);
    const bool abc = s.ablatedComponents.contains(prk);
    const bool sk  = s.skippedLayers.contains(s.activeLayer);
    if (auto sec = BeginSection("Selected component", true)) {
        KV({
            { "ref",          ref,     "accent" },
            { "shape",        shape,   "" },
            { "probes",       prc ? "1" : "0", prc ? "accent" : "" },
            { "ablated",      abc ? "yes" : "no", abc ? "bad" : "good" },
            { "layer skipped",sk  ? "yes" : "no", sk  ? "bad" : "" },
        });
        EndSection(sec);
    }
    if (ImGui::Button(prc ? "(o) detach probe"  : "(o) attach probe"))  s.toggleProbeComp(s.activeLayer, "W_Q");
    if (ImGui::Button(abc ? "X restore component" : "X ablate component")) s.toggleAblateComp(s.activeLayer, "W_Q");
    if (ImGui::Button(sk  ? "X restore layer"     : "X skip whole layer")) s.toggleLayerSkip(s.activeLayer);
    if (ImGui::Button(s.expandedLayers.contains(s.activeLayer) ? "v force-collapse" : "v force-expand"))
        s.toggleLayerExpand(s.activeLayer);
    ImGui::EndChild();
}

// ── Raw hex pane (only when showRaw) ───────────────────────────────────────
void DrawRawHex(AppState& s, Model& m) {
    char flag[16]; std::snprintf(flag, sizeof flag, "W_Q[L%02d]", s.activeLayer);
    DrawTitleBar("param_hex", "0x", flag, "param-hex");
    if (!ImGui::BeginChild("##rawhex_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    // [DATA HOOK] Model::getWeightSlice(name, offset, n) — n raw fp16
    // values from the named tensor starting at `offset`.  Engine reads
    // straight from the checkpoint tensor (mmap'd ideally).
    char tname[64]; std::snprintf(tname, sizeof tname,
                                   "blocks.%d.attn.W_Q.weight", s.activeLayer);
    const auto buf = m.getWeightSlice(tname, 0, 200);
    if (buf.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
        ImGui::TextUnformatted("// no tensor data available");
        ImGui::PopStyleColor();
    } else {
        HexView(buf, 0, 3, 28, HexMode::Fp16);
    }
    ImGui::EndChild();
}

}  // namespace

// Layout: outline | center(map / dist|ops) | (inspector + raw tabbed)
//
// dock tree:
//
//   root ─split_left(0.18)─→ outline,  rest1
//   rest1 ─split_right(0.22)─→ rest2,  right (inspector + raw tabbed)
//   rest2 ─split_down(0.30)─→ map,     bot
//   bot   ─split_left(0.55)─→ dist,    ops
void BuildArchitectureLayout(ImGuiID dock_id) {
    ImGui::DockBuilderRemoveNode(dock_id);
    ImGui::DockBuilderAddNode(dock_id, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dock_id, ImGui::GetMainViewport()->Size);

    ImGuiID outline_n, rest1, rest2, right_n, map_n, bot_n, dist_n, ops_n;
    ImGui::DockBuilderSplitNode(dock_id, ImGuiDir_Left,  0.18f, &outline_n, &rest1);
    ImGui::DockBuilderSplitNode(rest1,  ImGuiDir_Right, 0.22f, &right_n,   &rest2);
    ImGui::DockBuilderSplitNode(rest2,  ImGuiDir_Down,  0.30f, &bot_n,     &map_n);
    ImGui::DockBuilderSplitNode(bot_n,  ImGuiDir_Left,  0.55f, &dist_n,    &ops_n);

    ImGui::DockBuilderDockWindow("arch.outline",   outline_n);
    ImGui::DockBuilderDockWindow("arch.map",       map_n);
    ImGui::DockBuilderDockWindow("arch.dist",      dist_n);
    ImGui::DockBuilderDockWindow("arch.ops",       ops_n);
    ImGui::DockBuilderDockWindow("arch.inspector", right_n);
    ImGui::DockBuilderDockWindow("arch.raw",       right_n);   // tabbed with inspector
    ImGui::DockBuilderFinish(dock_id);
}

void SubmitArchitecturePanels(AppState& s, Model& m) {
    if (!s.hasModel()) {
        if (ImGui::Begin("arch.outline", nullptr, ImGuiWindowFlags_NoTitleBar)) {
            EmptyStatePlaceholder("// no model loaded — open a checkpoint via File ▸ Open");
        }
        ImGui::End();
        return;
    }
    if (ImGui::Begin("arch.outline",   nullptr, ImGuiWindowFlags_NoTitleBar)) DrawOutline  (s, m);
    ImGui::End();
    if (ImGui::Begin("arch.map",       nullptr, ImGuiWindowFlags_NoTitleBar)) DrawArchMap  (s, m);
    ImGui::End();
    if (ImGui::Begin("arch.dist",      nullptr, ImGuiWindowFlags_NoTitleBar)) DrawDist     (s, m);
    ImGui::End();
    if (ImGui::Begin("arch.ops",       nullptr, ImGuiWindowFlags_NoTitleBar)) DrawOps      (s, m);
    ImGui::End();
    if (ImGui::Begin("arch.inspector", nullptr, ImGuiWindowFlags_NoTitleBar)) DrawInspector(s, m);
    ImGui::End();
    if (s.showRaw) {
        if (ImGui::Begin("arch.raw",   nullptr, ImGuiWindowFlags_NoTitleBar)) DrawRawHex   (s, m);
        ImGui::End();
    }
}

}  // namespace llob
