#include "ui/chrome.hpp"

#include "style.hpp"
#include "ui/colormap.hpp"

#include <imgui_internal.h>

#include <algorithm>
#include <cstdio>
#include <string>

namespace llob {

void DrawTitleBar(const char* title, const char* icon, const char* flag,
                  const char* dockId, std::function<void()> controls) {
    const float bar_h = 22.0f;
    const ImVec2 p0   = ImGui::GetCursorScreenPos();
    const float  w    = ImGui::GetContentRegionAvail().x;
    const ImVec2 p1   = { p0.x + w, p0.y + bar_h };

    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p0, p1, Sty().bg_titlebar);
    dl->AddLine({p0.x, p1.y - 1}, {p1.x, p1.y - 1}, Sty().border);

    // Reserve a right-side area for the controls callback (if any).  This
    // is the EFFECTIVE width — controls() lays out its buttons within it
    // and the title-text region clips on its left edge so the title never
    // overruns into the controls.
    const float pad_x       = 8.0f;
    const float controls_w  = controls ? 220.0f : 0.0f;
    const float title_clip_right = p1.x - (controls ? controls_w + pad_x : pad_x);

    // Layout cursor inside the strip.
    float       cx    = p0.x + pad_x;
    const float ty    = p0.y + (bar_h - ImGui::GetFontSize()) * 0.5f;

    // Push a clip rect so any title text that would overflow into the
    // controls area (or past the right edge when controls is null) is
    // hidden rather than painting garbage.
    dl->PushClipRect(p0, ImVec2(title_clip_right, p1.y), true);
    if (icon && *icon) {
        dl->AddText({cx, ty}, Sty().accent, icon);
        cx += ImGui::CalcTextSize(icon).x + 6.0f;
    }
    dl->AddText({cx, ty}, Sty().text, title);
    cx += ImGui::CalcTextSize(title).x + 6.0f;
    if (flag && *flag) {
        dl->AddText({cx, ty}, Sty().text_dim, flag);
        cx += ImGui::CalcTextSize(flag).x + 6.0f;
    }
    // Drop the dock-id annotation when room is tight — it's purely a
    // debug aid (matches the JSX mock's faint `##dockId` suffix).
    if (dockId && *dockId) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "##%s", dockId);
        const float buf_w = ImGui::CalcTextSize(buf).x;
        if (cx + buf_w <= title_clip_right) {
            dl->AddText({cx, ty}, Sty().text_dim, buf);
        }
    }
    dl->PopClipRect();

    // Reserve the strip area so the parent window grows to include it; this
    // also advances the cursor to just below the strip.
    ImGui::SetCursorScreenPos(p0);
    ImGui::Dummy(ImVec2(w, bar_h));
    const ImVec2 below = ImGui::GetCursorScreenPos();

    // Controls float: render the actual ImGui buttons in a child region
    // anchored to the right end of the strip.
    if (controls) {
        ImGui::SetCursorScreenPos({p1.x - controls_w - pad_x, p0.y + 2.0f});
        ImGui::PushStyleColor(ImGuiCol_Button,        Sty().bg_titlebar);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, Sty().bg_header);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive,  Sty().bg_header_active);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4, 1));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,  ImVec2(4, 0));
        ImGui::BeginGroup();
        controls();
        ImGui::EndGroup();
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor(3);
        // Restore cursor to just below the strip and submit a zero-size
        // Dummy to clear the IsSetPos flag (so the strip's end-of-window
        // bounds-check doesn't re-flag it as an unanchored extension).
        ImGui::SetCursorScreenPos(below);
        ImGui::Dummy(ImVec2(0, 0));
    }
}

// ── Section ────────────────────────────────────────────────────────────────

SectionScope::~SectionScope() {
    if (!owned) return;
    if (open) ImGui::Unindent(2.0f);
    ImGui::Dummy(ImVec2(0, 4.0f));
    ImGui::PopID();
}

SectionScope BeginSection(const char* title, bool accent, const char* badge) {
    ImGui::PushID(title);
    const ImVec2 p0 = ImGui::GetCursorScreenPos();
    const float  w  = ImGui::GetContentRegionAvail().x;
    const float  h  = 20.0f;
    const ImVec2 p1 = { p0.x + w, p0.y + h };

    // Use ImGui's storage to remember open state.
    ImGuiID id = ImGui::GetID("##section_open");
    ImGuiStorage* st = ImGui::GetStateStorage();
    bool open = st->GetInt(id, 1) != 0;

    // Header strip
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p0, p1, Sty().bg_header);
    dl->AddRect      (p0, p1, Sty().border);
    if (accent) dl->AddLine({p0.x, p0.y}, {p0.x, p1.y}, Sty().accent, 2.0f);

    ImGui::InvisibleButton("##sec_hdr", ImVec2(w, h));
    if (ImGui::IsItemClicked()) { open = !open; st->SetInt(id, open ? 1 : 0); }

    const float tx = p0.x + 8.0f;
    const float ty = p0.y + (h - ImGui::GetFontSize()) * 0.5f;
    dl->AddText({tx, ty}, Sty().text_muted, open ? "v" : ">");
    const float lx = tx + 14.0f;
    dl->AddText({lx, ty}, accent ? Sty().accent : Sty().text_bright, title);

    if (badge && *badge) {
        const ImVec2 sz = ImGui::CalcTextSize(badge);
        const ImVec2 b0 = { p1.x - sz.x - 14.0f, p0.y + 2.0f };
        const ImVec2 b1 = { p1.x - 4.0f,         p1.y - 2.0f };
        dl->AddRectFilled(b0, b1, Sty().bg_input);
        dl->AddRect      (b0, b1, Sty().border);
        dl->AddText({b0.x + 5.0f, b0.y + (b1.y - b0.y - sz.y) * 0.5f}, Sty().text_muted, badge);
    }
    if (open) ImGui::Indent(2.0f);
    return SectionScope{ open, /*owned=*/true };
}

// Compatibility shim for `if (auto s = ...) { ...; EndSection(s); }` callers:
// transfer ownership to a temporary so the cleanup runs here exactly once.
void EndSection(SectionScope& s) {
    SectionScope take = std::move(s);
    (void)take;
}

// ── KV grid ────────────────────────────────────────────────────────────────

void KV(std::initializer_list<KVRow> rows, bool dense) {
    const float font = dense ? Sty().fs_xs : Sty().fs_sm;
    (void)font;  // ImGui font scaling handled by atlas; leave as-is for now.

    if (ImGui::BeginTable("##kv", 2, ImGuiTableFlags_SizingStretchProp |
                                   ImGuiTableFlags_NoHostExtendX)) {
        ImGui::TableSetupColumn("k", ImGuiTableColumnFlags_WidthFixed, 96.0f);
        ImGui::TableSetupColumn("v", ImGuiTableColumnFlags_WidthStretch);
        for (const auto& r : rows) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
            ImGui::TextUnformatted(r.k);
            ImGui::PopStyleColor();

            ImGui::TableSetColumnIndex(1);
            ImGui::PushStyleColor(ImGuiCol_Text, ToneColor(r.tone));
            // Right-align by computing width.
            const ImVec2 sz = ImGui::CalcTextSize(r.v.data(), r.v.data() + r.v.size());
            const float  cw = ImGui::GetContentRegionAvail().x;
            if (sz.x < cw) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (cw - sz.x));
            ImGui::TextUnformatted(r.v.data(), r.v.data() + r.v.size());
            ImGui::PopStyleColor();
        }
        ImGui::EndTable();
    }
}

// ── Bar ────────────────────────────────────────────────────────────────────

void Bar(float value, float width, float height, ImU32 color, const char* label) {
    const ImVec2 p0 = ImGui::GetCursorScreenPos();
    const ImVec2 p1 = { p0.x + width, p0.y + height };
    ImDrawList*  dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p0, p1, Sty().bg_input);
    dl->AddRect      (p0, p1, Sty().border);
    const float v = value < 0 ? 0 : (value > 1 ? 1 : value);
    dl->AddRectFilled(p0, { p0.x + width * v, p1.y }, color);

    ImGui::Dummy(ImVec2(width, height));
    if (label && *label) {
        ImGui::SameLine(0, 6);
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
        ImGui::TextUnformatted(label);
        ImGui::PopStyleColor();
    }
}

// ── Pill ───────────────────────────────────────────────────────────────────

void Pill(const char* text, const char* tone, bool solid) {
    const ImU32  fg   = ToneColor(tone);
    const ImU32  bg   = solid ? fg : (tone && std::string(tone) == "warn" ? Sty().warn_bg
                                  :  tone && std::string(tone) == "good" ? Sty().good_bg
                                  :  tone && std::string(tone) == "bad"  ? Sty().bad_bg
                                  :                                          Sty().accent_bg);
    const ImVec2 sz   = ImGui::CalcTextSize(text);
    const ImVec2 p0   = ImGui::GetCursorScreenPos();
    const float  pad  = 5.0f;
    const ImVec2 p1   = { p0.x + sz.x + pad * 2, p0.y + sz.y + 2 };
    ImDrawList*  dl   = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p0, p1, bg);
    dl->AddRect      (p0, p1, fg);
    dl->AddText({p0.x + pad, p0.y + 1}, solid ? Sty().bg_deep : fg, text);
    ImGui::Dummy({p1.x - p0.x, p1.y - p0.y});
}

// ── Workspace tabs ─────────────────────────────────────────────────────────

WorkspaceTabsResult DrawWorkspaceTabs(int& active, const char* const* labels, int n,
                                      const char* right_text) {
    WorkspaceTabsResult res;
    const float bar_h = 24.0f;
    const ImVec2 p0   = ImGui::GetCursorScreenPos();
    const float  full_w = ImGui::GetContentRegionAvail().x;
    const ImVec2 p1   = { p0.x + full_w, p0.y + bar_h };

    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p0, p1, Sty().bg_panel);
    dl->AddLine({p0.x, p1.y - 1}, {p1.x, p1.y - 1}, Sty().border);

    // Reserve the strip footprint up front so the parent's bounds already
    // include it.  We then drop invisible buttons by SetCursorScreenPos
    // inside the strip area and reset the cursor to its end-of-strip
    // position before returning.
    ImGui::Dummy(ImVec2(full_w, bar_h));
    const ImVec2 below = ImGui::GetCursorScreenPos();

    float cx = p0.x + 6.0f;
    for (int i = 0; i < n; ++i) {
        const bool   is_active = (i == active);
        const ImVec2 sz   = ImGui::CalcTextSize(labels[i]);
        const ImVec2 t0   = { cx,                    p0.y + 2 };
        const ImVec2 t1   = { cx + sz.x + 20.0f,     p1.y - 2 };
        if (is_active) {
            dl->AddRectFilled(t0, t1, Sty().accent_bg_strong);
            dl->AddRect      (t0, t1, Sty().accent);
        }
        dl->AddText({t0.x + 10.0f, t0.y + (t1.y - t0.y - sz.y) * 0.5f},
                    is_active ? Sty().accent : Sty().text_muted, labels[i]);

        ImGui::SetCursorScreenPos(t0);
        ImGui::PushID(i);
        ImGui::InvisibleButton("##wstab", ImVec2(t1.x - t0.x, t1.y - t0.y));
        if (ImGui::IsItemHovered()) res.hovered = i;
        if (ImGui::IsItemClicked() && active != i) { active = i; res.changed = true; }
        ImGui::PopID();
        cx = t1.x + 2.0f;
    }

    if (right_text && *right_text) {
        const ImVec2 sz = ImGui::CalcTextSize(right_text);
        dl->AddText({p1.x - sz.x - 10.0f, p0.y + (bar_h - sz.y) * 0.5f},
                    Sty().text_muted, right_text);
    }
    ImGui::SetCursorScreenPos(below);
    ImGui::Dummy(ImVec2(0, 0));   // clear IsSetPos flag for end-of-window check
    return res;
}

// ── Tree row ──────────────────────────────────────────────────────────────

bool TreeRow(const char* id_str, const char* label, TreeRowFlags f) {
    ImGui::PushID(id_str);
    const ImVec2 p0 = ImGui::GetCursorScreenPos();
    const float  w  = std::max(20.0f, ImGui::GetContentRegionAvail().x);
    const float  h  = Sty().row_h - 4.0f;
    const ImVec2 p1 = { p0.x + w, p0.y + h };
    ImDrawList* dl = ImGui::GetWindowDrawList();

    if (f.active) {
        dl->AddRectFilled(p0, p1, Sty().accent_bg_strong);
        dl->AddLine({p0.x, p0.y}, {p0.x, p1.y}, Sty().accent, 2.0f);
    }
    ImGui::InvisibleButton("##row", ImVec2(w, h));
    const bool clicked = f.selectable && ImGui::IsItemClicked();
    if (ImGui::IsItemHovered() && !f.active) {
        dl->AddRectFilled(p0, p1, Sty().bg_input_hover);
    }

    const ImU32 fg = f.fg ? f.fg : (f.active ? Sty().accent : Sty().text);
    const float ty = p0.y + (h - ImGui::GetFontSize()) * 0.5f;
    dl->AddText({p0.x + 8.0f + float(f.indent_px), ty}, fg, label);
    if (f.strikethru) {
        const ImVec2 sz = ImGui::CalcTextSize(label);
        const float  ly = ty + sz.y * 0.5f;
        dl->AddLine({p0.x + 8.0f + float(f.indent_px),        ly},
                    {p0.x + 8.0f + float(f.indent_px) + sz.x, ly}, fg);
    }
    ImGui::PopID();
    return clicked;
}

bool WasRightClicked() {
    return ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Right);
}

void EmptyStatePlaceholder(const char* message) {
    ImGui::Dummy(ImVec2(0, 24));
    ImGui::Indent(20);
    ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
    ImGui::TextUnformatted(message);
    ImGui::PopStyleColor();
    ImGui::Unindent(20);
}

}  // namespace llob
