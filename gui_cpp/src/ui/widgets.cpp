#include "ui/widgets.hpp"

#include "style.hpp"
#include "ui/chrome.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace llob {

namespace {

void RangeOrAuto(std::span<const float> data, float& lo, float& hi) {
    if (lo == hi) {
        lo = +1e9f; hi = -1e9f;
        for (float v : data) { lo = std::min(lo, v); hi = std::max(hi, v); }
        if (lo == +1e9f) { lo = 0; hi = 1; }
        if (lo == hi)    { hi = lo + 1; }
    }
}

}  // namespace

// ── ActivationHistogram (int + float overloads) ────────────────────────────

void ActivationHistogramF(std::span<const float> bins, float width, float height,
                          ImU32 color, std::span<const LensAnnotation> annotations) {
    const ImVec2 p0 = ImGui::GetCursorScreenPos();
    const ImVec2 p1 = { p0.x + width, p0.y + height };
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p0, p1, Sty().bg_input);
    dl->AddRect      (p0, p1, Sty().border);

    if (!bins.empty()) {
        float maxv = 0; for (float v : bins) maxv = std::max(maxv, v);
        if (maxv == 0) maxv = 1;
        const float bw = width / float(bins.size());
        for (std::size_t i = 0; i < bins.size(); ++i) {
            const float h = (bins[i] / maxv) * (height - 14);
            const ImVec2 b0 = { p0.x + i * bw,           p1.y - h - 2 };
            const ImVec2 b1 = { p0.x + (i + 1) * bw - 0.5f, p1.y - 2 };
            dl->AddRectFilled(b0, b1, color);
        }
    }
    // zero line
    AddDashedLine(dl, {p0.x + width / 2, p0.y}, {p0.x + width / 2, p1.y},
                  Sty().border_strong, 2, 2, 1);
    for (const auto& a : annotations) {
        const float x = p0.x + a.x * width;
        dl->AddLine({x, p0.y + 2}, {x, p1.y - 2}, a.color, 1);
        if (a.label) dl->AddText({x + 3, p0.y + 1}, a.color, a.label);
    }
    ImGui::Dummy(ImVec2(width, height));
}

void ActivationHistogram(std::span<const int> bins, float width, float height,
                         ImU32 color, std::span<const LensAnnotation> annotations) {
    std::vector<float> f(bins.size());
    for (std::size_t i = 0; i < bins.size(); ++i) f[i] = float(bins[i]);
    ActivationHistogramF(f, width, height, color, annotations);
}

// ── Sparkline ──────────────────────────────────────────────────────────────

void Sparkline(std::span<const float> data, const SparkOpts& o) {
    if (data.size() < 2) { ImGui::Dummy(ImVec2(o.width, o.height)); return; }
    const ImU32 color = o.color ? o.color : Sty().accent;
    float lo = o.min, hi = o.max; RangeOrAuto(data, lo, hi);
    const float range = hi - lo;

    const ImVec2 p0 = ImGui::GetCursorScreenPos();
    const ImVec2 p1 = { p0.x + o.width, p0.y + o.height };
    ImDrawList* dl = ImGui::GetWindowDrawList();

    if (o.baseline) {
        AddDashedLine(dl, {p0.x, p0.y + o.height/2}, {p1.x, p0.y + o.height/2},
                      Sty().border, 2, 2, 1);
    }

    std::vector<ImVec2> pts(data.size());
    for (std::size_t i = 0; i < data.size(); ++i) {
        const float x = p0.x + (i / float(data.size() - 1)) * o.width;
        const float y = p1.y - ((data[i] - lo) / range) * (o.height - 2) - 1;
        pts[i] = {x, y};
    }

    if (o.fill) {
        std::vector<ImVec2> poly = pts;
        poly.push_back({p1.x, p1.y});
        poly.push_back({p0.x, p1.y});
        dl->AddConvexPolyFilled(poly.data(), int(poly.size()), (color & 0x00ffffff) | (40u << 24));
    }
    dl->AddPolyline(pts.data(), int(pts.size()), color, ImDrawFlags_None, 1.0f);

    ImGui::Dummy(ImVec2(o.width, o.height));
}

// ── TensorHeatmap ──────────────────────────────────────────────────────────

void TensorHeatmap(const std::vector<std::vector<float>>& data,
                   const HeatmapOpts& o,
                   ImU32 (*colorFn)(float)) {
    if (data.empty() || data[0].empty()) { ImGui::Dummy({o.width, o.height}); return; }
    if (!colorFn) colorFn = HeatColor;

    const int rows = int(data.size());
    const int cols = int(data[0].size());
    const float W = o.width  > 0 ? o.width  : float(cols * 6);
    const float H = o.height > 0 ? o.height : float(rows * 6);
    const float cw = W / float(cols);
    const float ch = H / float(rows);

    float lo = o.minV, hi = o.maxV;
    if (lo == hi) {
        lo = +1e9f; hi = -1e9f;
        for (auto& r : data) for (float v : r) { lo = std::min(lo, v); hi = std::max(hi, v); }
        if (lo == +1e9f) { lo = 0; hi = 1; }
        if (lo == hi)    { hi = lo + 1; }
    }
    const float range = hi - lo;

    const ImVec2 p0 = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    if (o.border) dl->AddRect(p0, {p0.x + W, p0.y + H}, Sty().border);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (o.causal && j > i) {
                dl->AddRectFilled({p0.x + j*cw, p0.y + i*ch},
                                  {p0.x + (j+1)*cw + 0.5f, p0.y + (i+1)*ch + 0.5f},
                                  Sty().bg_deep);
                continue;
            }
            const float v = std::clamp((data[i][j] - lo) / range, 0.0f, 1.0f);
            dl->AddRectFilled({p0.x + j*cw, p0.y + i*ch},
                              {p0.x + (j+1)*cw + 0.5f, p0.y + (i+1)*ch + 0.5f},
                              colorFn(v));
        }
    }
    if (o.selected_col >= 0 && o.selected_col < cols) {
        const float sx = p0.x + o.selected_col * cw;
        dl->AddRect({sx, p0.y}, {sx + cw, p0.y + H}, Sty().accent, 0, 0, 1.5f);
    }
    ImGui::Dummy({W, H});
}

// ── AttentionThumb (clickable) ─────────────────────────────────────────────

bool AttentionThumb(const std::vector<std::vector<float>>& data,
                    int n, float size,
                    const char* label,
                    bool active, bool dim) {
    const ImVec2 p0 = ImGui::GetCursorScreenPos();
    const float  pad = 3.0f;
    const float  panel_w = size + pad * 2;
    const float  panel_h = size + pad * 2 + 12.0f;        // label row
    const ImVec2 p1 = { p0.x + panel_w, p0.y + panel_h };

    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p0, p1, active ? Sty().accent_bg : Sty().bg_panel);
    dl->AddRect      (p0, p1, active ? Sty().accent    : Sty().border);

    // grid
    const ImVec2 g0 = { p0.x + pad, p0.y + pad };
    const float  c  = size / float(n);
    for (int i = 0; i < n; ++i) for (int j = 0; j <= i; ++j) {
        if (data.empty()) continue;
        const float v = data[i % data.size()][j % data[i].size()];
        dl->AddRectFilled({g0.x + j*c, g0.y + i*c},
                          {g0.x + (j+1)*c + 0.5f, g0.y + (i+1)*c + 0.5f},
                          dim ? Sty().bg_panel_alt : HeatColor(std::clamp(v, 0.0f, 1.0f)));
    }
    // label
    if (label) {
        const ImU32 col = dim ? Sty().text_dim : Sty().text_muted;
        const ImVec2 sz = ImGui::CalcTextSize(label);
        dl->AddText({p0.x + (panel_w - sz.x) * 0.5f, p0.y + pad + size + 1}, col, label);
    }

    ImGui::InvisibleButton("##athumb", { panel_w, panel_h });
    return ImGui::IsItemClicked();
}

// ── Full AttentionHeatmap with token labels ────────────────────────────────

int AttentionHeatmap(const std::vector<std::vector<float>>& data,
                     std::span<const std::string_view> tokens,
                     const AttnHeatmapOpts& o) {
    const int n = std::min<int>(int(tokens.size()), o.maxCells);
    const float c = o.cellSize;
    const float W = n * c;
    const float H = n * c;
    const float left_w = 60.0f, top_h = 60.0f;

    const ImVec2 p_root = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();

    // Top labels (rotated would need a custom path; we draw vertically by
    // painting one char per row to keep it simple — looks like the mock).
    for (int j = 0; j < n; ++j) {
        const std::string_view t = tokens[j];
        const ImU32 col = (j == o.selected) ? Sty().accent : Sty().text_muted;
        const float  cx = p_root.x + left_w + j * c;
        for (int k = 0; k < int(t.size()) && k < 8; ++k) {
            char buf[2] = { t[k], 0 };
            dl->AddText({cx + c*0.5f - 3, p_root.y + 4 + k * 9.0f}, col, buf);
        }
    }

    // Left labels
    for (int i = 0; i < n; ++i) {
        const std::string_view t = tokens[i];
        const ImU32 col = (i == o.selected) ? Sty().accent : Sty().text_muted;
        const ImVec2 sz = ImGui::CalcTextSize(t.data(), t.data() + t.size());
        dl->AddText({p_root.x + left_w - sz.x - 4, p_root.y + top_h + i * c + (c - sz.y) * 0.5f},
                    col, t.data(), t.data() + t.size());
    }

    // Cells
    const ImVec2 g0 = { p_root.x + left_w, p_root.y + top_h };
    dl->AddRect(g0, {g0.x + W, g0.y + H}, Sty().border_strong);
    int clicked_col = -1;
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
        const bool masked = o.causal && j > i;
        const ImVec2 r0 = { g0.x + j * c, g0.y + i * c };
        const ImVec2 r1 = { r0.x + c,     r0.y + c };
        if (masked) { dl->AddRectFilled(r0, r1, Sty().bg_deep); continue; }
        const float v = (i < int(data.size()) && j < int(data[i].size())) ? data[i][j] : 0.0f;
        dl->AddRectFilled(r0, r1, HeatColor(std::clamp(v, 0.0f, 1.0f)));
        if (o.selected == i && j <= i) {
            dl->AddRect(r0, r1, Sty().accent);
        }
    }
    // Hit-test full cells region (column granularity).
    ImGui::SetCursorScreenPos(g0);
    ImGui::InvisibleButton("##attn_cells", {W, H});
    if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        const ImVec2 mp = ImGui::GetMousePos();
        const int j = int((mp.x - g0.x) / c);
        if (j >= 0 && j < n) clicked_col = j;
    }

    ImGui::SetCursorScreenPos({p_root.x, p_root.y + top_h + H + 4});
    ImGui::Dummy(ImVec2(0, 0));
    return clicked_col;
}

// ── TokenGutter ────────────────────────────────────────────────────────────

void TokenGutter(std::span<const std::string_view> tokens,
                 std::span<const float>            metric,
                 ImU32 (*colorFn)(float)) {
    if (!colorFn) colorFn = HeatColor;
    if (tokens.empty()) return;
    float lo = +1e9f, hi = -1e9f;
    for (float v : metric) { lo = std::min(lo, v); hi = std::max(hi, v); }
    const float range = (hi == lo) ? 1.0f : (hi - lo);

    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(1, 0));
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        const float n = (metric[i] - lo) / range;
        const ImU32 bg = colorFn(n);
        const ImVec2 sz = ImGui::CalcTextSize(tokens[i].data(), tokens[i].data() + tokens[i].size());
        const ImVec2 p0 = ImGui::GetCursorScreenPos();
        const ImVec2 p1 = { p0.x + std::max(8.0f, sz.x + 8.0f), p0.y + sz.y + 4 };
        ImDrawList* dl = ImGui::GetWindowDrawList();
        dl->AddRectFilled(p0, p1, bg);
        dl->AddRect      (p0, p1, IM_COL32(0, 0, 0, 80));
        const ImU32 fg = (n > 0.5f) ? IM_COL32(0,0,0,255) : Sty().text_bright;
        dl->AddText({p0.x + 4, p0.y + 2}, fg, tokens[i].data(), tokens[i].data() + tokens[i].size());
        ImGui::Dummy({p1.x - p0.x, p1.y - p0.y});
        if (i + 1 < tokens.size()) ImGui::SameLine();
    }
    ImGui::PopStyleVar();
    ImGui::NewLine();
}

// ── LogitBars ──────────────────────────────────────────────────────────────

void LogitBars(std::span<const LogitItem> items, float width) {
    float maxp = 0; for (const auto& it : items) maxp = std::max(maxp, it.prob);
    if (maxp <= 0) maxp = 1;
    for (const auto& it : items) {
        const float p = it.prob / maxp;
        // tok column
        ImGui::PushStyleColor(ImGuiCol_Text, it.selected ? Sty().accent : Sty().text);
        const std::string lbl = std::string(it.selected ? "> " : "  ") + "\"" + std::string(it.token) + "\"";
        ImGui::TextUnformatted(lbl.c_str());
        ImGui::PopStyleColor();
        ImGui::SameLine(95.0f);
        Bar(p, width - 230.0f, 12.0f, it.selected ? Sty().accent : Sty().accent_dim, nullptr);
        ImGui::SameLine();
        char pct[16]; std::snprintf(pct, sizeof pct, "%.2f%%", it.prob * 100);
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
        ImGui::TextUnformatted(pct);
        ImGui::PopStyleColor();
        ImGui::SameLine();
        char dlt[16];
        if (it.delta == 0) std::snprintf(dlt, sizeof dlt, "—");
        else               std::snprintf(dlt, sizeof dlt, "%+.2f%%", it.delta * 100);
        ImGui::PushStyleColor(ImGuiCol_Text, it.delta == 0 ? Sty().text_dim
                                          : it.delta > 0 ? Sty().good : Sty().bad);
        ImGui::TextUnformatted(dlt);
        ImGui::PopStyleColor();
    }
}

// ── MiniGrid ───────────────────────────────────────────────────────────────

void MiniGrid(std::span<const float> values, int cols, float cellSize, ImU32 (*colorFn)(float)) {
    if (!colorFn) colorFn = [](float v) { return DivergeColor(v); };
    const int rows = (int(values.size()) + cols - 1) / cols;
    const ImVec2 p0 = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    for (std::size_t i = 0; i < values.size(); ++i) {
        const int x = int(i) % cols, y = int(i) / cols;
        dl->AddRectFilled({p0.x + x * cellSize,         p0.y + y * cellSize},
                          {p0.x + (x+1) * cellSize - 0.5f, p0.y + (y+1) * cellSize - 0.5f},
                          colorFn(values[i]));
    }
    ImGui::Dummy({cols * cellSize, rows * cellSize});
}

// ── HexView ────────────────────────────────────────────────────────────────

void HexView(std::span<const float> buffer,
             std::size_t baseAddr, int bytesPerRow, int rows,
             HexMode mode, const char* (*nameFn)(int),
             int rowOffset) {
    const ImVec2 p_root = ImGui::GetCursorScreenPos();
    const float row_h = 14.0f;
    const float W = ImGui::GetContentRegionAvail().x;
    const ImVec2 p1 = { p_root.x + W, p_root.y + rows * row_h + 8 };
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p_root, p1, Sty().bg_input);
    dl->AddRect      (p_root, p1, Sty().border);

    for (int r = 0; r < rows; ++r) {
        const int    offset = r * bytesPerRow;
        const ImVec2 ry  = { p_root.x + 6, p_root.y + 4 + r * row_h };
        // address — rowOffset shifts so paged callers can show the
        // absolute address rather than the slice-relative one.
        char addr[24];
        std::snprintf(addr, sizeof addr, "0x%08zx",
                      baseAddr + std::size_t(rowOffset + r) * std::size_t(bytesPerRow) * 4);
        dl->AddText(ry, Sty().text_dim, addr);
        // values
        float cx = ry.x + 80;
        for (int c = 0; c < bytesPerRow; ++c) {
            const int   idx = offset + c;
            const float v   = idx < int(buffer.size()) ? buffer[idx] : 0.0f;
            char vb[24];
            switch (mode) {
                case HexMode::Fp16: std::snprintf(vb, sizeof vb, "% .4f", v); break;
                case HexMode::Hex: {
                    std::uint32_t bits;
                    std::memcpy(&bits, &v, sizeof bits);
                    std::snprintf(vb, sizeof vb, "%08x", bits);
                    break;
                }
                case HexMode::U16: std::snprintf(vb, sizeof vb, "% .2f", v); break;
            }
            const ImU32 col = (v < 0) ? Sty().bad : (v > 0.001f ? Sty().accent : Sty().text_dim);
            dl->AddText({cx, ry.y}, col, vb);
            cx += 64;
        }
        // optional row name — passed the ABSOLUTE offset so the caller can
        // resolve a tensor-wide row label even when paged.
        if (nameFn) {
            const char* name = nameFn(offset + rowOffset * bytesPerRow);
            if (name) dl->AddText({cx + 6, ry.y}, Sty().warn, name);
        }
    }
    ImGui::Dummy(ImVec2(W, rows * row_h + 8));
}

void HexViewVirtual(int totalRows, int colsPerRow, std::size_t baseAddr,
                    HexMode mode,
                    const std::function<std::vector<float>(int, int)>& fetchRows) {
    if (totalRows <= 0) {
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
        ImGui::TextUnformatted("// empty tensor");
        ImGui::PopStyleColor();
        return;
    }
    constexpr float row_h = 14.0f;

    // Use ImGuiListClipper to skip rendering offscreen rows.  Each Step()
    // gives a [DisplayStart, DisplayEnd) range; we fetch only those rows
    // from the engine and render them via the regular HexView with
    // rowOffset=DisplayStart so the address column stays absolute.
    ImGuiListClipper clipper;
    clipper.Begin(totalRows, row_h);
    while (clipper.Step()) {
        const int first = clipper.DisplayStart;
        const int last  = clipper.DisplayEnd;
        const int n     = last - first;
        if (n <= 0) continue;
        // Backend pages by float offset, not byte offset.
        std::vector<float> page = fetchRows(first, n);
        // Render this page at the current cursor (clipper has positioned
        // it for us).  HexView is single-pane; pass n rows + rowOffset
        // for absolute addressing.
        std::span<const float> view(page.data(), page.size());
        HexView(view, baseAddr, colsPerRow, n, mode, nullptr, first);
    }
    clipper.End();
}

// ── ImSliderF ──────────────────────────────────────────────────────────────

bool ImSliderF(const char* id, float& value, float minV, float maxV,
               const char* fmt, float width) {
    ImGui::PushID(id);
    ImGui::PushStyleColor(ImGuiCol_FrameBg,        Sty().bg_input);
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, Sty().bg_input_hover);
    ImGui::PushStyleColor(ImGuiCol_SliderGrab,     Sty().accent);
    ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, Sty().accent);
    ImGui::SetNextItemWidth(width);
    const bool changed = ImGui::SliderFloat("##s", &value, minV, maxV, fmt);
    ImGui::PopStyleColor(4);
    ImGui::PopID();
    return changed;
}

// ── Dashed line ────────────────────────────────────────────────────────────

void AddDashedLine(ImDrawList* dl, ImVec2 p1, ImVec2 p2, ImU32 col,
                   float dash, float gap, float thickness) {
    const float dx = p2.x - p1.x, dy = p2.y - p1.y;
    const float len = std::sqrt(dx * dx + dy * dy);
    if (len < 1e-3f) return;
    const float ux = dx / len, uy = dy / len;
    float t = 0;
    while (t < len) {
        const float s = std::min(t + dash, len);
        dl->AddLine({p1.x + ux * t, p1.y + uy * t},
                    {p1.x + ux * s, p1.y + uy * s}, col, thickness);
        t = s + gap;
    }
}

}  // namespace llob
