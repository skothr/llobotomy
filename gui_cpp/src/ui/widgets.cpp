#include "ui/widgets.hpp"

#include "appstate.hpp"
#include "logger.hpp"
#include "style.hpp"
#include "ui/chrome.hpp"

#include "llm_engine/model.hpp"

#include <algorithm>
#include <chrono>
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

    const float bw = bins.empty() ? 0.0f : width / float(bins.size());
    float maxv = 0;
    if (!bins.empty()) {
        for (float v : bins) maxv = std::max(maxv, v);
        if (maxv == 0) maxv = 1;
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

    // Hit-test: report the hovered bin via tooltip + highlight.  The bin
    // index assumes the histogram covers a symmetric ±1 range (the
    // dashed line at width/2 is the implied zero); callers that care
    // about a specific value range can compute their own and overlay a
    // richer tooltip via IsItemHovered().
    ImGui::Dummy(ImVec2(width, height));
    if (!bins.empty() && ImGui::IsItemHovered()) {
        const float mx  = ImGui::GetMousePos().x;
        const int   idx = std::clamp(int((mx - p0.x) / bw), 0, int(bins.size()) - 1);
        const ImVec2 hb0 = { p0.x + idx * bw,           p0.y };
        const ImVec2 hb1 = { p0.x + (idx + 1) * bw,     p1.y };
        dl->AddRectFilled(hb0, hb1, WithAlpha(Sty().accent, 0.18f));
        // Bin range over [-1, +1] normalised — true range is engine-side.
        const float vmin = -1.0f + 2.0f * (float(idx)     / float(bins.size()));
        const float vmax = -1.0f + 2.0f * (float(idx + 1) / float(bins.size()));
        ImGui::SetTooltip("bin %d / %zu\nrange [% .3f, % .3f] (normalised)\ncount %.0f",
                          idx, bins.size(), double(vmin), double(vmax), double(bins[idx]));
    }
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

    // Hit-test: snap to the nearest sample and surface its value.
    ImGui::Dummy(ImVec2(o.width, o.height));
    if (ImGui::IsItemHovered()) {
        const float t   = std::clamp((ImGui::GetMousePos().x - p0.x) / o.width, 0.0f, 1.0f);
        const int   idx = std::clamp(int(t * float(data.size() - 1) + 0.5f),
                                     0, int(data.size()) - 1);
        const ImVec2 pt = pts[idx];
        // Vertical guide + filled dot at the snapped sample
        dl->AddLine({pt.x, p0.y}, {pt.x, p1.y}, WithAlpha(Sty().accent, 0.45f), 1.0f);
        dl->AddCircleFilled(pt, 3.0f, Sty().accent);
        ImGui::SetTooltip("idx %d / %zu\nvalue %+.4f\nmin %+.4f  max %+.4f",
                          idx, data.size(), double(data[idx]), double(lo), double(hi));
    }
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
    // label — clip to the panel's interior so a label longer than the
    // thumb doesn't bleed into the next thumb on the row.
    if (label) {
        const ImU32 col = dim ? Sty().text_dim : Sty().text_muted;
        const ImVec2 sz = ImGui::CalcTextSize(label);
        dl->PushClipRect(p0, p1, true);
        dl->AddText({p0.x + (panel_w - sz.x) * 0.5f, p0.y + pad + size + 1}, col, label);
        dl->PopClipRect();
    }

    ImGui::InvisibleButton("##athumb", { panel_w, panel_h });
    const bool clicked = ImGui::IsItemClicked();

    // Hit-test individual cells of the attention grid — surfaces the raw
    // attention weight + (i, j) indices so the viewer can drill from
    // pattern → exact (query, key) attention coefficient.
    if (ImGui::IsItemHovered() && !data.empty()) {
        const ImVec2 mp = ImGui::GetMousePos();
        const int j = std::clamp(int((mp.x - g0.x) / c), 0, n - 1);
        const int i = std::clamp(int((mp.y - g0.y) / c), 0, n - 1);
        if (mp.x >= g0.x && mp.x < g0.x + size && mp.y >= g0.y && mp.y < g0.y + size) {
            const ImVec2 r0 = { g0.x + j * c, g0.y + i * c };
            const ImVec2 r1 = { r0.x + c,     r0.y + c };
            dl->AddRect(r0, r1, Sty().accent, 0, 0, 1.5f);
            const float v = (i < int(data.size()) && j < int(data[i].size()))
                           ? data[i][j] : 0.0f;
            const bool masked = (j > i);   // causal
            ImGui::SetTooltip("query  i=%d (row)\nkey    j=%d (col)\nattn   %.4f%s",
                              i, j, double(v), masked ? "  (masked)" : "");
        }
    }
    return clicked;
}

// ── Full AttentionHeatmap with token labels ────────────────────────────────

int AttentionHeatmap(const std::vector<std::vector<float>>& data,
                     std::span<const std::string_view> tokens,
                     const AttnHeatmapOpts& o) {
    const int n = std::min<int>(int(tokens.size()), o.maxCells);
    const float c = o.cellSize;
    const float W = n * c;
    const float H = n * c;

    // Label gutter sizes — left wide enough for the longest token name,
    // top tall enough for vertical char stack.  Both clipped to a max so
    // pathological tokens don't shrink the heatmap area too much.
    constexpr int   kMaxTopChars = 8;
    constexpr float kCharStepY   = 9.0f;
    const float top_h = kMaxTopChars * kCharStepY + 6.0f;       // ~78
    float left_w = 70.0f;
    for (int i = 0; i < n; ++i) {
        const ImVec2 sz = ImGui::CalcTextSize(tokens[i].data(),
                                              tokens[i].data() + tokens[i].size());
        left_w = std::max(left_w, sz.x + 8.0f);
    }
    left_w = std::min(left_w, 160.0f);   // cap at 160px

    const ImVec2 p_root = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();

    // Top labels — vertical char stack (one char per row).  Truncate at
    // kMaxTopChars so taller tokens don't blow past the heatmap top edge.
    for (int j = 0; j < n; ++j) {
        const std::string_view t = tokens[j];
        const ImU32 col = (j == o.selected) ? Sty().accent : Sty().text_muted;
        const float  cx = p_root.x + left_w + j * c;
        const int chars = std::min<int>(int(t.size()), kMaxTopChars);
        for (int k = 0; k < chars; ++k) {
            char buf[2] = { t[k], 0 };
            dl->AddText({cx + c*0.5f - 3, p_root.y + 4 + k * kCharStepY}, col, buf);
        }
    }

    // Left labels — right-aligned in the gutter.  Long tokens get
    // ellipsised with a leading "…" so the suffix stays meaningful.
    const float kLabelMaxW = left_w - 6.0f;
    for (int i = 0; i < n; ++i) {
        const std::string_view t = tokens[i];
        const ImU32 col = (i == o.selected) ? Sty().accent : Sty().text_muted;
        std::string truncated;
        const char* a = t.data();
        const char* b = t.data() + t.size();
        ImVec2 sz = ImGui::CalcTextSize(a, b);
        if (sz.x > kLabelMaxW) {
            // Drop chars from the FRONT until it fits, then prepend "…"
            std::string_view s = t;
            while (!s.empty() && ImGui::CalcTextSize(s.data(), s.data() + s.size()).x > kLabelMaxW - 8.0f) {
                s.remove_prefix(1);
            }
            truncated = "…" + std::string(s);
            a = truncated.data();
            b = truncated.data() + truncated.size();
            sz = ImGui::CalcTextSize(a, b);
        }
        dl->AddText({p_root.x + left_w - sz.x - 4, p_root.y + top_h + i * c + (c - sz.y) * 0.5f},
                    col, a, b);
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
    // Hit-test full cells region.  `selected` is the QUERY row (the
    // highlight code at line 235 paints `selected == i`), so we return
    // the ROW the click landed on, not the column.  Earlier code computed
    // (mp.x - g0.x) / c which is the column index — a real bug, the
    // highlighted row was driven by horizontal mouse position.
    ImGui::SetCursorScreenPos(g0);
    ImGui::InvisibleButton("##attn_cells", {W, H});
    if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        const ImVec2 mp = ImGui::GetMousePos();
        const int i = int((mp.y - g0.y) / c);
        if (i >= 0 && i < n) clicked_col = i;     // var misnamed, returns row
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

// ── Prompt input ───────────────────────────────────────────────────────────

bool DrawPromptInput(AppState& s, llmengine::Model& m) {
    const float avail_w = ImGui::GetContentRegionAvail().x;
    const float line_h  = ImGui::GetTextLineHeightWithSpacing();
    const float input_h = std::clamp(line_h * 4.0f, 60.0f, line_h * 6.0f);

    const bool can_submit = s.hasModel();

    if (!can_submit) {
        ImGui::TextDisabled("Load a checkpoint to submit prompts (File > Open).");
    } else if (!s.lastSubmittedPrompt.empty()) {
        const auto now   = std::chrono::steady_clock::now();
        const auto ms    = std::chrono::duration_cast<std::chrono::milliseconds>(
                               now - s.promptSubmittedAt).count();
        const double sec = double(ms) / 1000.0;
        ImGui::TextDisabled("Last submitted %.1fs ago (%zu chars)",
                            sec, s.lastSubmittedPrompt.size());
    } else {
        ImGui::TextDisabled("Enter a prompt; Enter to submit, Ctrl+Enter for newline.");
    }

    // Ensure capacity for the input buffer.  InputText with the
    // CallbackResize flag mutates the underlying string through the
    // callback as the user types past capacity.
    constexpr std::size_t kMinCap = 256;
    if (s.promptDraft.capacity() < kMinCap) s.promptDraft.reserve(kMinCap);

    struct CallbackCtx { std::string* str; };
    auto resize_cb = +[](ImGuiInputTextCallbackData* data) -> int {
        if (data->EventFlag == ImGuiInputTextFlags_CallbackResize) {
            auto* ctx = static_cast<CallbackCtx*>(data->UserData);
            ctx->str->resize(static_cast<std::size_t>(data->BufTextLen));
            data->Buf = ctx->str->data();
        }
        return 0;
    };
    CallbackCtx ctx{&s.promptDraft};

    bool submit = false;
    ImGui::PushID("prompt-input");
    ImGui::InputTextMultiline(
        "##draft",
        s.promptDraft.data(),
        s.promptDraft.capacity() + 1,
        ImVec2(avail_w, input_h),
        ImGuiInputTextFlags_AllowTabInput |
        ImGuiInputTextFlags_CallbackResize |
        ImGuiInputTextFlags_CtrlEnterForNewLine,
        resize_cb,
        &ctx);

    // Plain Enter (no Ctrl/Shift) while the input is focused → submit.
    // CtrlEnterForNewLine inverts the default so Ctrl+Enter inserts the
    // newline and bare Enter is "done".
    if (ImGui::IsItemFocused() &&
        ImGui::IsKeyPressed(ImGuiKey_Enter, false) &&
        !ImGui::GetIO().KeyCtrl &&
        !ImGui::GetIO().KeyShift) {
        submit = true;
    }
    ImGui::BeginDisabled(!can_submit);
    if (ImGui::Button("Submit", ImVec2(110, 0))) submit = true;
    ImGui::SameLine();
    ImGui::TextDisabled("(Enter to submit, Ctrl+Enter for newline)");
    ImGui::EndDisabled();
    ImGui::PopID();

    if (submit && can_submit) {
        // Strip the trailing newline the Enter keypress inserted before
        // our handler observed it.
        while (!s.promptDraft.empty() &&
               (s.promptDraft.back() == '\n' || s.promptDraft.back() == '\r')) {
            s.promptDraft.pop_back();
        }
        if (s.promptDraft.empty()) return false;

        s.lastSubmittedPrompt = s.promptDraft;
        s.promptSubmittedAt   = std::chrono::steady_clock::now();
        m.setActivePrompt(s.promptDraft);
        LLOB_LOG_INFO("prompt", "setActivePrompt: %zu chars", s.promptDraft.size());
        return true;
    }
    return false;
}

// ── Sampler controls ───────────────────────────────────────────────────────

bool DrawSamplerControls(AppState& s, llmengine::Model& m) {
    using SamplerConfig = llmengine::SamplerConfig;
    auto& cfg = s.samplerCfg;

    const bool can_apply = s.hasModel();

    bool changed = false;

    // ── Method radio ────────────────────────────────────────────────
    ImGui::PushID("sampler-controls");
    int method = static_cast<int>(cfg.method);
    if (ImGui::RadioButton("greedy", &method, static_cast<int>(SamplerConfig::Method::Greedy))) {
        cfg.method = SamplerConfig::Method::Greedy;
        changed    = true;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("sampling", &method, static_cast<int>(SamplerConfig::Method::Sampling))) {
        cfg.method = SamplerConfig::Method::Sampling;
        changed    = true;
    }

    // ── Chain stages (only meaningful when Sampling) ────────────────
    const bool sampling = (cfg.method == SamplerConfig::Method::Sampling);
    ImGui::BeginDisabled(!sampling);
    {
        if (ImGui::DragInt("top_k", &cfg.top_k, 1.0f, 0, 1000, "%d (0 = off)")) {
            cfg.top_k = std::max(0, cfg.top_k);
            changed   = true;
        }
        if (ImGui::DragFloat("top_p", &cfg.top_p, 0.005f, 0.0f, 1.0f, "%.3f (1.0 = off)")) {
            cfg.top_p = std::clamp(cfg.top_p, 0.0f, 1.0f);
            changed   = true;
        }
        if (ImGui::DragFloat("min_p", &cfg.min_p, 0.005f, 0.0f, 1.0f, "%.3f (0.0 = off)")) {
            cfg.min_p = std::clamp(cfg.min_p, 0.0f, 1.0f);
            changed   = true;
        }
        if (ImGui::DragFloat("temperature", &cfg.temperature, 0.01f, 0.0f, 10.0f, "%.2f")) {
            cfg.temperature = std::max(0.0f, cfg.temperature);
            changed         = true;
        }

        // Mirostat — replaces the temp+dist tail when enabled.
        const char* miro_labels[] = {"off", "v1", "v2"};
        if (ImGui::Combo("mirostat", &cfg.mirostat, miro_labels, 3)) {
            cfg.mirostat = std::clamp(cfg.mirostat, 0, 2);
            changed      = true;
        }
        if (cfg.mirostat != 0) {
            if (ImGui::DragFloat("mirostat_tau", &cfg.mirostat_tau, 0.05f, 0.1f, 20.0f, "%.2f")) {
                changed = true;
            }
            if (ImGui::DragFloat("mirostat_eta", &cfg.mirostat_eta, 0.01f, 0.001f, 1.0f, "%.3f")) {
                changed = true;
            }
        }

        int seed_int = static_cast<int>(cfg.seed);
        if (ImGui::DragInt("seed", &seed_int, 1.0f, 0, INT32_MAX, "%d")) {
            cfg.seed = static_cast<std::uint32_t>(std::max(0, seed_int));
            changed  = true;
        }
    }
    ImGui::EndDisabled();

    // ── Max generation tokens (always editable) ─────────────────────
    if (ImGui::DragInt("max_tokens", &s.maxGenerationTokens, 1.0f, 0, 4096,
                       "%d (0 = prefill only)")) {
        s.maxGenerationTokens = std::max(0, s.maxGenerationTokens);
        changed               = true;
    }

    if (changed) s.samplerDirty = true;

    // ── Apply button ────────────────────────────────────────────────
    ImGui::Spacing();
    ImGui::BeginDisabled(!can_apply || !s.samplerDirty);
    bool applied = false;
    if (ImGui::Button("Apply", ImVec2(110, 0))) applied = true;
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (!can_apply) {
        ImGui::TextDisabled("(load a checkpoint to apply)");
    } else if (s.samplerDirty) {
        ImGui::TextDisabled("(unsaved changes)");
    } else {
        ImGui::TextDisabled("(in sync with engine)");
    }

    ImGui::PopID();

    if (applied) {
        m.setSamplerConfig(cfg);
        m.setMaxGenerationTokens(s.maxGenerationTokens);
        s.samplerDirty = false;
        LLOB_LOG_INFO("sampler",
            "setSamplerConfig(method=%s top_k=%d top_p=%.3f min_p=%.3f temp=%.2f mirostat=%d) max_tokens=%d",
            cfg.method == SamplerConfig::Method::Greedy ? "greedy" : "sampling",
            cfg.top_k, double(cfg.top_p), double(cfg.min_p),
            double(cfg.temperature), cfg.mirostat, s.maxGenerationTokens);
        return true;
    }
    return false;
}

// ── Head grid ──────────────────────────────────────────────────────────────

namespace {

// Convert "L.h" UI keys to canonical "blocks.{L}.attn.head.{H}" strings
// (the wire format Model::setAblation expects).  Skips malformed keys
// silently — the engine logs warns for those.
std::vector<std::string> AblatedHeadsCanonical(const AppState& s) {
    std::vector<std::string> out;
    out.reserve(s.ablatedHeads.size());
    for (const auto& k : s.ablatedHeads) {
        const auto dot = k.find('.');
        if (dot == std::string::npos || dot == 0 || dot + 1 >= k.size()) continue;
        char buf[64];
        std::snprintf(buf, sizeof buf, "blocks.%.*s.attn.head.%s",
                      static_cast<int>(dot), k.c_str(), k.c_str() + dot + 1);
        out.emplace_back(buf);
    }
    return out;
}

}  // namespace

bool DrawHeadGrid(AppState& s, llmengine::Model& m) {
    const int n_layers = s.model.nLayers;
    const int n_heads  = s.model.nHeads;
    if (n_layers <= 0 || n_heads <= 0) {
        ImGui::TextDisabled("// no topology — load a model with non-zero nLayers/nHeads");
        return false;
    }

    // Cell sizing — fit to width so the grid is always one block wide.
    const float avail_w = ImGui::GetContentRegionAvail().x;
    constexpr float kPad      = 1.0f;
    constexpr float kRowLabel = 28.0f;       // "L 0" gutter on the left
    const float cell_w = std::max(8.0f,
        (avail_w - kRowLabel - 2.0f * kPad) / float(n_heads) - kPad);
    const float cell_h = std::min(cell_w, 18.0f);

    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImU32 col_base    = Sty().accent_dim;
    const ImU32 col_ablated = IM_COL32(20, 20, 22, 255);
    const ImU32 col_active  = Sty().accent;
    const ImU32 col_label   = Sty().text_muted;

    bool any_toggled = false;
    char hk[16];

    ImGui::PushID("head-grid");
    for (int L = 0; L < n_layers; ++L) {
        const ImVec2 row_origin = ImGui::GetCursorScreenPos();

        // Row label "L 03"
        char lbl[8];
        std::snprintf(lbl, sizeof lbl, "L%02d", L);
        dl->AddText({row_origin.x + 2.0f, row_origin.y + 2.0f}, col_label, lbl);

        for (int h = 0; h < n_heads; ++h) {
            std::snprintf(hk, sizeof hk, "%d.%d", L, h);
            const bool ablated = s.ablatedHeads.contains(hk);
            const bool active  = (L == s.activeLayer && h == s.activeHead);

            const float x = row_origin.x + kRowLabel + h * (cell_w + kPad);
            const float y = row_origin.y;
            const ImVec2 a{x, y};
            const ImVec2 b{x + cell_w, y + cell_h};

            dl->AddRectFilled(a, b, ablated ? col_ablated : col_base, 1.0f);
            if (active) {
                dl->AddRect(a, b, col_active, 1.0f, 0, 2.0f);
            }

            // Click hit test — invisible button on top of each cell.
            ImGui::SetCursorScreenPos(a);
            ImGui::PushID(L * 4096 + h);
            if (ImGui::InvisibleButton("c", ImVec2(cell_w, cell_h))) {
                s.toggleAblate(hk);
                s.activeLayer = L;
                s.activeHead  = h;
                any_toggled   = true;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("L%d head %d%s — click to %s",
                                  L, h, ablated ? " (ablated)" : "",
                                  ablated ? "restore" : "ablate");
            }
            ImGui::PopID();
        }
        ImGui::SetCursorScreenPos({row_origin.x, row_origin.y + cell_h + kPad});
    }
    ImGui::PopID();

    if (any_toggled) {
        // Push the updated ablation set to the engine.  Real backends
        // (LlamaCpp) translate this into cb_eval GPU write-back; others
        // record into view.surgery for honest UI reporting.
        m.setAblation(AblatedHeadsCanonical(s), {});
    }
    return any_toggled;
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
