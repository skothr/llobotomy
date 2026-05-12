#pragma once
#include <imgui.h>

#include <span>
#include <string_view>
#include <vector>

namespace llob {

// All widgets here are stateless ImDrawList paint helpers. They render at the
// current cursor position and advance it with a Dummy of their own size.

// Bar chart of bin counts; no axes, no labels.
//   bins: counts; auto-scaled to local max.
//   color: applied to all bars.
//   annotations: optional vertical lines with labels (x is normalized 0..1).
struct LensAnnotation { float x; const char* label; ImU32 color; };
void ActivationHistogram(std::span<const int>     bins,
                         float width, float height, ImU32 color,
                         std::span<const LensAnnotation> annotations = {});
void ActivationHistogramF(std::span<const float>  bins,
                          float width, float height, ImU32 color,
                          std::span<const LensAnnotation> annotations = {});

// Mini line plot. min/max auto-detected unless provided. Optional fill
// underneath the line at 18% opacity.
struct SparkOpts {
    ImU32  color    = 0;       // 0 → accent
    bool   fill     = false;
    float  width    = 120.0f;
    float  height   = 22.0f;
    float  min      = 0.0f;
    float  max      = 0.0f;    // min==max → auto-range
    bool   baseline = false;
};
void Sparkline(std::span<const float> data, const SparkOpts& o);

// 2D heatmap; `data` is row-major (rows × cols).
//   cellW/cellH: 0 = stretch to fit width/height.
//   colorFn: lambda taking [0..1] → ImU32. Defaults to llob::HeatColor.
//   minV/maxV: data range; if equal, auto-detected.
struct HeatmapOpts {
    float width   = 0.0f;
    float height  = 0.0f;
    float minV    = 0.0f;
    float maxV    = 0.0f;
    bool  causal  = false;
    int   selected_col = -1;
    bool  border  = true;
};
void TensorHeatmap(const std::vector<std::vector<float>>& data,
                   const HeatmapOpts& o,
                   ImU32 (*colorFn)(float) = nullptr);

// Attention thumbnail — small, clickable, no axis labels. n × n cells.
bool AttentionThumb(const std::vector<std::vector<float>>& data,
                    int n, float size,
                    const char* label,
                    bool active, bool dim);

// Full attention heatmap with token-name axes.
struct AttnHeatmapOpts {
    float cellSize    = 20.0f;
    int   maxCells    = 28;
    int   selected    = -1;
    bool  causal      = true;
};
int  AttentionHeatmap(const std::vector<std::vector<float>>& data,
                      std::span<const std::string_view> tokens,
                      const AttnHeatmapOpts& o);    // returns clicked col idx, -1

// Token gutter — row of colored token chips.
//   metric: per-token value (auto-normalized to local min/max).
//   colorFn: maps normalized 0..1 → ImU32.
void TokenGutter(std::span<const std::string_view> tokens,
                 std::span<const float>            metric,
                 ImU32 (*colorFn)(float));

// Logit comparison bars — vertical bars per top-k token; selected gets
// accent outline; delta drawn as small +/− tick.
struct LogitItem {
    std::string_view token;
    float            prob;
    float            delta;       // 0 if absent
    bool             selected;
};
void LogitBars(std::span<const LogitItem> items, float width = 280.0f);

// Mini grid — flat array as colored cells. Used for Q/K/V vector slices.
void MiniGrid(std::span<const float> values, int cols, float cellSize,
              ImU32 (*colorFn)(float) = nullptr);

// Hex view — fp16 / hex / int8 / u16. Pass an offset into a virtual address
// space for the address column. nameFn(offset) → optional row label string
// (return nullptr for none).
enum class HexMode { Fp16, Hex, U16 };
void HexView(std::span<const float> buffer,
             std::size_t baseAddr, int bytesPerRow, int rows,
             HexMode mode, const char* (*nameFn)(int) = nullptr);

// Slim slider with embedded value label (matches HANDOFF.md ImSlider spec).
// Returns true when value changed this frame.
bool ImSliderF(const char* id, float& value, float minV, float maxV,
               const char* fmt, float width);

// Dashed line helper (no native ImDrawList equivalent). Steps along the
// segment by dash + gap pixels.
void AddDashedLine(ImDrawList* dl, ImVec2 p1, ImVec2 p2, ImU32 col,
                   float dash = 3.0f, float gap = 2.0f, float thickness = 1.0f);

}  // namespace llob
