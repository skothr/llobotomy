#pragma once
#include <imgui.h>

#include <functional>
#include <span>
#include <vector>
#include <string_view>
#include <vector>

// Forward declarations for widgets that need AppState / Model — keeps
// this header free of the heavier appstate.hpp / model.hpp pulls.
namespace llmengine { struct Model; }

namespace llob {
struct AppState;

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
//
// `rowOffset` is the absolute row index of the first row in `buffer` (the
// caller's view into a larger virtual buffer); only affects the address
// column.  Defaults to 0 for non-paged use.
enum class HexMode { Fp16, Hex, U16 };
void HexView(std::span<const float> buffer,
             std::size_t baseAddr, int bytesPerRow, int rows,
             HexMode mode, const char* (*nameFn)(int) = nullptr,
             int rowOffset = 0);

// Virtualized hex view for tensors that may be many GB.  Uses
// ImGuiListClipper to determine which rows are visible, then asks
// `fetchRows(first, n)` to materialise only those rows (vector of length
// n*colsPerRow floats).  Caller passes the total row count + cols/row.
//
// fetchRows is invoked at most once per ListClipper step; on the empty-
// data return (e.g. backend has no tensor at this name), the page is
// rendered as muted "—" placeholders so the address column still shows.
void HexViewVirtual(int totalRows, int colsPerRow, std::size_t baseAddr,
                    HexMode mode,
                    const std::function<std::vector<float>(int firstRow, int n)>& fetchRows);

// Slim slider with embedded value label (matches HANDOFF.md ImSlider spec).
// Returns true when value changed this frame.
bool ImSliderF(const char* id, float& value, float minV, float maxV,
               const char* fmt, float width);

// Dashed line helper (no native ImDrawList equivalent). Steps along the
// segment by dash + gap pixels.
void AddDashedLine(ImDrawList* dl, ImVec2 p1, ImVec2 p2, ImU32 col,
                   float dash = 3.0f, float gap = 2.0f, float thickness = 1.0f);

// Prompt input — the shared "tell the engine what to inspect" widget.
// Surfaced in inference / attention / probes workspaces (anywhere the
// user benefits from controlling the active forward pass).  All call
// sites share AppState::promptDraft so the input state persists across
// workspace switches.  Submitting calls m.setActivePrompt(text) and
// updates AppState::lastSubmittedPrompt + promptSubmittedAt.
//
// Renders: multi-line input + Submit button.  Ctrl+Enter also submits.
// When the backend has no model loaded (s.hasModel() == false) the
// widget renders disabled with a hint to load a checkpoint first.
//
// Returns true if a submit happened this frame.
bool DrawPromptInput(AppState& s, llmengine::Model& m);

}  // namespace llob
