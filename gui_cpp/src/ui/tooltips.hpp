#pragma once
#include "llm_engine/model.hpp"

#include <string_view>

namespace llob {

// Rich hover-tooltips for arch-map components / heads / layers.  Call
// IMMEDIATELY after the matching InvisibleButton — each helper checks
// `IsItemHovered(DelayNormal)` itself and is a no-op when not hovered.

// Component cell: tensor shape, dtype, param count, frob/op norms, and
// a mini histogram of the weight distribution.
//   `comp_ref` matches the ref string passed to the cell (e.g. "W_Q",
//   "W_in_gate", "norm1").  Uses canonical engine naming
//   `blocks.<L>.<area>.<comp>.weight` for the tensor lookup.
void ComponentTooltip(llmengine::Model& m, int layer, std::string_view comp_ref);

// Single attention head: pattern label, mini attention thumb, per-head
// stats from getHeadStats.
void HeadTooltip(llmengine::Model& m, int layer, int head);

// Block label: layer summary (resid/attn/mlp norms + dead-neuron count).
void LayerTooltip(llmengine::Model& m, int layer);

// Render any persistent panels pinned via middle-click on a tooltip-
// eligible item.  Each panel reuses the matching tooltip's body — same
// data, same code — so a real-backend hook lights up both surfaces.
// Submit once per frame near the end of the main loop (after panels;
// before the status bar is fine).
void DrawPinnedPanels(llmengine::Model& m);

}  // namespace llob
