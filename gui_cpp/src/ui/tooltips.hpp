#pragma once
#include "model/model.hpp"

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
void ComponentTooltip(Model& m, int layer, std::string_view comp_ref);

// Single attention head: pattern label, mini attention thumb, per-head
// stats from getHeadStats.
void HeadTooltip(Model& m, int layer, int head);

// Block label: layer summary (resid/attn/mlp norms + dead-neuron count).
void LayerTooltip(Model& m, int layer);

}  // namespace llob
