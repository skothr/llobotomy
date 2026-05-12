#pragma once
#include <imgui.h>

namespace llob {

// Color a token-loss / metric value 0..1 with the canonical heat ramp
// (blue → cyan → amber → red).  Convenience wrappers that pick a sensible
// text color for high-contrast over the background.
ImU32 TextOverHeat(float v);   // 0..1; matches TokenGutter's white-on-dark/black-on-light flip

// Pick foreground color for a tone token used by KV / Section headers.
// Tones: "" / "accent" / "good" / "warn" / "bad" / "info" / "muted"
ImU32 ToneColor(const char* tone);

}  // namespace llob
