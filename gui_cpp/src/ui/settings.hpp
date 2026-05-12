#pragma once
#include "appstate.hpp"

#include <filesystem>

namespace llob {

// Where the persisted preferences file lives.  Resolution order:
//   1. $XDG_CONFIG_HOME/llobotomy/settings.json
//   2. $HOME/.config/llobotomy/settings.json
//   3. ./settings.json   (last-resort fallback)
std::filesystem::path SettingsPath();

// Read the persisted theme/density/accent/showRaw/liveAnim into `s`.
// Silently no-ops when the file is absent or malformed — the defaults
// already sitting on AppState are kept.
void SettingsLoad(AppState& s);

// Atomically write the persisted fields out (write-tmp + rename).
// Creates parent dirs on first save.  Logs a warning on IO failure.
bool SettingsSave(const AppState& s);

// Submit the Settings modal once per frame.  Pass `request_open=true` from
// the View ▸ Settings… menu handler to make it appear.  Edits to theme /
// density / accent are applied (ApplyStyle) and persisted (SettingsSave)
// the moment they happen; the modal has no explicit Save button.
void DrawSettingsModal(AppState& s, bool request_open);

}  // namespace llob
