#pragma once
#include "appstate.hpp"

namespace llob {

// Help ▸ About modal.  Shows version + log file path + build mode +
// a "Copy diagnostics" button that puts the boot info on the clipboard.
// Pass `request_open=true` from the menu / shortcut handler.
void DrawAboutDialog(const AppState& s, bool request_open);

// Bottom-right floating banner that surfaces the most recent
// Warn/Error/Fatal log entry for ~5 seconds.  Click to dismiss earlier.
// Submit once per frame near the end of the main loop.
void DrawErrorToast(const AppState& s);

}  // namespace llob
