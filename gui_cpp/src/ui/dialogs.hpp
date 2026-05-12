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

// View ▸ Key bindings… modal.  Lists every Action with its current
// primary + secondary KeyChord.  Click any chord cell to enter capture
// mode (next keypress overwrites).  Saves to disk on every change.
//
// Returns true while a chord-capture is in progress so the main loop's
// HandleShortcuts can stand down.
void DrawKeybindingsModal(AppState& s, bool request_open);
bool IsCapturingKeybind();

}  // namespace llob
