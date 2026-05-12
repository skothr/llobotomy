#pragma once
#include "appstate.hpp"
#include "model/model.hpp"

namespace llob {

// Triggers — populated by the menu / Ctrl+chord shortcuts.  Pass them in
// to DispatchFileDialogs each frame; the dispatcher opens the matching
// IGFD popup on the first frame the trigger fires, then renders the
// (possibly multi-frame) dialog and fires the wired Model::* method on
// confirmation.
struct FileDialogActions {
    bool open_ckpt    = false;
    bool save_probe   = false;
    bool export_state = false;
};

// Submit once per frame near the end of the main loop (after panels;
// before status bar is fine).  Modal dialogs over-render everything else,
// so order doesn't matter for them.
void DispatchFileDialogs(AppState& s, Model& m, const FileDialogActions& act);

}  // namespace llob
