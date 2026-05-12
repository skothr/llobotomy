#pragma once
#include "appstate.hpp"
#include "model/model.hpp"

namespace llob {

// Each workspace draws inside a full-bleed dockspace under the workspace tab
// strip.  Build the dockspace's split layout once via DockBuilder; subsequent
// frames let the user resize freely.  Pass-through ImGuiID for stability
// across workspace switches.
void DrawArchitectureWorkspace(AppState& s, Model& m);
void DrawInferenceWorkspace   (AppState& s, Model& m);
void DrawAttentionWorkspace   (AppState& s, Model& m);
void DrawProbesWorkspace      (AppState& s, Model& m);
void DrawTrainingWorkspace    (AppState& s, Model& m);
void DrawFineTuneWorkspace    (AppState& s, Model& m);
void DrawDatasetsWorkspace    (AppState& s, Model& m);
void DrawRawTensorsWorkspace  (AppState& s, Model& m);
void DrawLogsWorkspace        (AppState& s, Model& m);

}  // namespace llob
