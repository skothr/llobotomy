#pragma once
#include "appstate.hpp"
#include "model/model.hpp"

#include <imgui.h>

namespace llob {

// Each workspace exposes two functions:
//
//   Build<Name>Layout(dock_id) — called by main.cpp the first frame the
//   workspace's dockspace exists (or after View ▸ Reset workspace layout).
//   Calls DockBuilder*: Remove + Add + SetNodeSize + Split* + DockWindow* +
//   Finish, populating the named panel windows into split nodes.
//
//   Submit<Name>Panels(s, m) — called every frame the workspace is active.
//   Submits each panel as a top-level ImGui::Begin window with NoTitleBar
//   so the custom DrawTitleBar serves as the chrome.  Panels auto-route
//   into the dockspace via the prior DockBuilderDockWindow assignment.
//
// Panel window names follow the pattern "<short_label>.<panel>" so they
// have stable IDs in imgui.ini.

// architecture (key "arch")
void BuildArchitectureLayout (ImGuiID dock_id);
void SubmitArchitecturePanels(AppState& s, Model& m);

// inference (key "inf")
void BuildInferenceLayout (ImGuiID dock_id);
void SubmitInferencePanels(AppState& s, Model& m);

// attention (key "attn")
void BuildAttentionLayout (ImGuiID dock_id);
void SubmitAttentionPanels(AppState& s, Model& m);

// probes (key "probes")
void BuildProbesLayout (ImGuiID dock_id);
void SubmitProbesPanels(AppState& s, Model& m);

// training (key "train")
void BuildTrainingLayout (ImGuiID dock_id);
void SubmitTrainingPanels(AppState& s, Model& m);

// finetune (key "ft")
void BuildFineTuneLayout (ImGuiID dock_id);
void SubmitFineTunePanels(AppState& s, Model& m);

// datasets (key "data")
void BuildDatasetsLayout (ImGuiID dock_id);
void SubmitDatasetsPanels(AppState& s, Model& m);

// raw_tensors (key "raw")
void BuildRawTensorsLayout (ImGuiID dock_id);
void SubmitRawTensorsPanels(AppState& s, Model& m);

// logs (key "logs")
void BuildLogsLayout (ImGuiID dock_id);
void SubmitLogsPanels(AppState& s, Model& m);

}  // namespace llob
