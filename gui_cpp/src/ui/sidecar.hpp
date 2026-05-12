#pragma once
#include "appstate.hpp"

#include <filesystem>
#include <string_view>

namespace llob {

// Per-checkpoint sidecar JSON — stores ablation/probe/expanded/skipped
// sets so they round-trip across runs.  Saved next to the checkpoint:
//
//   /foo/bar/model.pt  →  /foo/bar/model.llobotomy.json
//
// Lives separately from settings.json because it's per-checkpoint, not
// per-user.  Schema documented in src/ui/sidecar.cpp.

// Compute the sidecar path for the given checkpoint path.  Returns an
// empty path when the input is empty.
std::filesystem::path SidecarPath(std::string_view checkpointPath);

// Load the sidecar (if it exists) and merge into AppState.  Replaces
// (does NOT append to) the current ablation/probe/skip/expand sets.
// Logs a debug line either way.  Silently no-ops if the file is absent.
void SidecarLoad(AppState& s, std::string_view checkpointPath);

// Write the sidecar atomically (write-tmp + rename).  Logs a warning on
// failure.  No-op when checkpointPath is empty.
bool SidecarSave(const AppState& s, std::string_view checkpointPath);

}  // namespace llob
