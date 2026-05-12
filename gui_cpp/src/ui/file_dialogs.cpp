// File dialogs — wraps ImGuiFileDialog (libs/ImGuiFileDialog) for the
// File ▸ Open checkpoint / Save probe set / Export state actions.
//
// One render pump per frame — DispatchFileDialogs:
//   1. opens the matching IGFD popup on the first frame the trigger fires
//   2. submits Display() for each known dialog (Display is a no-op if not
//      open, and returns true the frame the user closes it)
//   3. on confirmation, fires the matching Model::* method and logs

#include "ui/file_dialogs.hpp"

#include "logger.hpp"
#include "ui/sidecar.hpp"

#include <ImGuiFileDialog.h>

#include <string>

namespace llob {

namespace {

constexpr const char* kOpenCkptKey   = "##fd_open_ckpt";
constexpr const char* kSaveProbeKey  = "##fd_save_probe";
constexpr const char* kExportKey     = "##fd_export";

// Filter strings — IGFD glob syntax is "( description ){ .ext1, .ext2 }" or
// just ".ext1,.ext2".  The .* fallback lets the user pick anything when
// the engine accepts non-canonical extensions.
constexpr const char* kCkptFilters   = ".pt,.safetensors,.bin,.gguf,.*";
constexpr const char* kProbeFilters  = ".pt,.json,.*";
constexpr const char* kExportFilters = ".json,.*";

// Default size for the dialog window
const ImVec2 kMinSize{640.0f, 420.0f};

}  // namespace

void DispatchFileDialogs(AppState& s, Model& m, const FileDialogActions& act) {
    auto* fd = ImGuiFileDialog::Instance();

    // Trigger -> open
    if (act.open_ckpt) {
        IGFD::FileDialogConfig cfg;
        cfg.path  = ".";                                  // CWD
        cfg.flags = ImGuiFileDialogFlags_Modal;
        fd->OpenDialog(kOpenCkptKey, "Open checkpoint",
                       kCkptFilters, cfg);
    }
    if (act.save_probe) {
        IGFD::FileDialogConfig cfg;
        cfg.path     = "./out/probes";                    // backend convention
        cfg.fileName = "probe.pt";
        cfg.flags    = ImGuiFileDialogFlags_ConfirmOverwrite |
                       ImGuiFileDialogFlags_Modal;
        fd->OpenDialog(kSaveProbeKey, "Save probe set",
                       kProbeFilters, cfg);
    }
    if (act.export_state) {
        IGFD::FileDialogConfig cfg;
        cfg.path     = "./out";
        cfg.fileName = "state.json";
        cfg.flags    = ImGuiFileDialogFlags_ConfirmOverwrite |
                       ImGuiFileDialogFlags_Modal;
        fd->OpenDialog(kExportKey, "Export state snapshot",
                       kExportFilters, cfg);
    }

    // Render + dispatch
    if (fd->Display(kOpenCkptKey, ImGuiWindowFlags_NoCollapse, kMinSize)) {
        if (fd->IsOk()) {
            const std::string path = fd->GetFilePathName();
            // [DATA HOOK] Model::loadCheckpoint(path) — base default returns
            // {ok=true} without doing work; real backend mmaps the file +
            // populates ModelInfo.  Async-load progress arrives via the
            // engine log bridge (see ENGINE_API.md §2.2).
            const auto res = m.loadCheckpoint(path);
            if (res.ok) {
                LLOB_LOG_INFO("ckpt", "loadCheckpoint(%s) ok", path.c_str());
                s.checkpointPath = path;
                s.loadFromModel(m);   // refresh ModelInfo + sample tokens
                SidecarLoad(s, path); // restore prior ablation/probe sets
            } else {
                LLOB_LOG_ERROR("ckpt", "loadCheckpoint(%s) failed: %s",
                               path.c_str(), res.error.c_str());
            }
        }
        fd->Close();
    }
    if (fd->Display(kSaveProbeKey, ImGuiWindowFlags_NoCollapse, kMinSize)) {
        if (fd->IsOk()) {
            const std::string path = fd->GetFilePathName();
            // [DATA HOOK] Model::saveProbe(path) — engine serialises the
            // currently-edited probe to <path> + a JSON sidecar.
            m.saveProbe(path);
            LLOB_LOG_INFO("probe", "saved probe -> %s", path.c_str());
        }
        fd->Close();
    }
    if (fd->Display(kExportKey, ImGuiWindowFlags_NoCollapse, kMinSize)) {
        if (fd->IsOk()) {
            const std::string path = fd->GetFilePathName();
            // [DATA HOOK] Model::exportSnapshot(path) — engine writes the
            // session state (ablations, probes, steering, features) to JSON.
            m.exportSnapshot(path);
            LLOB_LOG_INFO("export", "wrote state -> %s", path.c_str());
            // Also refresh the canonical sidecar so the next open of this
            // checkpoint picks up the same state automatically.
            SidecarSave(s, s.checkpointPath);
        }
        fd->Close();
    }
}

}  // namespace llob
