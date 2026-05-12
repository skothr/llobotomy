#pragma once
#include <imgui.h>

#include <array>
#include <filesystem>
#include <string>
#include <string_view>

namespace llob {

// Every keyboard-bindable user action.  Add new actions here AND in the
// kAllActions table in keybindings.cpp (label/group/default chords).
enum class Action : int {
    OpenSettings,
    OpenAbout,

    OpenCheckpoint,
    SaveProbe,
    ExportState,

    NewProject,
    CloseProject,
    Quit,

    PrevWorkspace,
    NextWorkspace,

    Workspace1, Workspace2, Workspace3, Workspace4, Workspace5,
    Workspace6, Workspace7, Workspace8, Workspace9,

    NudgeLayerUp,
    NudgeLayerDown,
    NudgeTokenLeft,
    NudgeTokenRight,
    ToggleRun,

    ResetWorkspaceLayout,
    FitArchMap,

    Count,
};

constexpr int kNumActions = static_cast<int>(Action::Count);

const char* ActionLabel(Action a);   // "Open settings"
const char* ActionGroup(Action a);   // "App", "Project", "Navigation", ...

// One key combo: zero-or-more modifiers + a single non-modifier key.
struct KeyChord {
    bool     ctrl  = false;
    bool     shift = false;
    bool     alt   = false;
    ImGuiKey key   = ImGuiKey_None;

    bool empty() const { return key == ImGuiKey_None; }
    bool matches(const ImGuiIO& io) const;   // true the frame this chord fires
};

// "Ctrl+Shift+T" / "Space" / "1" / "(unbound)"
std::string FormatChord(const KeyChord& c);
// Inverse — accepts the same format.  Empty chord on parse failure.
KeyChord    ParseChord (std::string_view s);

// Two chords per action: primary + secondary.  Both are honoured by
// Pressed(); the UI shows both columns in the rebind table.
struct Keybindings {
    std::array<std::array<KeyChord, 2>, kNumActions> chords{};

    static Keybindings Defaults();
    bool Pressed(Action a) const;   // checks current frame's ImGui IO

    // Persistence — same XDG-aware path resolution as settings.json,
    // file is keybindings.json.  Save is atomic (write-tmp + rename).
    static std::filesystem::path Path();
    void Load();
    bool Save() const;
};

// Convenient enum→action iteration used by the rebind UI.
struct ActionDescriptor {
    Action      action;
    const char* label;
    const char* group;
    KeyChord    primary_default;
    KeyChord    secondary_default;
};
const ActionDescriptor* AllActions(int& count);   // returns a static array

}  // namespace llob
