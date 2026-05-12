#include "keybindings.hpp"

#include "logger.hpp"

#include <imgui.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace llob {

namespace {

// ── Default bindings table ────────────────────────────────────────────────
// Primary follows standard editor conventions; secondary is the bare-key
// variant where it makes sense (Workspace 1-9 + arrows + space).
constexpr KeyChord K(ImGuiKey k, bool ctrl = false, bool shift = false, bool alt = false) {
    return {ctrl, shift, alt, k};
}

const ActionDescriptor kAll[] = {
    // App
    { Action::OpenSettings,         "Open settings",            "App",         K(ImGuiKey_Comma, true), {} },
    { Action::OpenAbout,            "Open About dialog",        "App",         {},                       {} },
    // File
    { Action::OpenCheckpoint,       "Open checkpoint",          "File",        K(ImGuiKey_O, true),      {} },
    { Action::SaveProbe,            "Save probe set",           "File",        K(ImGuiKey_S, true),      {} },
    { Action::ExportState,          "Export state snapshot",    "File",        K(ImGuiKey_E, true),      {} },
    // Project tabs
    { Action::NewProject,           "New project tab",          "Project",     K(ImGuiKey_T, true),      {} },
    { Action::CloseProject,         "Close project tab",        "Project",     K(ImGuiKey_W, true),      {} },
    { Action::Quit,                 "Quit application",         "Project",     K(ImGuiKey_Q, true),      {} },
    // Workspace nav
    { Action::PrevWorkspace,        "Previous workspace",       "Navigation",  K(ImGuiKey_Tab, true, true), {} },
    { Action::NextWorkspace,        "Next workspace",           "Navigation",  K(ImGuiKey_Tab, true),    {} },
    { Action::Workspace1,           "Workspace 1 (arch)",       "Navigation",  K(ImGuiKey_1, true),      K(ImGuiKey_1) },
    { Action::Workspace2,           "Workspace 2 (inf)",        "Navigation",  K(ImGuiKey_2, true),      K(ImGuiKey_2) },
    { Action::Workspace3,           "Workspace 3 (attn)",       "Navigation",  K(ImGuiKey_3, true),      K(ImGuiKey_3) },
    { Action::Workspace4,           "Workspace 4 (probes)",     "Navigation",  K(ImGuiKey_4, true),      K(ImGuiKey_4) },
    { Action::Workspace5,           "Workspace 5 (train)",      "Navigation",  K(ImGuiKey_5, true),      K(ImGuiKey_5) },
    { Action::Workspace6,           "Workspace 6 (ft)",         "Navigation",  K(ImGuiKey_6, true),      K(ImGuiKey_6) },
    { Action::Workspace7,           "Workspace 7 (data)",       "Navigation",  K(ImGuiKey_7, true),      K(ImGuiKey_7) },
    { Action::Workspace8,           "Workspace 8 (raw)",        "Navigation",  K(ImGuiKey_8, true),      K(ImGuiKey_8) },
    { Action::Workspace9,           "Workspace 9 (logs)",       "Navigation",  K(ImGuiKey_9, true),      K(ImGuiKey_9) },
    // Editor
    { Action::NudgeLayerUp,         "Nudge active layer ↑",     "Editor",      K(ImGuiKey_UpArrow),      {} },
    { Action::NudgeLayerDown,       "Nudge active layer ↓",     "Editor",      K(ImGuiKey_DownArrow),    {} },
    { Action::NudgeTokenLeft,       "Nudge active token ←",     "Editor",      K(ImGuiKey_LeftArrow),    {} },
    { Action::NudgeTokenRight,      "Nudge active token →",     "Editor",      K(ImGuiKey_RightArrow),   {} },
    { Action::ToggleRun,            "Toggle run / pause",       "Editor",      K(ImGuiKey_Space),        {} },
    // View
    { Action::ResetWorkspaceLayout, "Reset workspace layout",   "View",        {},                       {} },
    { Action::FitArchMap,           "Fit arch map to viewport", "View",        K(ImGuiKey_0, true),      {} },
};
static_assert(sizeof(kAll) / sizeof(kAll[0]) == kNumActions,
              "kAll must have one entry per Action value");

// ── Key-name table ────────────────────────────────────────────────────────
// Just the keys we want to allow as bindable.  Modifiers (Left/RightCtrl
// etc) are intentionally absent — those live on KeyChord.{ctrl,shift,alt}.
struct Named { ImGuiKey k; const char* name; };
const Named kKeyNames[] = {
    {ImGuiKey_A,"A"},{ImGuiKey_B,"B"},{ImGuiKey_C,"C"},{ImGuiKey_D,"D"},
    {ImGuiKey_E,"E"},{ImGuiKey_F,"F"},{ImGuiKey_G,"G"},{ImGuiKey_H,"H"},
    {ImGuiKey_I,"I"},{ImGuiKey_J,"J"},{ImGuiKey_K,"K"},{ImGuiKey_L,"L"},
    {ImGuiKey_M,"M"},{ImGuiKey_N,"N"},{ImGuiKey_O,"O"},{ImGuiKey_P,"P"},
    {ImGuiKey_Q,"Q"},{ImGuiKey_R,"R"},{ImGuiKey_S,"S"},{ImGuiKey_T,"T"},
    {ImGuiKey_U,"U"},{ImGuiKey_V,"V"},{ImGuiKey_W,"W"},{ImGuiKey_X,"X"},
    {ImGuiKey_Y,"Y"},{ImGuiKey_Z,"Z"},
    {ImGuiKey_0,"0"},{ImGuiKey_1,"1"},{ImGuiKey_2,"2"},{ImGuiKey_3,"3"},
    {ImGuiKey_4,"4"},{ImGuiKey_5,"5"},{ImGuiKey_6,"6"},{ImGuiKey_7,"7"},
    {ImGuiKey_8,"8"},{ImGuiKey_9,"9"},
    {ImGuiKey_F1,"F1"},{ImGuiKey_F2,"F2"},{ImGuiKey_F3,"F3"},{ImGuiKey_F4,"F4"},
    {ImGuiKey_F5,"F5"},{ImGuiKey_F6,"F6"},{ImGuiKey_F7,"F7"},{ImGuiKey_F8,"F8"},
    {ImGuiKey_F9,"F9"},{ImGuiKey_F10,"F10"},{ImGuiKey_F11,"F11"},{ImGuiKey_F12,"F12"},
    {ImGuiKey_Tab,"Tab"},{ImGuiKey_Space,"Space"},{ImGuiKey_Enter,"Enter"},
    {ImGuiKey_Escape,"Esc"},{ImGuiKey_Backspace,"Backspace"},{ImGuiKey_Delete,"Del"},
    {ImGuiKey_Home,"Home"},{ImGuiKey_End,"End"},{ImGuiKey_PageUp,"PgUp"},{ImGuiKey_PageDown,"PgDn"},
    {ImGuiKey_UpArrow,"Up"},{ImGuiKey_DownArrow,"Down"},
    {ImGuiKey_LeftArrow,"Left"},{ImGuiKey_RightArrow,"Right"},
    {ImGuiKey_Comma,","},{ImGuiKey_Period,"."},{ImGuiKey_Slash,"/"},
    {ImGuiKey_Semicolon,";"},{ImGuiKey_Apostrophe,"'"},
    {ImGuiKey_LeftBracket,"["},{ImGuiKey_RightBracket,"]"},
    {ImGuiKey_Backslash,"\\"},{ImGuiKey_GraveAccent,"`"},
    {ImGuiKey_Minus,"-"},{ImGuiKey_Equal,"="},
};

const char* KeyToName(ImGuiKey k) {
    for (const auto& n : kKeyNames) if (n.k == k) return n.name;
    return "?";
}
ImGuiKey NameToKey(std::string_view name) {
    for (const auto& n : kKeyNames) if (name == n.name) return n.k;
    return ImGuiKey_None;
}

}  // namespace

// ── Public ────────────────────────────────────────────────────────────────

const char* ActionLabel(Action a) { return kAll[static_cast<int>(a)].label; }
const char* ActionGroup(Action a) { return kAll[static_cast<int>(a)].group; }
const ActionDescriptor* AllActions(int& count) { count = kNumActions; return kAll; }

bool KeyChord::matches(const ImGuiIO& io) const {
    if (key == ImGuiKey_None) return false;
    if (io.KeyCtrl  != ctrl)  return false;
    if (io.KeyShift != shift) return false;
    if (io.KeyAlt   != alt)   return false;
    return ImGui::IsKeyPressed(key, /*repeat=*/false);
}

std::string FormatChord(const KeyChord& c) {
    if (c.empty()) return "(unbound)";
    std::string out;
    if (c.ctrl)  out += "Ctrl+";
    if (c.shift) out += "Shift+";
    if (c.alt)   out += "Alt+";
    out += KeyToName(c.key);
    return out;
}

KeyChord ParseChord(std::string_view s) {
    KeyChord c;
    if (s.empty() || s == "(unbound)") return c;
    while (true) {
        const auto plus = s.find('+');
        if (plus == std::string_view::npos) break;
        const std::string_view tok = s.substr(0, plus);
        if      (tok == "Ctrl")  c.ctrl  = true;
        else if (tok == "Shift") c.shift = true;
        else if (tok == "Alt")   c.alt   = true;
        else break;     // unknown modifier → fall through to key parse
        s.remove_prefix(plus + 1);
    }
    c.key = NameToKey(s);
    if (c.key == ImGuiKey_None) c = {};   // unparseable → unbound
    return c;
}

Keybindings Keybindings::Defaults() {
    Keybindings kb;
    for (int i = 0; i < kNumActions; ++i) {
        kb.chords[i][0] = kAll[i].primary_default;
        kb.chords[i][1] = kAll[i].secondary_default;
    }
    return kb;
}

bool Keybindings::Pressed(Action a) const {
    const ImGuiIO& io = ImGui::GetIO();
    const auto& slots = chords[static_cast<int>(a)];
    return slots[0].matches(io) || slots[1].matches(io);
}

std::filesystem::path Keybindings::Path() {
    if (const char* x = std::getenv("XDG_CONFIG_HOME"); x && *x)
        return std::filesystem::path(x) / "llobotomy" / "keybindings.json";
    if (const char* h = std::getenv("HOME"); h && *h)
        return std::filesystem::path(h) / ".config" / "llobotomy" / "keybindings.json";
    return std::filesystem::path("./keybindings.json");
}

void Keybindings::Load() {
    *this = Defaults();
    const auto p = Path();
    std::ifstream in(p);
    if (!in) {
        LLOB_LOG_DEBUG("kb", "no bindings file at %s — using defaults", p.c_str());
        return;
    }
    std::stringstream buf; buf << in.rdbuf();
    const std::string body = buf.str();

    // Grep-style: one line per action like
    //   "OpenSettings": ["Ctrl+,", ""],
    // We parse each action's line independently — unknown actions are
    // ignored (forward-compat for renames).
    for (int i = 0; i < kNumActions; ++i) {
        const std::string label = std::string("\"") + kAll[i].label + "\"";
        const auto kpos = body.find(label);
        if (kpos == std::string::npos) continue;
        const auto lb = body.find('[', kpos);
        const auto rb = (lb == std::string::npos) ? std::string::npos : body.find(']', lb);
        if (lb == std::string::npos || rb == std::string::npos) continue;
        std::string_view inside(body.data() + lb + 1, rb - lb - 1);

        std::array<std::string_view, 2> tokens{};
        const auto comma = inside.find(',');
        if (comma == std::string_view::npos) {
            tokens[0] = inside;
        } else {
            tokens[0] = inside.substr(0, comma);
            tokens[1] = inside.substr(comma + 1);
        }
        for (int slot = 0; slot < 2; ++slot) {
            std::string_view t = tokens[slot];
            while (!t.empty() && (t.front() == ' ' || t.front() == '"' || t.front() == '\t'))
                t.remove_prefix(1);
            while (!t.empty() && (t.back()  == ' ' || t.back()  == '"' || t.back()  == '\t'))
                t.remove_suffix(1);
            chords[i][slot] = ParseChord(t);
        }
    }
    LLOB_LOG_INFO("kb", "loaded %s", p.c_str());
}

bool Keybindings::Save() const {
    const auto p = Path();
    std::error_code ec;
    std::filesystem::create_directories(p.parent_path(), ec);
    const auto tmp = p.string() + ".tmp";
    std::ofstream out(tmp);
    if (!out) {
        LLOB_LOG_WARN("kb", "could not open %s for write", tmp.c_str());
        return false;
    }
    out << "{\n";
    for (int i = 0; i < kNumActions; ++i) {
        out << "  \"" << kAll[i].label << "\": ["
            << "\"" << FormatChord(chords[i][0]) << "\", "
            << "\"" << FormatChord(chords[i][1]) << "\"]"
            << (i + 1 == kNumActions ? "\n" : ",\n");
    }
    out << "}\n";
    out.close();
    std::filesystem::rename(tmp, p, ec);
    if (ec) {
        LLOB_LOG_WARN("kb", "rename %s -> %s failed: %s",
                      tmp.c_str(), p.c_str(), ec.message().c_str());
        return false;
    }
    return true;
}

}  // namespace llob
