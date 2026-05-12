// imscoped.hpp
//
// Drop-in RAII scope guards for Dear ImGui (v1.92.x docking).
// Header-only, zero dependencies beyond <imgui.h>. C++17 compatible (uses
// [[nodiscard]]); benefits from C++23 if available but does not require it.
//
// =============================================================================
// Why this exists
// =============================================================================
// Dear ImGui's API uses Begin*/End* and Push*/Pop* pairs that are easy to leave
// unbalanced — especially across early returns, exceptions, and tangled control
// flow. The cost of a missed End/Pop is a debug assertion at best, a stack
// corruption / silent UI breakage at worst.
//
// These guards make the pairing impossible to forget: every Begin/Push happens
// in a constructor, every End/Pop happens in the destructor, and the destructor
// runs even on early return / throw.
//
// =============================================================================
// The two destructor patterns (and why)
// =============================================================================
// Dear ImGui's Begin/End rules are NOT uniform across the API. Per imgui.h:
//
//   "Begin and BeginChild are the only odd ones out": you ALWAYS call End() /
//   EndChild() regardless of the Begin*'s return value. Every other Begin*X*
//   only pairs with End*X* when the Begin call returned true.
//
// So:
//   ImScoped::Window / Child         → ALWAYS-end (destructor calls End/EndChild
//                                      unconditionally)
//   Everything else (Menu, Popup,    → CONDITIONAL-end (destructor calls End*X*
//   Combo, ListBox, Table, TabBar,    only if the Begin*X* returned true)
//   TabItem, Tooltip, MenuBar,
//   DragDropSource/Target, etc.)
//
// The conditional-end guards expose `open` (a bool) and an explicit operator
// bool() so you write:
//
//     if (auto m = ImScoped::Menu("File")) {
//         ImGui::MenuItem("New");
//         ImGui::MenuItem("Open");
//     }   // ~Menu() calls EndMenu() iff open == true
//
// The always-end guards do NOT skip work via short-circuit: you must still
// check the bool to decide whether to *submit* widgets, but the End() call is
// guaranteed to run regardless.
//
//     if (auto w = ImScoped::Window("Tools")) {
//         ImGui::Text("...");   // only submitted when visible
//     }   // ~Window() calls End() unconditionally
//
// =============================================================================
// Usage notes
// =============================================================================
// - These are MOVE-DELETED. A scope guard belongs to one scope. If you find
//   yourself wanting to move one, you probably want a unique_ptr<T> with a
//   custom deleter instead — but at that point you've left the scope-guard
//   pattern behind.
// - These are NOT exception-safe in a "throw across the body" sense for
//   ImGui's internal state — ImGui itself isn't exception-safe in many places.
//   They DO ensure End/Pop runs on stack unwind, which is enough to keep your
//   process alive long enough to log the error.
// - The IDs / names / flags you pass through verbatim — see imgui.h for
//   full signatures.
// - Style/Item count parameters: each guard pops exactly the count it pushed.
//   For Push(N) / Pop(N) bulk operations, either nest guards or call ImGui's
//   functions directly.
//
// =============================================================================

#ifndef IMSCOPED_HPP
#define IMSCOPED_HPP

#include <imgui.h>

namespace ImScoped {

// -----------------------------------------------------------------------------
// Helper: deleted copy + move boilerplate. The guards are scope-bound.
// -----------------------------------------------------------------------------
#define IMSCOPED_NONCOPYABLE_NONMOVABLE(T)              \
    T(const T&)            = delete;                    \
    T& operator=(const T&) = delete;                    \
    T(T&&)                 = delete;                    \
    T& operator=(T&&)      = delete

// =============================================================================
// ALWAYS-END guards: End/EndChild run unconditionally (per upstream rule).
// =============================================================================

class Window {
public:
    bool open;
    explicit Window(const char* name, bool* p_open = nullptr, ImGuiWindowFlags flags = 0)
        : open(ImGui::Begin(name, p_open, flags)) {}
    ~Window() { ImGui::End(); }
    IMSCOPED_NONCOPYABLE_NONMOVABLE(Window);
    [[nodiscard]] explicit operator bool() const noexcept { return open; }
};

class Child {
public:
    bool open;
    Child(const char* str_id,
          const ImVec2& size = ImVec2(0, 0),
          ImGuiChildFlags child_flags = 0,
          ImGuiWindowFlags window_flags = 0)
        : open(ImGui::BeginChild(str_id, size, child_flags, window_flags)) {}
    Child(ImGuiID id,
          const ImVec2& size = ImVec2(0, 0),
          ImGuiChildFlags child_flags = 0,
          ImGuiWindowFlags window_flags = 0)
        : open(ImGui::BeginChild(id, size, child_flags, window_flags)) {}
    ~Child() { ImGui::EndChild(); }
    IMSCOPED_NONCOPYABLE_NONMOVABLE(Child);
    [[nodiscard]] explicit operator bool() const noexcept { return open; }
};

// =============================================================================
// CONDITIONAL-END guards: End* runs only if the corresponding Begin* returned
// true. The bool conversion / `open` field is what you check to skip the body.
// =============================================================================

class MainMenuBar {
public:
    bool open;
    MainMenuBar() : open(ImGui::BeginMainMenuBar()) {}
    ~MainMenuBar() { if (open) ImGui::EndMainMenuBar(); }
    IMSCOPED_NONCOPYABLE_NONMOVABLE(MainMenuBar);
    [[nodiscard]] explicit operator bool() const noexcept { return open; }
};

class MenuBar {
public:
    bool open;
    MenuBar() : open(ImGui::BeginMenuBar()) {}
    ~MenuBar() { if (open) ImGui::EndMenuBar(); }
    IMSCOPED_NONCOPYABLE_NONMOVABLE(MenuBar);
    [[nodiscard]] explicit operator bool() const noexcept { return open; }
};

class Menu {
public:
    bool open;
    explicit Menu(const char* label, bool enabled = true)
        : open(ImGui::BeginMenu(label, enabled)) {}
    ~Menu() { if (open) ImGui::EndMenu(); }
    IMSCOPED_NONCOPYABLE_NONMOVABLE(Menu);
    [[nodiscard]] explicit operator bool() const noexcept { return open; }
};

class Combo {
public:
    bool open;
    Combo(const char* label, const char* preview_value, ImGuiComboFlags flags = 0)
        : open(ImGui::BeginCombo(label, preview_value, flags)) {}
    ~Combo() { if (open) ImGui::EndCombo(); }
    IMSCOPED_NONCOPYABLE_NONMOVABLE(Combo);
    [[nodiscard]] explicit operator bool() const noexcept { return open; }
};

class ListBox {
public:
    bool open;
    ListBox(const char* label, const ImVec2& size = ImVec2(0, 0))
        : open(ImGui::BeginListBox(label, size)) {}
    ~ListBox() { if (open) ImGui::EndListBox(); }
    IMSCOPED_NONCOPYABLE_NONMOVABLE(ListBox);
    [[nodiscard]] explicit operator bool() const noexcept { return open; }
};

class Tooltip {
public:
    bool open;
    Tooltip() : open(ImGui::BeginTooltip()) {}
    ~Tooltip() { if (open) ImGui::EndTooltip(); }
    IMSCOPED_NONCOPYABLE_NONMOVABLE(Tooltip);
    [[nodiscard]] explicit operator bool() const noexcept { return open; }
};

class ItemTooltip {
public:
    bool open;
    ItemTooltip() : open(ImGui::BeginItemTooltip()) {}
    ~ItemTooltip() { if (open) ImGui::EndTooltip(); }
    IMSCOPED_NONCOPYABLE_NONMOVABLE(ItemTooltip);
    [[nodiscard]] explicit operator bool() const noexcept { return open; }
};

class Popup {
public:
    bool open;
    explicit Popup(const char* str_id, ImGuiWindowFlags flags = 0)
        : open(ImGui::BeginPopup(str_id, flags)) {}
    ~Popup() { if (open) ImGui::EndPopup(); }
    IMSCOPED_NONCOPYABLE_NONMOVABLE(Popup);
    [[nodiscard]] explicit operator bool() const noexcept { return open; }
};

class PopupModal {
public:
    bool open;
    explicit PopupModal(const char* name, bool* p_open = nullptr, ImGuiWindowFlags flags = 0)
        : open(ImGui::BeginPopupModal(name, p_open, flags)) {}
    ~PopupModal() { if (open) ImGui::EndPopup(); }
    IMSCOPED_NONCOPYABLE_NONMOVABLE(PopupModal);
    [[nodiscard]] explicit operator bool() const noexcept { return open; }
};

class PopupContextItem {
public:
    bool open;
    explicit PopupContextItem(const char* str_id = nullptr, ImGuiPopupFlags flags = 1)
        : open(ImGui::BeginPopupContextItem(str_id, flags)) {}
    ~PopupContextItem() { if (open) ImGui::EndPopup(); }
    IMSCOPED_NONCOPYABLE_NONMOVABLE(PopupContextItem);
    [[nodiscard]] explicit operator bool() const noexcept { return open; }
};

class PopupContextWindow {
public:
    bool open;
    explicit PopupContextWindow(const char* str_id = nullptr, ImGuiPopupFlags flags = 1)
        : open(ImGui::BeginPopupContextWindow(str_id, flags)) {}
    ~PopupContextWindow() { if (open) ImGui::EndPopup(); }
    IMSCOPED_NONCOPYABLE_NONMOVABLE(PopupContextWindow);
    [[nodiscard]] explicit operator bool() const noexcept { return open; }
};

class PopupContextVoid {
public:
    bool open;
    explicit PopupContextVoid(const char* str_id = nullptr, ImGuiPopupFlags flags = 1)
        : open(ImGui::BeginPopupContextVoid(str_id, flags)) {}
    ~PopupContextVoid() { if (open) ImGui::EndPopup(); }
    IMSCOPED_NONCOPYABLE_NONMOVABLE(PopupContextVoid);
    [[nodiscard]] explicit operator bool() const noexcept { return open; }
};

class Table {
public:
    bool open;
    Table(const char* str_id, int columns, ImGuiTableFlags flags = 0,
          const ImVec2& outer_size = ImVec2(0.0f, 0.0f), float inner_width = 0.0f)
        : open(ImGui::BeginTable(str_id, columns, flags, outer_size, inner_width)) {}
    ~Table() { if (open) ImGui::EndTable(); }
    IMSCOPED_NONCOPYABLE_NONMOVABLE(Table);
    [[nodiscard]] explicit operator bool() const noexcept { return open; }
};

class TabBar {
public:
    bool open;
    explicit TabBar(const char* str_id, ImGuiTabBarFlags flags = 0)
        : open(ImGui::BeginTabBar(str_id, flags)) {}
    ~TabBar() { if (open) ImGui::EndTabBar(); }
    IMSCOPED_NONCOPYABLE_NONMOVABLE(TabBar);
    [[nodiscard]] explicit operator bool() const noexcept { return open; }
};

class TabItem {
public:
    bool open;
    explicit TabItem(const char* label, bool* p_open = nullptr, ImGuiTabItemFlags flags = 0)
        : open(ImGui::BeginTabItem(label, p_open, flags)) {}
    ~TabItem() { if (open) ImGui::EndTabItem(); }
    IMSCOPED_NONCOPYABLE_NONMOVABLE(TabItem);
    [[nodiscard]] explicit operator bool() const noexcept { return open; }
};

class DragDropSource {
public:
    bool open;
    explicit DragDropSource(ImGuiDragDropFlags flags = 0)
        : open(ImGui::BeginDragDropSource(flags)) {}
    ~DragDropSource() { if (open) ImGui::EndDragDropSource(); }
    IMSCOPED_NONCOPYABLE_NONMOVABLE(DragDropSource);
    [[nodiscard]] explicit operator bool() const noexcept { return open; }
};

class DragDropTarget {
public:
    bool open;
    DragDropTarget() : open(ImGui::BeginDragDropTarget()) {}
    ~DragDropTarget() { if (open) ImGui::EndDragDropTarget(); }
    IMSCOPED_NONCOPYABLE_NONMOVABLE(DragDropTarget);
    [[nodiscard]] explicit operator bool() const noexcept { return open; }
};

// =============================================================================
// Push/Pop guards (no return value to check; the guard always runs).
// =============================================================================

class ID {
public:
    explicit ID(const char* id)        { ImGui::PushID(id); }
    explicit ID(const void* id)        { ImGui::PushID(id); }
    explicit ID(int id)                { ImGui::PushID(id); }
    ID(const char* begin, const char* end) { ImGui::PushID(begin, end); }
    ~ID() { ImGui::PopID(); }
    IMSCOPED_NONCOPYABLE_NONMOVABLE(ID);
};

class Font {
public:
    // Pass nullptr for `font` to keep the current font; pass 0.0f for
    // `size_unscaled` to keep the current size. Both fields were generalized
    // in v1.92's font-system rework.
    explicit Font(ImFont* font, float size_unscaled = 0.0f) {
        ImGui::PushFont(font, size_unscaled);
    }
    ~Font() { ImGui::PopFont(); }
    IMSCOPED_NONCOPYABLE_NONMOVABLE(Font);
};

class StyleColor {
public:
    StyleColor(ImGuiCol idx, ImU32 col)        { ImGui::PushStyleColor(idx, col); }
    StyleColor(ImGuiCol idx, const ImVec4& col){ ImGui::PushStyleColor(idx, col); }
    ~StyleColor() { ImGui::PopStyleColor(1); }
    IMSCOPED_NONCOPYABLE_NONMOVABLE(StyleColor);
};

class StyleVar {
public:
    StyleVar(ImGuiStyleVar idx, float v)         { ImGui::PushStyleVar(idx, v); }
    StyleVar(ImGuiStyleVar idx, const ImVec2& v) { ImGui::PushStyleVar(idx, v); }
    ~StyleVar() { ImGui::PopStyleVar(1); }
    IMSCOPED_NONCOPYABLE_NONMOVABLE(StyleVar);
};

class ItemFlag {
public:
    ItemFlag(ImGuiItemFlags option, bool enabled) {
        ImGui::PushItemFlag(option, enabled);
    }
    ~ItemFlag() { ImGui::PopItemFlag(); }
    IMSCOPED_NONCOPYABLE_NONMOVABLE(ItemFlag);
};

class ItemWidth {
public:
    explicit ItemWidth(float w) { ImGui::PushItemWidth(w); }
    ~ItemWidth() { ImGui::PopItemWidth(); }
    IMSCOPED_NONCOPYABLE_NONMOVABLE(ItemWidth);
};

class TextWrapPos {
public:
    explicit TextWrapPos(float wrap_local_pos_x = 0.0f) {
        ImGui::PushTextWrapPos(wrap_local_pos_x);
    }
    ~TextWrapPos() { ImGui::PopTextWrapPos(); }
    IMSCOPED_NONCOPYABLE_NONMOVABLE(TextWrapPos);
};

class Group {
public:
    Group()  { ImGui::BeginGroup(); }
    ~Group() { ImGui::EndGroup(); }
    IMSCOPED_NONCOPYABLE_NONMOVABLE(Group);
};

class Disabled {
public:
    explicit Disabled(bool disabled = true) { ImGui::BeginDisabled(disabled); }
    ~Disabled() { ImGui::EndDisabled(); }
    IMSCOPED_NONCOPYABLE_NONMOVABLE(Disabled);
};

class ClipRect {
public:
    ClipRect(const ImVec2& clip_min,
             const ImVec2& clip_max,
             bool intersect_with_current_clip_rect) {
        ImGui::PushClipRect(clip_min, clip_max, intersect_with_current_clip_rect);
    }
    ~ClipRect() { ImGui::PopClipRect(); }
    IMSCOPED_NONCOPYABLE_NONMOVABLE(ClipRect);
};

#undef IMSCOPED_NONCOPYABLE_NONMOVABLE

}  // namespace ImScoped

#endif  // IMSCOPED_HPP
