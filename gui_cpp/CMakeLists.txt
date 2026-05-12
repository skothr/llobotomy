cmake_minimum_required(VERSION 3.20)
project(llobotomy CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

# When ON, MockModel returns deterministic fake data (so the UI looks
# alive in dev / screenshots).  When OFF, MockModel returns empty / NaN
# sentinels — the UI shows "no data" placeholders so a release build
# can't accidentally ship with mock numbers.  Always disable for any
# build that links a real engine backend.
option(LLOB_USE_MOCK_DATA "Have MockModel return deterministic mock data" ON)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
find_package(PkgConfig REQUIRED)
# IMPORTED_TARGET pulls in transitive deps (X11, dl, pthread...) on Linux.
pkg_check_modules(GLFW REQUIRED IMPORTED_TARGET glfw3)
find_package(OpenGL REQUIRED)

# Vendored Dear ImGui (libs/imgui, v1.92.7 docking).
set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs/imgui)
add_library(imgui STATIC
  ${IMGUI_DIR}/imgui.cpp
  ${IMGUI_DIR}/imgui_demo.cpp
  ${IMGUI_DIR}/imgui_draw.cpp
  ${IMGUI_DIR}/imgui_tables.cpp
  ${IMGUI_DIR}/imgui_widgets.cpp
  ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
  ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui SYSTEM PUBLIC
  ${IMGUI_DIR} ${IMGUI_DIR}/backends
)
target_link_libraries(imgui PUBLIC OpenGL::GL PkgConfig::GLFW)
# Local libglfw3.a is static — explicitly add its transitive Linux deps so
# the linker can resolve X11 / threading / dl symbols.
if(UNIX AND NOT APPLE)
  find_package(Threads REQUIRED)
  find_package(X11 REQUIRED)
  target_link_libraries(imgui PUBLIC ${X11_LIBRARIES} Threads::Threads ${CMAKE_DL_LIBS})
endif()

# Vendored ImGuiFileDialog v0.6.8 (libs/ImGuiFileDialog).  MIT licensed;
# see libs/ImGuiFileDialog/LICENSE.  Single-translation-unit build; the
# config header next to the source is the empty default (we do NOT enable
# bookmarks / thumbnails — keep the dependency surface minimal).
add_library(imgui_file_dialog STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/libs/ImGuiFileDialog/ImGuiFileDialog.cpp
)
target_include_directories(imgui_file_dialog SYSTEM PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/libs/ImGuiFileDialog
)
target_link_libraries(imgui_file_dialog PUBLIC imgui)

# ---------------------------------------------------------------------------
# llobotomy executable
# ---------------------------------------------------------------------------
add_executable(llobotomy
  src/main.cpp
  src/style.cpp
  src/appstate.cpp
  src/logger.cpp
  src/model/mock_model.cpp
  src/ui/colormap.cpp
  src/ui/chrome.cpp
  src/ui/dialogs.cpp
  src/ui/dockhost.cpp
  src/ui/file_dialogs.cpp
  src/ui/settings.cpp
  src/ui/sidecar.cpp
  src/ui/widgets.cpp
  src/workspaces/architecture.cpp
  src/workspaces/inference.cpp
  src/workspaces/attention.cpp
  src/workspaces/probes.cpp
  src/workspaces/training.cpp
  src/workspaces/finetune.cpp
  src/workspaces/datasets.cpp
  src/workspaces/raw_tensors.cpp
  src/workspaces/logs.cpp
)
target_include_directories(llobotomy PRIVATE src libs)
target_link_libraries(llobotomy PRIVATE imgui imgui_file_dialog)
target_compile_options(llobotomy PRIVATE
  -Wall -Wextra -Wpedantic
  -Wno-c99-designator
)
if(LLOB_USE_MOCK_DATA)
  target_compile_definitions(llobotomy PRIVATE LLOB_USE_MOCK_DATA=1)
endif()
