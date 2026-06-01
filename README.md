# llobotomy

Native C++/Dear ImGui frontend for local LLM inspection and intervention.
This is the rework that supersedes the React/FastAPI GUI that previously lived
at `testing/gui/` in the `skothr/llm` monorepo — the C++ shell absorbs the same
panels (architecture, inference, attention, probes, training, finetune,
datasets, raw tensors, logs) and redraws them with custom DrawList components.

**Status: rework-in-progress.** The shell, workspaces, and engine backends are
present and build, but the design is actively being reworked against new design
information. Treat panel layout and interaction contracts as in flux.

## Layout

```
gui_cpp/          the application (project(llobotomy), C++23, Dear ImGui)
llm_engine_cpp/   the engine library it links (llmengine::llm_engine)
CMakeLists.txt    thin workspace root: add_subdirectory(gui_cpp)
```

`gui_cpp/CMakeLists.txt` pulls in `llm_engine_cpp/` via a relative
`add_subdirectory(../llm_engine_cpp)`, so the two directories must stay
siblings. The top-level `CMakeLists.txt` exists so that configuring from the
workspace root resolves that relative path correctly.

## Build

```bash
cmake -S . -B build
cmake --build build
```

The binary lands at `build/gui_cpp/llobotomy`.

### Dependencies

- A C++23 compiler (gcc or clang).
- CMake ≥ 3.20.
- GLFW3 (discovered via `pkg-config`, target `glfw3`) and OpenGL.
- On Linux, X11, threads, and `dl` (resolved automatically).

Dear ImGui (`v1.92.7-docking`, pinned in `gui_cpp/.imgui-version`) is fetched at
configure time via CMake `FetchContent`. Override the ref with
`-DIMGUI_FETCH_REF=<tag>`, or point at a local clone with
`-DIMGUI_LOCAL_DIR=/path/to/imgui` to skip the fetch. `ImGuiFileDialog` (v0.6.8)
and the `imscoped.hpp` RAII guards are vendored under `gui_cpp/libs/`. The engine
library vendors `cpp-httplib` and `nlohmann/json` under `llm_engine_cpp/libs/`.

### Build options

| Option | Default | Effect |
|---|---|---|
| `LLOB_USE_MOCK_DATA` | `ON` | When ON, `MockModel` returns deterministic fake data so the UI looks alive in dev/screenshots. When OFF, mock getters return no-data sentinels (NaN / `{}`) so a release build can't ship mock numbers. The value cascades to the engine's `LLM_ENGINE_USE_MOCK_DATA`. Disable for any build linking a real backend. |
| `LLM_ENGINE_BUILD_LLAMA_CPP` | `OFF` | Compile the embedded `LlamaCppEngine` (links a pre-built `libllama.so`; set `-DLLAMA_CPP_ROOT=...`). With it OFF, the `llama_cpp` backend is unavailable but the build stays dependency-free. |

Example real-backend build:

```bash
cmake -S . -B build-real -DLLOB_USE_MOCK_DATA=OFF -DLLM_ENGINE_BUILD_LLAMA_CPP=ON
cmake --build build-real
```

## Run

The backend is selected at startup by the `LLOB_BACKEND` environment variable:

| `LLOB_BACKEND` | Backend | Relevant env |
|---|---|---|
| `mock` (default) | `MockModel` — deterministic fake data | — |
| `hf` | `HFProxyEngine` — HTTP/JSON to a FastAPI server | `LLOB_BACKEND_URL` (default `http://127.0.0.1:8000`) |
| `gguf` | `GgufInspectorEngine` — native read-only GGUF topology + state dict | `LLOB_GGUF_PATH` (auto-loads on startup) |
| `llama_cpp` | `LlamaCppEngine` — embedded llama.cpp inference + activation capture | `LLOB_GGUF_PATH`; requires `LLM_ENGINE_BUILD_LLAMA_CPP=ON` |

An unrecognized value logs a warning and falls back to `mock`.

```bash
LLOB_BACKEND=gguf LLOB_GGUF_PATH=/path/to/model.gguf ./build/gui_cpp/llobotomy
```

The engine library ships a `scripts/demo.sh` that runs the full end-to-end loop
(convert a cached HF model to GGUF, build with llama.cpp enabled, launch the
binary pointed at the GGUF). See `llm_engine_cpp/README.md`.

## Framework conventions

`gui_cpp` uses Dear ImGui directly (it does not link the `imgui-toolkit`
`ImTool::` foundation library at this point). The host-side conventions for
ImGui work in this lineage of projects live in the **imgui-toolkit** repo at
`/home/ai/ai-projects/imgui-toolkit`; its skill at
`imgui-toolkit/skill/SKILL.md` (now promoted to the
`imgui-skothr-toolkit-plugin` Claude Code plugin) is authoritative for those
conventions. Read it before writing or restructuring C++/ImGui code here.

## License

The application (`gui_cpp/`) and the workspace as a whole are licensed under
GPLv3 (see `LICENSE`). The bundled engine library `llm_engine_cpp/` carries its
own MIT `LICENSE`, and the vendored third-party components (Dear ImGui,
ImGuiFileDialog, cpp-httplib, nlohmann/json) retain their own upstream licenses.

Author: Michael Lannum.

