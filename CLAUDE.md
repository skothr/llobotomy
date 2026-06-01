# HARD RULE: every session MUST work in a git worktree

**Not a guideline. A hard rule.** Sessions that share the main checkout
let uncommitted work from one session get swept into another's `git add`
and commit. Worktrees are the fix.

## Pre-flight check — BEFORE your first edit/write/bash-write

Run this as the very first thing in any session at this project:

```bash
GIT_DIR=$(cd "$(git rev-parse --git-dir)" && pwd -P)
GIT_COMMON=$(cd "$(git rev-parse --git-common-dir)" && pwd -P)
[ "$GIT_DIR" = "$GIT_COMMON" ] && echo "MAIN CHECKOUT — MUST CREATE WORKTREE" || echo "in worktree — proceed"
```

If it says "MAIN CHECKOUT": **STOP. Do not edit any file.** Create a
worktree at `.claude/worktrees/<scope>/` on its own branch first (e.g.
via `EnterWorktree name=<scope>`). `.claude/` is gitignored.

The only commits permitted directly on the default branch (`master`) in
the main checkout are integration commits: merge commits (`gh pr merge`)
or convention-establishing changes to `CLAUDE.md` / `.gitignore` itself.
Everything else — even one-line fixes — goes through a worktree.

## Session lifecycle

1. `EnterWorktree name=<scope>` — creates the worktree and switches the
   session into it. Conventional branch prefixes: `feat/`, `fix/`,
   `refactor/`, `docs/`, `session/`.
2. All `Edit`/`Write`/`Bash` calls now operate inside the worktree;
   master in the main checkout is untouched. Never edit files in another
   session's sibling worktree (read-only inspection is fine).
3. End: `git push -u origin <branch>`, open a PR with `gh pr create`,
   and merge to `master` via PR. After merge, remove the worktree
   (`git worktree remove .claude/worktrees/<scope>`).

---

# llobotomy — native C++/Dear ImGui LLM inspection frontend

This repo is the rework superseding the prior React/FastAPI GUI. Two
sibling directories:

- `gui_cpp/` — the application (`project(llobotomy)`, C++23, Dear ImGui).
- `llm_engine_cpp/` — the engine library it links
  (`llmengine::llm_engine`): abstract `Model` interface, `MockModel`,
  `HFProxyEngine`, `GgufInspectorEngine`, `LlamaCppEngine`.

`gui_cpp` pulls in the engine via a relative
`add_subdirectory(../llm_engine_cpp)`, so the two stay siblings. The
top-level `CMakeLists.txt` is a thin workspace root
(`add_subdirectory(gui_cpp)`) so `cmake -S . -B build` from the root
resolves that relative path.

## Build / run

```bash
cmake -S . -B build && cmake --build build   # binary: build/gui_cpp/llobotomy
```

- `LLOB_USE_MOCK_DATA` (default ON) — `MockModel` returns deterministic
  fake data; OFF returns no-data sentinels so a release build can't ship
  mock numbers. Cascades to the engine's `LLM_ENGINE_USE_MOCK_DATA`.
  Disable for any build linking a real backend.
- `LLM_ENGINE_BUILD_LLAMA_CPP` (default OFF) — compile the embedded
  `LlamaCppEngine` (links a pre-built `libllama.so`; set `LLAMA_CPP_ROOT`).
- Runtime backend chosen by `LLOB_BACKEND` env: `mock` (default), `hf`
  (`LLOB_BACKEND_URL`), `gguf` / `llama_cpp` (`LLOB_GGUF_PATH`). Unknown
  values warn and fall back to `mock`.

## C++ / Dear ImGui conventions

`gui_cpp` uses Dear ImGui directly (pinned `v1.92.7-docking` via CMake
`FetchContent`; see `gui_cpp/.imgui-version`). It does **not** currently
link the imgui-toolkit `ImTool::` library — but the host-side ImGui
conventions for this lineage of projects are authoritative in the
imgui-toolkit repo at `/home/ai/ai-projects/imgui-toolkit`, via its skill
`imgui-toolkit/skill/SKILL.md` (promoted to the
`imgui-skothr-toolkit-plugin` plugin). Read it before writing or
restructuring C++/ImGui code here.

- Pair every `Begin*`/`Push*` with the RAII guards in
  `gui_cpp/libs/imscoped.hpp` rather than hand-balancing `End*`/`Pop*`.
- The shell (`src/main.cpp`) owns window/loop bring-up, menubar, project
  and workspace tabs, shortcuts, and dispatch. Per-workspace layout lives
  under `src/workspaces/`; shared widgets under `src/ui/`.
- Strict warnings are on (`-Wall -Wextra -Wpedantic`); vendored deps are
  marked `SYSTEM` so their warnings don't leak. `-Wno-c99-designator` is
  clang-only (gcc would error on the unknown flag).

### Honest-empty contract (engine)

Concrete backends inherit `Model` directly, never `MockModel`.
Unimplemented getters return the no-data sentinel (NaN / `{}` / `kNoInt`),
never a silent fall-through to mock data. `getCapabilities()` stays
conservative-false until `loadCheckpoint` succeeds. Locked in by the
engine's `tests/test_no_mock_leak.cpp`.

## Engine tests

The engine's own test suite builds only when `llm_engine_cpp` is the
top-level CMake project (not when `gui_cpp` adds it as a subdirectory),
so the GUI build stays clean of test binaries. To run them:

```bash
cmake -S llm_engine_cpp -B llm_engine_cpp/build && cmake --build llm_engine_cpp/build
./llm_engine_cpp/scripts/verify.sh --llama   # 4-config matrix (smoke + deep)
```

Deep tests env-gate on `LLOB_DEEP_GGUF_PATH`, `LLOB_DEEP_LLAMA_GGUF_PATH`,
and `LLOB_INTEGRATION_TEST=1` + a live FastAPI backend. See
`llm_engine_cpp/docs/INTEGRATION_TESTS.md`.

## clangd

`gui_cpp/.clangd` reads `compile_commands.json` from `build/`
(`CMAKE_EXPORT_COMPILE_COMMANDS` is ON). Configure into `build/` for
clangd to resolve includes; `UnusedIncludes: Strict` is set, so prune
includes flagged unused.

