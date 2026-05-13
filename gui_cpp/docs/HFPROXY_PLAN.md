# HFProxyEngine — phased plan

Companion to [`ENGINE_API.md`](ENGINE_API.md) (the architectural contract)
and [`MODEL_HOOKS.md`](MODEL_HOOKS.md) (the flat hook inventory).  This
file tracks the staged implementation of `HFProxyEngine` — the first real
backend, a thin C++ wrapper around the existing FastAPI server at
`testing/gui/backend/`.

## Why this backend first

Per ENGINE_API.md §8: lowest engineering cost.  The Python backend already
exists (the React frontend exercises every endpoint), and a thin
HTTP+(eventually)WS wrapper validates the abstraction without the
build-complexity of pybind11 or the activation-capture fragility of
patched-`cb_eval` llama.cpp.

## Activation

```bash
LLOB_BACKEND=hf ./build-real/llobotomy
LLOB_BACKEND=hf LLOB_BACKEND_URL=http://10.0.0.5:8000 ./build-real/llobotomy
```

Default URL: `http://127.0.0.1:8000`.  Defaulting `LLOB_BACKEND` (or
unset) picks `MockModel`, preserving the existing dev workflow.

## Phase status

| Phase | Hooks | Status |
|---|---|---|
| 1 | `loadCheckpoint`, `unloadCheckpoint`, `drainEngineLogs` | shipped |
| 2 | `getModelInfo` (cache `/api/sessions/{n}/info`), `getCurrentTokens`, `getAttentionPattern`, `getQKVStats`, `getResidualSummary`, `getLogitLensTrajectory` | next |
| 3 | `setAblation` (→ `surgery` POST + commit), `setSteering`/`clearSteering` | TBD |
| 4 | WebSocket streams (`/ws/.../generate` for live token stream + `getOutputLogits`) | TBD |
| 5 | Backend additions: state-dict enumeration, weight stats, tensor pages | TBD |

## Endpoint mapping

What the backend serves vs. what the C++ `Model` interface needs.  "—"
means no backend route exists yet; that hook returns the no-data
sentinel until phase 5 adds one.

| `Model::*` method | FastAPI route | Status |
|---|---|---|
| `loadCheckpoint(path)` | `POST /api/sessions {name, model_id, mode}` | Phase 1 |
| `unloadCheckpoint()` | `DELETE /api/sessions/{name}` | Phase 1 |
| `getModelInfo` (via `AppState::loadFromModel`) | `GET /api/sessions/{name}/info` | Phase 2 |
| `getCurrentTokens` | `POST /api/sessions/{name}/tokenize` (+ optional `decode-ids`) | Phase 2 |
| `getAttentionPattern` | `POST /api/sessions/{name}/decode-head` | Phase 2 |
| `getQKVStats`, `getHeadStats` | `POST /api/sessions/{name}/decode-head` | Phase 2 |
| `getResidualSummary` | `POST /api/sessions/{name}/decode-residual` | Phase 2 |
| `getLogitLensTrajectory` | `WS /ws/sessions/{name}/logit-lens` | Phase 4 |
| `getMlpFeatures` | `POST /api/sessions/{name}/decode-neuron` | Phase 2 |
| `getOutputLogits` | `WS /ws/sessions/{name}/generate` (last frame) | Phase 4 |
| `setAblation` | `POST /api/sessions/{name}/surgery` + commit | Phase 3 |
| `setSteering` / `clearSteering` | `WS /ws/sessions/{name}/intervene` | Phase 4 |
| `getActivation` (paged) | `POST /api/sessions/{name}/decode-residual-grid` | Phase 2 |
| `getStateDict`, `getTensorMeta`, `getWeightSlice`, `getWeightHistogram`, `getTensorStats`, `getSingularValues` | — | Phase 5 (backend additions) |
| `getTrainingState`, `getTrainingMetrics`, `getTrainingLoss`, `getGradFlowPerLayer`, `getPerLayerLoss`, `pause/resume/step/reset/stopTraining` | — | not in scope of this backend |
| `getLoRAConfig`, `getOptimizerConfig`, `getDataConfig`, `getEvalDiff`, `getEvalLossCurve`, `getABSample`, `getDeltaWHeatmap` | — | not in scope of this backend |
| `getDatasets`, `getSample`, `getSampleStats`, `getDatasetDistribution`, `getTokenIds` | — | not in scope of this backend |
| `getFeatureLibrary`, `getFeatureCard`, `getFeatureExamples`, `getCoFiringFeatures`, `getSAETrainingMetrics`, `getProbeLibrary`, `getProbeTrainState`, `startProbeTraining`, `saveProbe`, `exportSnapshot` | — | not in scope of this backend |
| `getEngineMetrics` | derived (frame counter etc.) | Phase 2 partial |
| `drainEngineLogs` | local FIFO of HTTP errors | Phase 1 |

The "not in scope" rows correspond to React-frontend features that simply
don't exist on the FastAPI side today (training loop, LoRA finetune,
dataset browser, SAE training).  Their Model::* hooks return the no-data
sentinel and the corresponding workspace panels show `—` placeholders.
That's fine — the architecture/inference/attention/probes workspaces will
still light up with real model data in Phase 2, which is where the
demonstrated value lives.

## Threading

Phase 1 is entirely synchronous: every method runs on the UI thread.
`loadCheckpoint` blocks the UI for 10s+ on a multi-billion-parameter
model — acceptable for a one-shot user action.

Phase 2 introduces the ENGINE_API.md §3.1 three-thread architecture: a
worker thread that polls samplers asynchronously and writes into a
double-buffered cache that the UI thread reads lock-free.

## Class layout

`HFProxyEngine` inherits from `MockModel` (not `Model` directly) so the
~50 unwired methods inherit MockModel's defaults.  In `LLOB_USE_MOCK_DATA=OFF`
builds those defaults return the no-data sentinel for the type; in
`LLOB_USE_MOCK_DATA=ON` builds they return mock data.  The mock-ON case
is intentionally useful for incremental development — load a real model,
verify the topology populates, while the rest of the UI continues to
show familiar mock data instead of `—` placeholders.

The class uses PIMPL to keep `httplib.h` (~10K LOC, ~700KB) out of the
public header.  See `src/model/hf_proxy_engine.{hpp,cpp}`.

## End-to-end smoke (manual)

```bash
# Terminal 1 — bring up the FastAPI backend
cd testing/gui
.venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000

# Terminal 2 — launch llobotomy with the hf backend
cd testing/gui_cpp
LLOB_BACKEND=hf ./build-real/llobotomy

# In the UI: File ▸ Open checkpoint → type "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#                                   → Open
# Watch the Logs workspace for "[hf] loaded session 'llobotomy' model=..."
# (Phase 2 will populate the architecture workspace with real layer counts;
#  Phase 1 is wiring-only and only logs the load.)
```

## Dependencies vendored under `libs/`

- `libs/cpp-httplib/` — header-only HTTP client, MIT, v0.44.0
- `libs/nlohmann_json/` — header-only JSON, MIT, v3.12.0

Both are SYSTEM includes via `target_include_directories(... SYSTEM ...)`
so their warnings don't drown out our own code's strict-warning channel.

## What this backend can't do (and won't pretend to)

The FastAPI server is a model-surgery + per-token inspection tool, not a
training runtime.  The training / finetune / datasets / SAE-training
workspaces will continue to show no-data sentinels under this backend.
Wiring those requires either:

- backend-side additions (a training loop, dataset enumeration, etc.);
- or a different concrete `Model` subclass that talks to a runtime which
  has those concepts (e.g. a libtorch in-process backend with its own
  optimizer).

That's a deliberate scope choice — the goal here is to validate the
abstraction with a backend that already exists, not to backfill
end-to-end functionality across every workspace.
