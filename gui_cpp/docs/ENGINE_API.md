# ENGINE_API — what a real backend has to provide

Companion to [`MODEL_HOOKS.md`](MODEL_HOOKS.md).  That file is the flat
inventory of every method the UI calls; this one is the **architectural
contract** a real `Model` implementation must honour: lifecycle, threading,
streaming, error semantics, and the gaps the current `Model` interface still
has when measured against a production backend.

> **Status:** spec, not yet implemented.  The shipped `MockModel` returns
> deterministic fake data (or empty sentinels with `LLOB_USE_MOCK_DATA=OFF`).
> Three concrete backends are sketched in §6 — pick one to bind first.

---

## 1. Implementation surface

A backend is a class that derives from `llob::Model` (`src/model/model.hpp`)
and overrides every virtual method.  The base class has no state; the
default implementations either return the no-data sentinel for the type or
no-op (mutators).  A backend that doesn't yet implement a hook is therefore
**not broken** — the UI just renders `—` placeholders for the unwired
fields.

Three groups of methods, each with its own threading + lifetime story:

| Group | Cardinality | Cost | Caching |
|---|---|---|---|
| **Pure-config getters** (`getModelInfo`, `getStateDict`, `getTensorMeta`, `getLoRAConfig`, …) | once per checkpoint | O(1) — read off the loaded model | cache results in the backend; UI re-asks every frame |
| **Per-frame samplers** (`getActivation`, `getAttentionPattern`, `getLiveActivations`, `getOutputLogits`, `getResidualSummary`, `getEngineMetrics`, …) | once per visible panel per frame | depends — see §3 | the backend decides; the UI does NOT memoise |
| **Mutators** (`pauseTraining`, `stepTraining`, `loadCheckpoint`, `saveProbe`, `exportSnapshot`, `startProbeTraining`, …) | event-driven | usually slow (file IO, GPU work) | run on the engine thread; UI fires and forgets |

A backend MUST be able to handle **all three groups concurrently** — the UI
thread will call config getters and per-frame samplers while a mutator is
in flight on the engine thread.

---

## 2. Lifecycle

### 2.1 Boot / shutdown

```
main()
 ├─ glfwInit + ImGui + AppState
 ├─ LoggerInit
 ├─ SettingsLoad
 ├─ Model* m = new YourBackend();        // construct empty (no checkpoint)
 ├─ AppState::loadFromModel(*m);          // pulls ModelInfo + sample tokens
 │                                          (no-op until a checkpoint is open)
 └─ main loop … on close:
     delete m;                            // dtor releases tensors / GPU
```

The constructor MUST be cheap — no checkpoint loading.  The UI is interactive
the moment `main()` returns; expensive setup happens inside
`loadCheckpoint` (§2.2).

### 2.2 Checkpoint open / close

The UI invokes a single new hook for both transitions:

```cpp
struct CheckpointResult {
    bool        ok;
    std::string error;       // populated only when ok == false
};
virtual CheckpointResult loadCheckpoint(const std::string& path);
virtual void             unloadCheckpoint();
```

Required behaviour:

1. **Synchronous result, asynchronous work.** `loadCheckpoint` must return
   `ok=true` *as soon as the file is validated* (header read, shape parsed,
   tokenizer found).  Heavy work (mmap'ing the weights, allocating GPU
   buffers, warming the KV cache) runs on a worker thread.  Until that
   completes, per-frame samplers may return `kNoFloat` / empty vectors —
   the UI will show `loading…` placeholders.
2. **Emit progress through the log channel.**  Use `drainEngineLogs()` to
   surface "loaded blocks.0…blocks.31" / "tokenizer ready" / "warmup
   complete" so the user sees what's happening.  Severity = `Info`;
   failures = `Error`.
3. **Repopulate AppState topology.** The UI calls `loadFromModel(m)` after
   a successful load to refresh `AppState.model` (`ModelInfo`).  The
   backend MUST have `getModelInfo()` ready by the time `loadCheckpoint`
   returns.
4. **`unloadCheckpoint` is idempotent.** Calling it on an empty backend is
   a no-op.  Calling it during `loadCheckpoint` cancels the load (engine
   thread checks a stop flag).

The current `Model` interface lacks both methods — they're the **first**
addition needed for a real backend.  Wire `File ▸ Open checkpoint` to
`loadCheckpoint` once Phase 3 (file dialogs) lands.

### 2.3 Switching backends

Today the code path is `MockModel mm; Model& model = mm;` — single
implementation chosen at compile time.  For a real workflow we want to
switch at runtime (e.g. swap from `HFTransformerEngine` to
`LlamaCppEngine`).  The recommended pattern:

```cpp
// main.cpp
std::unique_ptr<Model> model;
const auto choice = std::getenv("LLOB_BACKEND");   // mock | hf | llama_cpp | ...
if      (!choice || choice == std::string("mock"))     model = std::make_unique<MockModel>();
else if (choice == std::string("hf"))                  model = std::make_unique<HFTransformerEngine>();
else if (choice == std::string("llama_cpp"))           model = std::make_unique<LlamaCppEngine>();
else { LLOB_LOG_ERROR("init", "unknown LLOB_BACKEND=%s", choice); return 1; }
```

Add a `--backend` CLI flag once GLFW window args are parsed.  Switching
mid-session (without process restart) is intentionally out of scope — the
state to migrate (active tensor, ablations, probes) doesn't necessarily
have meaning across backends.

---

## 3. Threading model

The UI is single-threaded (the main GLFW loop).  Per-frame samplers are
called from the UI thread up to ~60 times a second, every visible panel.

### 3.1 Three-thread architecture (recommended)

```
                       ┌──────────────┐
                       │  UI thread   │  (main loop, ImGui::NewFrame …)
                       │              │
                       │  reads:      │
                       │  - ModelInfo │   (immutable post-load)
                       │  - cached    │
                       │    samplers  │   (mutex-protected snapshots)
                       │              │
                       │  writes:     │
                       │  - mutators  │   (push command onto engine queue)
                       │  - log ring  │   (mutex-protected)
                       └──────┬───────┘
                              │ command queue (lock-free SPSC)
                              ▼
                       ┌──────────────┐
                       │ Engine thread│   (forward pass, training step,
                       │              │    checkpoint load, …)
                       │              │
                       │  every step: │   - run forward
                       │              │   - update sample caches
                       │              │      (atomic swap or mutex)
                       │              │   - push log lines
                       │              │   - mark a snapshot version
                       └──────┬───────┘
                              │ optional: streaming tokens via ring
                              ▼
                       ┌──────────────┐
                       │ Stream thread│   (only for live generation)
                       │              │   - reads logits per step
                       │              │   - publishes token + top-k
                       └──────────────┘
```

### 3.2 Per-frame samplers MUST be cheap

A backend that runs a real forward pass inside `getAttentionPattern` will
freeze the UI to single-digit FPS.  Every per-frame sampler returns
**cached** data:

- Forward-pass outputs (activations, attention, logits) are captured **once
  per generation step** and stored in a per-layer cache.  Per-frame samplers
  read from the cache.
- Stats (`getHeadStats`, `getTrainingMetrics`, `getEngineMetrics`) are
  computed asynchronously and updated on a fixed cadence (4 Hz is plenty —
  the UI's `liveAnim` flag already gates 4 Hz visual ticks).

The cache must be safe to read concurrently with engine writes.  Two
patterns work:

- **Double-buffer.** Engine writes to buffer B, atomic-swaps `current` to B
  when done.  UI reads `current`.  No lock.  Cheapest for fixed-shape
  outputs (per-layer activations, attention matrices).
- **Mutex + shared_ptr<const T>.** Engine builds a new T, locks, swaps the
  shared_ptr.  UI grabs the shared_ptr under the lock and releases it.
  Slightly more allocation; works for variably-sized outputs (logit lens
  trajectories, feature lists).

### 3.3 Mutators run on the engine thread

`pauseTraining`, `stepTraining`, `loadCheckpoint`, `saveProbe`, etc.
return immediately on the UI thread; they push a command to the engine
queue and the engine processes it on its next loop tick.  The UI watches
for completion via:

- `getTrainingState().running` flips after `pauseTraining`/`resumeTraining`
- `getEngineMetrics().fwd_time_ms` updates after each forward pass
- `drainEngineLogs()` surfaces the engine's own log lines (incl. "saved
  probe to /path/foo.pt", "checkpoint loaded in 4.2s")

For long-running mutators (load, train), the engine SHOULD write a
progress fraction to a `getProgress()` hook (not yet in the interface;
add when needed) so the UI can show a progress bar.

### 3.4 Log fan-in

The Logger module (`src/logger.cpp`) is already thread-safe — multiple
threads can call `LLOB_LOG_*` concurrently.  But the engine thread should
NOT call `LLOB_LOG_*` directly when it holds a GIL / heavy lock; instead
push log lines to its own queue and let `drainEngineLogs()` (called once
per frame from the UI thread) hand them to the Logger:

```cpp
std::vector<LogEntry> YourBackend::drainEngineLogs() {
    std::vector<LogEntry> out;
    std::lock_guard<std::mutex> lk(mu_);
    out.swap(pending_logs_);
    return out;
}
```

main.cpp already pumps this every frame.

---

## 4. Per-area requirements

### 4.1 Activations / attention

Capture happens via PyTorch `register_forward_hook` (HF backend) or by
wrapping each block in instrumented code (llama.cpp / native).  The raw
captured tensors are stored in a per-step cache:

```python
# pseudocode for the HF backend
class CaptureBundle:
    attn:        Dict[layer, Tensor[heads, q, k]]   # post-softmax
    q, k, v:     Dict[(layer, head), Tensor[d_head]]
    resid_pre:   Dict[layer, Tensor[d_model]]
    resid_post:  Dict[layer, Tensor[d_model]]
    mlp_post:    Dict[layer, Tensor[d_mlp]]
    logits:      Tensor[vocab]
```

Per-frame samplers take a slice from the bundle (UI requests one head, one
token, etc.) and return it as `vector<float>` / `vector<vector<float>>`.

**Activations get pagination:** add an `offset` arg to `getActivation` so
`HexViewVirtual` can fetch true mid-tensor slices instead of CPU-slicing
the whole thing (Phase 6 leaves a TODO inline for this):

```cpp
// future signature
virtual std::vector<float> getActivation(int layer, int kind,
                                         std::size_t offset, std::size_t n);
```

### 4.2 Attention patterns

`getAttentionPattern(layer, head, seqLen, bias)` returns the post-softmax
attention matrix as a `vector<vector<float>>` of shape `[seqLen, seqLen]`.
Causal mask is the engine's responsibility — return a triangular matrix.

For long contexts (4k+) consider returning a downsampled / pooled matrix
when `seqLen > 256`.  The UI heatmap is screen-bound — there's no value
in a 4096×4096 matrix where each cell is sub-pixel.

### 4.3 Logit lens trajectory

`getLogitLensTrajectory(token, kLayers)` projects each layer's residual
through the unembed and reports per-layer top-k.  Prerequisites the
engine must compute:

- Cache `unembed = model.lm_head.weight` (or tie-weights with `wte`).
- For each layer L: `logits_L = unembed @ resid_post[L]` — top-k via
  partial sort.
- `is_resolved` marks the first layer where the eventual top-1 (final
  layer's argmax) becomes top-1.

This is cheap enough to run once per generation step (one mat-vec per
layer).

### 4.4 Steering / interventions

The current interface has `getSteering()` for read-only display.  When
a real backend supports steering, it needs:

```cpp
virtual void setSteering(const SteeringConfig& cfg);
virtual void clearSteering();
```

Same with **ablation** — currently the UI tracks ablated heads in
`AppState.ablatedHeads` but no `Model::*` method tells the engine to
actually zero those heads.  Add:

```cpp
virtual void setAblation(const std::vector<std::string>& ablated_heads,
                         const std::vector<std::string>& ablated_components);
```

Engine calls a forward-hook that masks the listed heads/components.  The
UI calls `setAblation` whenever the set changes (debounce — once per
~200ms is enough).

### 4.5 Training control

`pauseTraining` / `resumeTraining` / `stepTraining` / `resetTraining` /
`stopTraining` — engine implements these as flags on the training loop's
tick.  `getTrainingState().running` reflects the engine's actual state
(post-flag-application), not the UI's intent.

`getGradFlowPerLayer` reads `param.grad.norm()` per layer after each
backward pass.  Cache it; the UI reads every frame.

### 4.6 Probe / SAE training

`startProbeTraining(name, kind, location, dataset)` kicks off an
asynchronous training job.  The engine spawns a worker, returns
immediately.  Progress shows up via `getProbeTrainState(name)` — fields:

```cpp
struct ProbeTrainState {
    bool                training = false;
    int                 step;
    float               train_acc;
    float               val_acc;
    std::vector<float>  val_curve;     // last 100 val accuracies
};
```

`saveProbe(name)` serialises the probe (weights + meta) to
`./out/probes/<name>.pt + .json`.  Engine implementation: just `pickle.dump`
or libtorch's `torch::save` + a JSON sidecar.

---

## 5. Error handling

### 5.1 Guarantees

- `noexcept` is the default for all per-frame samplers.  Internal errors
  return the no-data sentinel for the type and emit a log line.
- `loadCheckpoint` returns `CheckpointResult{ok=false, error="..."}` on
  parse failure.  The UI surfaces the error in a modal.
- Mutators (`pauseTraining`, etc.) are best-effort.  The engine logs if
  it can't honour the request (e.g. `stepTraining` while not training).

### 5.2 OOM / device errors

CUDA OOM is the loudest failure mode.  When it happens:

1. Engine catches the runtime error in its forward / backward.
2. Pushes `LLOB_LOG_FATAL("oom", "CUDA OOM at layer %d during fwd")`.
3. Sets `getEngineMetrics().cuda_mem_used_GB = NaN` to signal the UI.
4. Stops the training loop (sets `running = false`).
5. UI surfaces the error toast (Phase 7 — already wired to `WARN+`).

OOM during `loadCheckpoint` returns `ok=false` with the message.

### 5.3 Cross-thread crashes

If the engine thread terminates via uncaught exception, the UI thread
should detect it and downgrade to no-data mode (every sampler returns
sentinels).  Pattern:

```cpp
std::atomic<bool> engine_alive_ = true;
void EngineMain() {
    try { … } catch (...) {
        engine_alive_ = false;
        LLOB_LOG_FATAL("engine", "engine thread died; UI in degraded mode");
    }
}
```

The UI MAY periodically check `engine_alive_` and grey out the menubar's
device tag.  Optional polish.

---

## 6. Concrete backend sketches

### 6.1 HF Transformers (Python via embedded interpreter or RPC)

**Embedded interpreter** (pybind11):
- Single process, GIL-managed.  Engine thread holds the GIL during
  forward passes; per-frame samplers acquire the GIL briefly to read
  cached tensors (released before returning).
- Pros: zero-copy access to cached PyTorch tensors via `torch::Tensor`
  C++ API.
- Cons: build complexity (Python dev headers), interpreter init cost.

**FastAPI proxy** (current React frontend's approach):
- Engine = separate Python process serving HTTP + WebSocket.
- C++ frontend issues HTTP GET for samplers, WebSocket for streams.
- Pros: clean separation, can colocate engine on a GPU box and run UI
  on a laptop.
- Cons: per-call latency (~1ms), JSON serialisation overhead.
- **The current `testing/gui/backend/` is exactly this**; the C++ port can
  reuse it almost verbatim.  Add a thin `HFProxyEngine : Model` that wraps
  HTTP/WS calls.  **Phase 1 is shipped — see [`HFPROXY_PLAN.md`](HFPROXY_PLAN.md)
  for the per-method status table and the staged plan.**

### 6.2 llama.cpp (native C++)

llama.cpp already has all the inference machinery; the missing piece is
**activation capture**.  Use the `cb_eval` callback (per-tensor compute
hook) to grab the post-softmax attention + residuals.  See the existing
[`reference_python_env`](../../testing/llm_surgeon/) work for the pattern.

Wrap llama.cpp's context in `class LlamaCppEngine : Model`:

```cpp
class LlamaCppEngine : public Model {
    llama_context* ctx_;
    std::vector<float> attn_cache_;       // populated by cb_eval
    // …
};
```

Pro: same process, no GIL, native fp16 / quantized inference.
Con: activation capture requires patching cb_eval to recognise tensor
names — fragile across llama.cpp versions.

### 6.3 libtorch (native C++ + PyTorch saved tensors)

libtorch exposes the full ATen tensor API in C++.  Loading is just
`torch::jit::load` for a TorchScript checkpoint, or a custom
deserialiser for safetensors.

Pro: matches HF model formats, ATen tensors are zero-copy.
Con: no built-in transformer architecture — you write the forward pass.
Practical only if you're already reimplementing the model anyway.

---

## 7. Hooks the UI needs that the current interface lacks

Cataloging the gaps the spec exposed.  These are the additions a real
backend will need beyond what's in `MODEL_HOOKS.md` today:

| Hook | Used by | Purpose |
|---|---|---|
| `loadCheckpoint(path) → CheckpointResult` | File ▸ Open (Phase 3) | open a model |
| `unloadCheckpoint()` | File ▸ Close, app shutdown | tear down |
| `getModelInfo() → ModelInfo` | `AppState::loadFromModel` post-open | populate topology |
| `getCurrentTokens() → vector<string>` | same | populate `sampleTokens` |
| `getActivation(layer, kind, offset, n)` | `inf.raw` HexViewVirtual | true mid-tensor paging (current sig fetches from offset 0 only) |
| `setAblation(heads, comps)` | UI on ablation set change | tell engine to zero those heads in fwd |
| `setSteering(cfg)` | UI on steering edit | activate steering vector |
| `clearSteering()` | UI on steering disable | deactivate |
| `getProgress() → {kind, step, total}` | future progress bar UI | long-running task progress |

When wiring a backend, declare the new struct types in
`src/model/model.hpp` next to the existing ones, add the virtual method,
mock-implement in `src/model/mock_model.cpp`, and update
[`MODEL_HOOKS.md`](MODEL_HOOKS.md).

---

## 8. Picking the first backend

Recommended order:

1. **Reuse the FastAPI proxy** (§6.1 second variant).  Lowest engineering
   cost — the Python backend already exists at `testing/gui/backend/` and
   the React frontend exercises every endpoint.  Write `HFProxyEngine` as
   a thin HTTP+WS wrapper.  Validates the architecture; gets the C++ build
   off `MockModel` fastest.
2. **Embedded llama.cpp** (§6.2) when you need lower latency or the
   FastAPI dependency becomes annoying.  Activation capture via `cb_eval`
   has a known recipe in `testing/llm_surgeon/`.
3. **Embedded pybind11 + HF** (§6.1 first variant) only if neither of the
   above is acceptable.  Build complexity is the main downside.

Each backend is a separate `Model` subclass selected via the `LLOB_BACKEND`
env / CLI.  No code rewrites required — the abstraction is exactly what
the polish phases of the C++ port were designed around.
