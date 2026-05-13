# HFProxy Phase 3 — intervention

> Status: not started. Follows Phase 2 (capture flow, shipped 2026-05-13).
> This doc is the implementation contract.

## Goal

Wire `Model::setAblation` / `setSteering` / `clearSteering` on
`HFProxyEngine` so the UI's intervention controls actually mask heads
and apply steering vectors during the next forward pass. The
substrate already carries the structured representation
(`InterventionSet::ablated_heads` as `vector<AttentionHeadRef>`,
`steering` as `SteeringConfig`); Phase 3 makes the mutators flow
through to the Python runtime.

## Python side

Two existing FastAPI surfaces to reuse:

- `POST /api/sessions/{n}/surgery` — already accepts a single op
  (`zero_head`, `zero_component`, etc.) and queues it. Used by the
  React frontend's intervention queue today.
- `POST /api/sessions/{n}/surgery/commit` — flushes queued ops to
  the model (irreversibly writes a delta into the loaded weights).

Phase 3 doesn't change the wire format — it just exposes them to the
C++ engine. No new Python code needed unless `setSteering` requires a
new endpoint (the existing `/intervene` WS handler runs steering as
part of a generation stream; for capture-only sessions we'd need a
new `POST /api/sessions/{n}/steering` that installs a vector without
generating).

**Recommend:** add `POST /api/sessions/{n}/steering` (idempotent
install + clear via DELETE) so the C++ flow doesn't need a WS. Body:
`{layer: str, alpha: float, source: str, vector: [floats] | null}`.

## C++ side

In `HFProxyEngine::Impl`, track the pending intervention set under
the existing mutex. Three new methods, all worker-thread-dispatched:

```cpp
void HFProxyEngine::setAblation(
    const std::vector<std::string>& head_canonical,
    const std::vector<std::string>& component_canonical) override
{
    std::lock_guard lk(m_impl->mu);
    m_impl->pending_intervention = { /* parsed refs */ };
    m_impl->dirty_intervention   = true;
    m_impl->wake_cv.notify_all();   // worker will pick it up
}
```

The worker, on the `dirty_intervention` flag:

1. Diff pending against currently-installed (`m_view.surgery`).
2. For each newly-added head: POST `/surgery` with
   `{op:"zero_head", layer:L, head:H}`.
3. For each newly-removed head: POST `/surgery/revert` with that op
   index.
4. After all ops queued, POST `/surgery/commit` to make them effective.
5. On success: write `pending_intervention` into `m_view.surgery`
   (under mutex). On HTTP failure: log + leave `m_view.surgery`
   unchanged (the UI shows stale state; user can retry).

`setSteering` is simpler: one POST per call (or DELETE for
`clearSteering`); update `m_view.surgery.steering` on success.

## Capture invalidation

Any successful intervention change invalidates the current
`CaptureBundle` (the bytes were captured under a different
intervention state). The worker should:

1. `m_view.current.store(nullptr)` after commit succeeds.
2. The UI sees `view().current.load() == nullptr` and renders
   "capture stale, re-run prompt" placeholders.
3. The next `setActivePrompt` call repopulates.

Alternative: auto-re-capture on intervention change. Probably too
aggressive — the user may want to tweak the ablation set several
times before re-running. Leave it as explicit.

## Capability bit

After Phase 3 lands, `getCapabilities().has_intervention = true`.

## Tests

**Smoke** in `tests/test_hf_proxy_intervention.cpp`: dead-URL engine,
`setAblation({"blocks.5.attn.head.3"}, {})` → no crash, no
exception; `m_view.surgery.ablated_heads` stays empty (no successful
commit possible against dead URL); log shows the HTTP failure with
non-fatal severity.

**Deep** in same file, gated by `LLOB_INTEGRATION_TEST=1`: live
backend, load TinyLlama, baseline `getAttentionPattern(5, 3, ...)`,
`setAblation({head.canonical()}, {})`, wait for commit, re-capture,
verify the ablated head's attention is now zeroed.

## Open questions

1. **Atomicity of multi-op intervention** — if the user sets 8
   ablations at once, do we want 8 individual `/surgery` POSTs +
   one `/surgery/commit`, or a batch op? The existing FastAPI side
   queues, so individual POSTs are fine. Performance: 8 round-trips
   at ~10ms each = 80ms before commit. Acceptable.

2. **Surgery delta storage in `ModelView`** — `InterventionSet::weight_deltas`
   is the canonical home for weight-modifying ops. Phase 3 doesn't
   populate it because the Python side stores the original weights
   internally for revert; the C++ side just sees "applied" vs.
   "not applied". If the user wants to inspect what changed, we'd
   need a new endpoint that returns the delta tensor — defer to a
   Phase 5-ish backend addition.

3. **Steering vector storage** — `SteeringConfig` has `source`,
   `layer`, `alpha`, `cos_sim`. The actual vector lives only on
   the Python side (where it was computed via mean-of-prompts).
   For the C++ side to expose the vector itself (e.g. for "visualise
   the steering vector"), we'd need a new endpoint. Defer.

## Effort estimate

- C++ engine wiring: ~150 LOC
- Python `/steering` endpoint (if needed): ~80 LOC
- Smoke + deep tests: ~250 LOC

Roughly half a session.
