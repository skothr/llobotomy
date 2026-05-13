# HFProxy Phase 4 — WebSocket streams

> Status: not started. Follows Phase 3 (intervention). The two remaining
> WS-driven flows: logit lens trajectory + live token generation.

## Goal

Wire `Model::getLogitLensTrajectory(token, kLayers)` and
`Model::getOutputLogits(k)` against the existing FastAPI WS handlers
at `/ws/sessions/{n}/logit-lens` and `/ws/sessions/{n}/generate`.

Unlike Phase 2's REST capture flow, these are *streams* — the server
emits a frame per layer (logit-lens) or per generated token
(generate), and the UI shows them as they arrive. The engine
maintains a long-lived WS connection per stream during its lifetime.

## C++ side architecture

Add a thin WS helper class so the two stream flows share the
connection-lifecycle code:

```cpp
class WsSession {
public:
    using OnFrame = std::function<void(const nlohmann::json&)>;
    using OnClose = std::function<void(int code, const std::string& reason)>;

    WsSession(std::string base_url, std::string ws_path,
              OnFrame on_frame, OnClose on_close);
    ~WsSession();

    bool send_json(const nlohmann::json& msg);   // request next stream / send config
    bool is_open() const;
    // ...
};
```

Implementation uses cpp-httplib's WebSocket support (or, if that's
not present, a small ad-hoc WS frame parser — RFC 6455 is ~200 LOC).

## Per-flow wiring

### Logit lens (`getLogitLensTrajectory`)

UI behaviour: when the inference workspace's "logit lens" panel is
visible, the engine should be streaming a fresh trajectory for the
active token. The trajectory is per-layer, so a single WS exchange
yields `n_layers` frames.

```cpp
struct LogitLensWorker {
    // Runs on the engine thread. Opens a WS to /ws/.../logit-lens,
    // sends {token: int, k: int}, receives one frame per layer:
    //   {layer: int, top1: str, p1: float, top2: str, p2: float,
    //    entropy: float, is_resolved: bool}
    // Accumulates into the active CaptureBundle's
    //   `derived` (per-token logit_lens vector<LogitLensRow>)
    // On final layer, atomic-store the bundle.
};
```

Trigger: UI changes which token it's inspecting → engine calls
`setActiveLogitLensToken(int token)` (new mutator, default no-op).
Backend cancels any in-flight stream, opens a new one.

### Generate (`getOutputLogits`)

UI behaviour: when the user clicks "generate", the engine streams
tokens until EOS or max-tokens. Each frame yields:

```json
{"step": int, "token_id": int, "token_str": str,
 "top_probs": [{"token": str, "prob": float}, ...]}
```

The substrate's `CaptureBundle.token_ids` / `token_strs` extends with
each frame. `getOutputLogits(k)` returns the top-k from the *last*
frame.

Wire `setActivePrompt` to optionally also kick off generation (if a
new mutator `setGenerateConfig` is set). Or — simpler — add an
explicit `generate(max_tokens, temperature, ...)` mutator that's
separate from prompt-changes.

## Capability bits

After Phase 4 lands, `getCapabilities().has_logit_lens = true` and
`has_token_stream = true`.

## Threading

A second worker thread for WS handling (the existing SamplerWorker
handles heartbeat + capture REST flows). Reason: WS read is blocking,
and we don't want it serialised behind a heartbeat HTTP roundtrip.

```
                  ┌──────────────┐
                  │  UI thread   │ — reads view().current.load()
                  └──────┬───────┘
                         │ setActivePrompt / setActiveLogitLensToken
                         ▼
                  ┌──────────────┐
                  │ SamplerWorker│ — heartbeat + REST capture (Phase 2)
                  └──────────────┘
                  ┌──────────────┐
                  │ WsWorker     │ — long-lived WS conns for logit-lens
                  │              │   + generate; per-frame callback into
                  │              │   view.current bundle
                  └──────────────┘
```

WsWorker is created lazily on first `setActivePrompt` (since both
streams need a session loaded first), torn down at unloadCheckpoint.

## Tests

**Smoke** in `tests/test_hf_proxy_ws.cpp`: dead-URL engine,
`setActivePrompt(...)` → WsWorker creation doesn't crash, immediate
WS-connect-failure logs but is non-fatal; `getLogitLensTrajectory`
and `getOutputLogits` return empty.

**Deep** in same file, gated by `LLOB_INTEGRATION_TEST=1`: live
backend, TinyLlama, `setActivePrompt("The capital of France is")`,
wait for capture + first generate token, verify `getOutputLogits(5)`
top-1 contains "Paris" or close cognate.

## Effort estimate

- WsSession helper: ~250 LOC (incl. ad-hoc WS frame parser if needed)
- LogitLensWorker + Generate wiring: ~250 LOC
- Smoke + deep tests: ~200 LOC

Roughly one focused session, contingent on cpp-httplib's WebSocket
support being sufficient (it ships with a `WebSocketClient` class —
verify before writing the ad-hoc parser).
