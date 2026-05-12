# MODEL_HOOKS — every data hook the llobotomy UI needs

Every panel in every workspace pulls its data through the `Model` interface
defined in [`src/model/model.hpp`](../src/model/model.hpp). This file is the
flat inventory: every method, what it returns, where it's called, and what
real backend (HuggingFace bridge / native llama.cpp / libtorch / FastAPI
proxy) should provide.

The default implementation is `MockModel`, which has two modes:

- **`-DLLOB_USE_MOCK_DATA=ON`** (CMake default) — returns deterministic fake
  data so the UI looks alive in dev / for screenshots.
- **`-DLLOB_USE_MOCK_DATA=OFF`** (release / real-backend builds) — every
  method returns the no-data sentinel for its type (empty vector / NaN
  / `""`); the UI shows `// no … available` placeholders.

When wiring a real backend, write a class deriving from `llob::Model` that
implements each method. Build with `LLOB_USE_MOCK_DATA=OFF` and swap the
`MockModel` instance in `main.cpp` for your subclass.

## Sentinels

| Type | "No data" value | Header constant |
|---|---|---|
| `float` | NaN | `kNoFloat` |
| `int` | -1 | `kNoInt` |
| `int64_t` | -1 | `kNoSize` |
| `std::string` | `""` | — |
| `std::vector<T>` | `{}` | — |

UI code uses `FmtFloat / FmtInt / FmtSize / FmtTokens / Or(...)` from
[`src/ui/fmt.hpp`](../src/ui/fmt.hpp) so every sentinel renders as `—`.

## Hook table

### Architecture / weights

| Method | Returns | UI sites | Real backend should… |
|---|---|---|---|
| `getParamBreakdown(layer)` | `vector<ParamBreakdownRow>` | `architecture.cpp` Inspector → "Parameter breakdown" | sum parameter counts for each component group of the named layer |
| `getLiveActivations(layer)` | `LiveActivations` | `architecture.cpp` Inspector → "Live activations" | capture forward-hook outputs (norms, attn entropy, dead-neuron count) for the active token |
| `getStateDict()` | `vector<TensorMeta>` | `raw_tensors.cpp` state_dict, `inference.cpp` raw pane | enumerate every tensor in the loaded checkpoint with name+dtype+shape+size |
| `getTensorMeta(name)` | `TensorMeta` | `raw_tensors.cpp` tensor_view + tensor_stats | resolve a single name (typically from a click) without re-listing everything |

### Activations / attention (per-token)

| Method | Returns | UI sites | Real backend should… |
|---|---|---|---|
| `getAttentionPattern(layer, head, seqLen, bias)` | `vector<vector<float>>` (causal NxN) | `attention.cpp` head_browser thumbs + main heatmap | run forward, intercept the attention softmax for that head, return the matrix |
| `getActivation(layer, kind, n)` | `vector<float>` | `attention.cpp` Q/K/V slices, `inference.cpp` raw pane | dispatch on `kind` ∈ {0=q, 1=k, 2=v, default=resid_post}, return the d_head / d_model slice |
| `getHeadNorm(layer, head)` | `float` ∈ [0,1] | arch map head tinting + per-head bars in inference probe panel | per-head ‖attention output‖ normalised against its corpus distribution |
| `getComponentNorm(layer, comp)` | `float` ∈ [0,1] | arch map component tinting (currently unused; reserved) | per-component activation norm normalised |

### Inference workspace

| Method | Returns | UI sites | Real backend should… |
|---|---|---|---|
| `getResidualContribution(layer)` | `{attn, mlp}` | residual_flow per-layer bars | attn_out_norm and mlp_out_norm for the current token's forward pass |
| `getResidualSummary(layer)` | `ResidualSummary` | residual_flow KV footer + L*_probe Layer summary | aggregated norms + cos(prev) + kurtosis + effective rank |
| `getLogitLensTrajectory(token, kLayers)` | `vector<LogitLensRow>` | forward_pass logit-lens table | for each layer, project the residual through `unembed` and report top-1 / top-2 / entropy. `is_resolved=true` marks the layer where the eventual top-1 first becomes top-1 |
| `getOutputLogits(k)` | `vector<LogitDist>` | forward_pass output bars | top-k tokens at the final layer; `delta` = current vs base run (zero when no baseline cached) |
| `getMlpFeatures(layer, k)` | `vector<MlpFeatureActivation>` | L*_probe → "MLP feature activations" | top-k post-SiLU MLP feature indices + signed activations |
| `getTokenLossPerToken(layer)` | `vector<float>` | token_strip → "cross-entropy loss per token" | per-token CE loss at the named layer (lm_head intermediate) |
| `getSurprisalDelta()` | `vector<float>` | token_strip → "surprisal Δ vs base" | requires a baseline (un-ablated) cached run; returns base − current per token |
| `getActiveProbes()` | `vector<ProbeEntry>` | probe_controls → "Active probes" | currently-attached probe heads (NOT the user's pending intent — that's `AppState.probedHeads` etc.) |
| `getSteering()` | `SteeringConfig` | probe_controls → "Steering vector" | active steering vector source / layer / α / cos-sim |

### Attention workspace

| Method | Returns | UI sites | Real backend should… |
|---|---|---|---|
| `getQKVStats(layer, head, token)` | `QKVStats` | qkv_inspector top KV | norms of Q, K, V for the selected token + selected-token attention fractions to {bos, self, prev} |
| `getHeadStats(layer, head)` | `vector<HeadStatRow>` | head_stats → "Pattern statistics" | pre-computed per-head pattern stats (entropy, diagonal weight, prev-token bias, BOS attention, induction score, …) — engine should cache these per-head |
| `getPatchSource()` | `PatchSourceState` | ablation_queue → "Patch source" | activation-patching source description: prompt, target pos, component, observed effect (nats) |
| `getHeadBias(layer, head)` | `HeadBias` enum | head_browser thumb labels + main heatmap title | classifier output: which behavioural pattern best describes this head (diag / prev / first / broad / induction). Engine source: rules + a tiny classifier on the attention matrix |

### Probes / SAE workspace

| Method | Returns | UI sites | Real backend should… |
|---|---|---|---|
| `getFeatureLibrary(filter)` | `vector<FeatureSummary>` | feat-browser table | enumerate features in the active SAE/probe set; substring-match `label` and `id` against `filter` |
| `getFeatureCard(featureId)` | `FeatureCard` | feat-detail Feature card | layer location, type, sparsity, fires/M tok, mean act, decoder ↑/↓ logit summaries |
| `getFeatureExamples(featureId, k)` | `vector<FeatureExample>` | feat-detail Top activating examples | top-k corpus contexts where the feature fires hardest, split into (pre, highlight, post) for inline emphasis |
| `getCoFiringFeatures(featureId, threshold)` | `vector<CoFiringEntry>` | feat-detail Co-firing features | features whose firing pattern is most correlated with this one (cosine of activation vectors over the corpus) |
| `getSAETrainingMetrics(saeId)` | `SAETrainingMetrics` | sae_training panel | live SAE-training counters + history vectors for sparklines |
| `getProbeLibrary()` | `vector<ProbeEntry>` | probe_ops Probe library | saved probes available to attach (reads `./out/probes/*.pt + meta`) |
| `getProbeTrainState(name)` | `ProbeTrainState` | probe_ops Train new probe + Validation curves | live training-state for the named probe (training flag, step, accs, val curve) |
| `getRecentExports()` | `vector<ExportEntry>` | export panel Recent exports | recent files in `./out/`, sorted by mtime |
| `startProbeTraining(name, kind, location, dataset)` | void | probe_ops "▶ train probe" button | kick off a new probe-training job |
| `saveProbe(name)` | void | probe_ops "save" button | persist the named probe to disk |
| `exportSnapshot(path)` | void | export panel "↗ export state" button | serialise the full session (ablations, probes, features, steering) to a JSON sidecar |

### Training workspace

| Method | Returns | UI sites | Real backend should… |
|---|---|---|---|
| `getTrainingState()` | `TrainingState` | run_summary title + control button labels | current step / total + running flag |
| `getTrainingMetrics()` | `vector<TrainingMetricCard>` | run_summary big metric cards | LOSS / VAL LOSS / LR / TOK_PER_SEC / GPU_UTIL / GRAD_NORM / EPOCH formatted for display |
| `getTrainingLoss(maxSteps)` | `LossCurve` | loss_plot Sparkline | last `maxSteps` train + val loss values |
| `getGradFlowPerLayer()` | `vector<float>` | grad_flow per-layer bars | per-layer gradient norm at the most recent backward pass |
| `getPerLayerLoss(maxSteps)` | `vector<vector<float>>` | layerwise_loss sparkline grid | 2D `[layer][step]` matrix of per-layer loss values |
| `pauseTraining / resumeTraining / stepTraining / resetTraining / stopTraining` | void | control panel buttons | gate the training loop's tick |

### Finetune workspace

| Method | Returns | UI sites | Real backend should… |
|---|---|---|---|
| `getLoRAConfig()` | `LoRAConfig` | lora_config LoRA · adapter | rank/α/dropout/target/trainable params/method (PEFT-style) |
| `getOptimizerConfig()` | `OptimizerConfig` | lora_config Optimizer | name/lr/betas/warmup/schedule/weight_decay |
| `getDataConfig()` | `DataConfig` | lora_config Data | dataset/n_train/n_eval/ctx_len/packing |
| `getEvalDiff(benchmark)` | `EvalDiffMetric` | eval_diff base vs ft | base/ft/Δ for the named benchmark (lm-eval-harness etc.) |
| `getEvalLossCurve()` | `vector<float>` | eval_diff loss curves | finetune loss progression (downsampled for sparkline) |
| `getABSample()` | `ABSample` | eval_diff A/B sample | one example: prompt + base response + ft response |
| `getDeltaWHeatmap(numLayers, numComponents)` | `vector<vector<float>>` | diff_run heatmap | `‖ΔW‖` per-layer × per-component (LoRA-merged minus base) |
| `getDeltaWComponentNames()` | `vector<string>` | diff_run X-axis labels | column headers ("Q", "K", …, "rn1", "rn2", "b1", "b2") |
| (TBD) `setLayerFrozen(layer, bool)` | void | layer_freeze tile clicks | currently UI-only state; wire when finetune control is real |

### Datasets workspace

| Method | Returns | UI sites | Real backend should… |
|---|---|---|---|
| `getDatasets()` | `vector<DatasetSummary>` | datasets list | enumerate available datasets with size + token count |
| `getSample(dataset, sampleId)` | `DatasetSample` | sample_browser text panel | text + highlight span list for the sample at this position; spans mark interesting regions for inline emphasis |
| `getSampleStats(dataset, sampleId)` | `DatasetSampleStats` | sample_browser Token statistics | len / ppl(base) / ppl(ft) / avg surprisal / top fired feature |
| `getDatasetDistribution(dataset)` | `DatasetDistribution` | dataset_stats distribution + source mix | document length histogram + source-mix bars |
| `getTokenIds(dataset, sampleId, n)` | `vector<float>` | token_ids hex pane (returned as fp for shared HexView path) | first `n` token ids of the sample; UI casts in display |

### Raw tensors workspace

| Method | Returns | UI sites | Real backend should… |
|---|---|---|---|
| `getWeightSlice(name, offset, n)` | `vector<float>` | raw_tensors Hex view + arch param_hex | n raw values from the named tensor starting at `offset` (mmap if possible) |
| `getWeightHistogram(name, bins)` | `vector<int>` | arch param_distribution + inference resid_post + probes feature card | bin counts for the value distribution of the named tensor |
| `getTensorStats(name)` | `TensorStats` | raw_tensors tensor_stats Norms + slice footer | min/max/mean/std/Frobenius/op-norm/inf-norm/effective-rank (cached during checkpoint indexing) |
| `getSingularValues(name, k)` | `vector<float>` | tensor_stats Singular values | top-k singular values (cached, not computed per-frame) |
| `getTensorSlice2D(name, axis0, axis1, rows, cols)` | `vector<vector<float>>` | raw_tensors 2D slice | 2D window into a higher-dim tensor (engine paging) |
| `getDiffSlice2D(name, rows, cols)` | `vector<vector<float>>` | raw_tensors diff_view heatmap | 2D delta between current and base checkpoint |
| `getDiffStats(name)` | `DiffStats` | raw_tensors diff_view metrics | Frobenius norm of delta + cosine + top-k delta singular values |

### Engine / runtime

| Method | Returns | UI sites | Real backend should… |
|---|---|---|---|
| `getEngineMetrics()` | `EngineMetrics` | logs metrics panel + (future) menubar status block | warn/err rates, cuda mem used/total, cpu %, fwd time, fps, device + dtype tags |

## Hooks intentionally left unwired

A handful of fields render `—` because there is no `Model::*` method for
them yet — they're either UI-only state today or were judged not worth a
hook until the real backend is in scope. They're tagged `// [DATA HOOK]`
inline at the call site so they're easy to find:

- **Architecture inspector → act / norm / rope θ rows** — config strings on
  ModelInfo. Add to `ModelInfo` when a real loader exists.
- **Training control panel → ckpt every / eval every / hf-sync / wandb** —
  add a `TrainingScheduleConfig` struct + getter.
- **Finetune layer_freeze → Δ params / GPU mem est.** — derive from
  `LoRAConfig.trainable_M` and `EngineMetrics`; not yet wired.
- **Probes export → active features count** — engine-side probe registry
  filter; placeholder `—` today.

## Conventions

1. **Per-frame cost matters.** Every `Model::*` method may be called once
   per frame on every visible workspace's draw. Cache aggressively in the
   backend; the UI doesn't memoise on its side.
2. **Methods never throw on missing data.** They return the no-data
   sentinel and the UI prints a placeholder. `noexcept(false)` is reserved
   for genuine IO failure (tensor reads from disk).
3. **Mutators (`startProbeTraining`, `pauseTraining`, …) default to no-op
   on the base interface.** A backend that doesn't implement them is not
   broken; the UI button just won't do anything when clicked.
4. **Names in `Model::getX(name)` are tensor / dataset / probe identifiers
   in the engine's own naming scheme.** UI code synthesises them from
   workspace state (e.g. `"blocks.<L>.attn.W_Q.weight"`); a real backend
   that uses different names should expose its naming convention via
   `getStateDict()` and let the UI follow whatever it returns.

## Adding a new hook

1. Add the struct type (if any) and the virtual method to `Model` in
   `src/model/model.hpp`.
2. Add the matching out-of-line declaration to `MockModel` in the same
   file.
3. Implement it in `src/model/mock_model.cpp` wrapped in
   `#if LLOB_USE_MOCK_DATA / #else return Empty; / #endif`.
4. Call it from the workspace, with a `// [DATA HOOK]` comment naming
   what the real backend should provide.
5. Add a row to the table above.
