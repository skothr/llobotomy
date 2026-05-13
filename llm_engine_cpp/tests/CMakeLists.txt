# Tests for the llm_engine library.
#
# Two tiers (per cmake/AddLabeledTest.cmake):
#
#   smoke — fast (< 100ms each), deterministic, hand-constructed
#           fixtures, no live services.  Runs on every commit.
#   deep  — heavy data / live-service / real-model validation.  Each
#           one self-skips with exit 0 when its required env var or
#           fixture isn't present.  Run via `ctest -L deep`.
#
# Helper macros add_smoke_test() / add_deep_test() defined in
# ../cmake/AddLabeledTest.cmake.
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/AddLabeledTest.cmake)

# ── Substrate (the ModelView / TensorHandle / DerivedCache primitives) ────
add_smoke_test(NAME tensor_handle     SOURCES test_tensor_handle.cpp)
add_smoke_test(NAME derived_cache     SOURCES test_derived_cache.cpp)
add_smoke_test(NAME model_view        SOURCES test_model_view.cpp)

# MockModel surface compat — sanity-check that the new ModelView path
# coexists with the existing per-DTO getters.
add_smoke_test(NAME mock_model_compat SOURCES test_mock_model_compat.cpp)

# Substrate contract — invariants every backend must uphold (capability
# mirror, clear() resets everything, atomic current safety, tokenizer
# slot consistency, intervention canonical roundtrip).
add_smoke_test(NAME substrate_contract SOURCES test_substrate_contract.cpp)

# MockModel data-shape sweep — every per-DTO getter returns sensibly-
# shaped data under MOCK_DATA=ON and the no-data sentinel under =OFF.
# Catches getter regressions and false-data leaks in release builds.
add_smoke_test(NAME mock_data_shapes SOURCES test_mock_data_shapes.cpp)

# Threading stress — 4 readers + 1 writer on ModelView::current for
# 5000 cycles; verifies no torn reads, ref-count safety under
# concurrent atomic shared_ptr load/store.  Pure substrate test.
add_smoke_test(NAME threading_stress SOURCES test_threading_stress.cpp)

# LoadOptions roundtrip — verifies the "ignore unknown extras"
# forward-compat contract every backend must uphold.
add_smoke_test(NAME load_options SOURCES test_load_options.cpp)

# DerivedCache threading — same-key contention (compute-at-most-once),
# disjoint-key parallelism (exactly-once each), concurrent invalidate.
add_smoke_test(NAME derived_cache_threading SOURCES test_derived_cache_threading.cpp)

# Exhaustive path-API coverage — every documented ModelView::get(path)
# resolves to the right variant arm; malformed paths return monostate
# without crashing.
add_smoke_test(NAME path_api SOURCES test_path_api.cpp)

# ── Wave C — GgufInspectorEngine ──────────────────────────────────────────
# GGUF parser — hand-constructed binary buffers, no real model file required.
add_smoke_test(NAME gguf_parser       SOURCES test_gguf_parser.cpp)

# End-to-end: build GGUF → load → read_slice.
add_smoke_test(NAME gguf_inspector_engine SOURCES test_gguf_inspector_engine.cpp)

# Deep tier: load a REAL GGUF file (converted from HF via the llama.cpp
# converter, or any user-supplied path).  Self-skips when
# LLOB_DEEP_GGUF_PATH is unset.
add_deep_test(NAME real_gguf
              SOURCES test_real_gguf.cpp
              REQUIRES_ENV LLOB_DEEP_GGUF_PATH)

# ── Wave A — HFProxyEngine Phase 2 ────────────────────────────────────────
# Smoke: construct with dead URL, assert all getters return safe sentinels
# and don't crash.
add_smoke_test(NAME hf_proxy_capture  SOURCES test_hf_proxy_capture.cpp)

# Deep / integration tier: same source compiled with the integration flag.
# Run requires a live FastAPI backend + TinyLlama loaded.  The test self-
# skips (exit 0) when LLOB_INTEGRATION_TEST=1 isn't set at runtime.
add_executable(test_hf_proxy_capture_integration test_hf_proxy_capture.cpp)
target_compile_definitions(test_hf_proxy_capture_integration PRIVATE
    LLOB_INTEGRATION_TEST_ENABLED=1)
target_link_libraries(test_hf_proxy_capture_integration PRIVATE llmengine::llm_engine)
add_test(NAME hf_proxy_capture_integration COMMAND test_hf_proxy_capture_integration)
set_tests_properties(hf_proxy_capture_integration PROPERTIES LABELS "deep")

# ── Wave A — HFProxyEngine Phase 3 (intervention wiring) ──────────────────
# Smoke: dead-URL engine, setAblation/setSteering/clearSteering no-crash;
# m_view.surgery stays empty (no commit possible against dead URL); invalid
# canonical names are skipped gracefully.
add_smoke_test(NAME hf_proxy_intervention SOURCES test_hf_proxy_intervention.cpp)

# Deep / integration tier: live backend + TinyLlama.  Loads baseline capture,
# ablates head (5,3), re-captures, verifies pattern changed.  Self-skips when
# LLOB_INTEGRATION_TEST=1 isn't set at runtime.
add_executable(test_hf_proxy_intervention_integration test_hf_proxy_intervention.cpp)
target_compile_definitions(test_hf_proxy_intervention_integration PRIVATE
    LLOB_INTEGRATION_TEST_ENABLED=1)
target_link_libraries(test_hf_proxy_intervention_integration PRIVATE llmengine::llm_engine)
add_test(NAME hf_proxy_intervention_integration COMMAND test_hf_proxy_intervention_integration)
set_tests_properties(hf_proxy_intervention_integration PROPERTIES LABELS "deep")

# ── Wave A — HFProxyEngine Phase 4 (WebSocket streams) ───────────────────
# Smoke: dead-URL engine, getLogitLensTrajectory + getOutputLogits return {}
# without crashing; capabilities.has_logit_lens / has_token_stream stay false
# when load fails.
add_smoke_test(NAME hf_proxy_ws SOURCES test_hf_proxy_ws.cpp)

# Deep / integration tier: live backend + TinyLlama.  Asserts logit-lens rows
# are non-empty and getOutputLogits returns top-k with probabilities in [0,1].
# Self-skips when LLOB_INTEGRATION_TEST=1 isn't set at runtime.
add_executable(test_hf_proxy_ws_integration test_hf_proxy_ws.cpp)
target_compile_definitions(test_hf_proxy_ws_integration PRIVATE
    LLOB_INTEGRATION_TEST_ENABLED=1)
target_link_libraries(test_hf_proxy_ws_integration PRIVATE llmengine::llm_engine)
add_test(NAME hf_proxy_ws_integration COMMAND test_hf_proxy_ws_integration)
set_tests_properties(hf_proxy_ws_integration PROPERTIES LABELS "deep")

# ── Wave D — LlamaCppEngine ───────────────────────────────────────────────
# Smoke tier: no real model needed; tests construction, bad-path failure,
# post-unload sentinels.  Compiles and runs with LLM_ENGINE_BUILD_LLAMA_CPP
# ON or OFF (graceful-failure path tested in the OFF case).
add_smoke_test(NAME llama_cpp_smoke SOURCES test_llama_cpp_smoke.cpp)

# Deep tier: requires LLOB_DEEP_LLAMA_GGUF_PATH env var pointing at a GGUF.
# The test binary self-skips (exit 0) when the var is absent.
add_deep_test(NAME llama_cpp_real
              SOURCES test_llama_cpp_real.cpp
              REQUIRES_ENV LLOB_DEEP_LLAMA_GGUF_PATH)

# Propagate the llama.cpp availability flag to the test binaries so their
# informational messages are accurate.
if(LLM_ENGINE_BUILD_LLAMA_CPP)
  target_compile_definitions(test_llama_cpp_smoke PRIVATE LLM_ENGINE_HAVE_LLAMA_CPP=1)
  target_compile_definitions(test_llama_cpp_real  PRIVATE LLM_ENGINE_HAVE_LLAMA_CPP=1)
endif()
