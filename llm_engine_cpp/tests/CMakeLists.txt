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

# ── Wave C — GgufInspectorEngine ──────────────────────────────────────────
# GGUF parser — hand-constructed binary buffers, no real model file required.
add_smoke_test(NAME gguf_parser       SOURCES test_gguf_parser.cpp)

# End-to-end: build GGUF → load → read_slice.
add_smoke_test(NAME gguf_inspector_engine SOURCES test_gguf_inspector_engine.cpp)

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
