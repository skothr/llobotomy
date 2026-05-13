# Tests for the llm_engine library.
#
# Self-contained — no external test framework.  Each test is a small
# program with assert()s; CTest runs them and fails the build on any
# non-zero exit.  Once the test surface grows past ~5 files we can
# switch to GoogleTest via FetchContent (the plan mentions this — but
# the first PR keeps the dependency surface flat).

add_executable(test_tensor_handle test_tensor_handle.cpp)
target_link_libraries(test_tensor_handle PRIVATE llmengine::llm_engine)
add_test(NAME tensor_handle COMMAND test_tensor_handle)

add_executable(test_derived_cache test_derived_cache.cpp)
target_link_libraries(test_derived_cache PRIVATE llmengine::llm_engine)
add_test(NAME derived_cache COMMAND test_derived_cache)

add_executable(test_model_view test_model_view.cpp)
target_link_libraries(test_model_view PRIVATE llmengine::llm_engine)
add_test(NAME model_view COMMAND test_model_view)

# MockModel surface compat — sanity-check that the new ModelView path
# coexists with the existing per-DTO getters.
add_executable(test_mock_model_compat test_mock_model_compat.cpp)
target_link_libraries(test_mock_model_compat PRIVATE llmengine::llm_engine)
add_test(NAME mock_model_compat COMMAND test_mock_model_compat)

# GGUF parser — hand-constructed binary buffers, no real model file required.
add_executable(test_gguf_parser test_gguf_parser.cpp)
target_link_libraries(test_gguf_parser PRIVATE llmengine::llm_engine)
add_test(NAME gguf_parser COMMAND test_gguf_parser)

# GgufInspectorEngine — end-to-end: build GGUF → load → read_slice.
add_executable(test_gguf_inspector_engine test_gguf_inspector_engine.cpp)
target_link_libraries(test_gguf_inspector_engine PRIVATE llmengine::llm_engine)
add_test(NAME gguf_inspector_engine COMMAND test_gguf_inspector_engine)

# HFProxyEngine Phase 2 — smoke: construct with dead URL, assert all getters
# return safe sentinels and don't crash.  Integration tier (setActivePrompt
# round-trip against a live FastAPI + TinyLlama) requires
# -DLLOB_INTEGRATION_TEST_ENABLED and LLOB_INTEGRATION_TEST=1 at runtime.
add_executable(test_hf_proxy_capture test_hf_proxy_capture.cpp)
target_link_libraries(test_hf_proxy_capture PRIVATE llmengine::llm_engine)
add_test(NAME hf_proxy_capture COMMAND test_hf_proxy_capture)

# Integration build variant: same source, compiled with the integration flag.
add_executable(test_hf_proxy_capture_integration test_hf_proxy_capture.cpp)
target_compile_definitions(test_hf_proxy_capture_integration PRIVATE
    LLOB_INTEGRATION_TEST_ENABLED=1)
target_link_libraries(test_hf_proxy_capture_integration PRIVATE llmengine::llm_engine)
# Not added to ctest by default — only run manually when a live backend is up.
