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
