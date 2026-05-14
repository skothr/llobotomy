# AddLabeledTest.cmake — helper for tier-tagged CTest registration.
#
# Two tiers:
#   smoke — fast, deterministic, no fixtures.  Runs on every commit /
#           CI run.  Each test should finish in well under 1 second.
#   deep  — heavy data / behaviour validation.  May load real model
#           fixtures or run against a live service.  Run on demand via
#           `ctest -L deep`.  Should fail-skip (zero exit) when its
#           required fixture / env var is absent rather than hanging or
#           erroring.
#
# Usage:
#   include(cmake/AddLabeledTest.cmake)
#   add_smoke_test(NAME tensor_handle SOURCES test_tensor_handle.cpp)
#   add_deep_test (NAME gguf_real     SOURCES test_gguf_real.cpp
#                  REQUIRES_ENV LLOB_DEEP_GGUF_PATH)
#
# REQUIRES_ENV is a documentation aid only — the test binary itself is
# responsible for the skip-when-absent behaviour (read getenv, log a
# skip line, exit 0).  Listing it here makes the dependency visible at
# the CMake level so `ctest -L deep` output explains itself.

function(_add_labeled_test_internal label)
  cmake_parse_arguments(PARSE_ARGV 1 ARG
    "" "NAME" "SOURCES;REQUIRES_ENV;LINK_LIBRARIES")

  if(NOT ARG_NAME)
    message(FATAL_ERROR "add_${label}_test: NAME is required")
  endif()
  if(NOT ARG_SOURCES)
    message(FATAL_ERROR "add_${label}_test(${ARG_NAME}): SOURCES is required")
  endif()

  set(target test_${ARG_NAME})
  add_executable(${target} ${ARG_SOURCES})
  target_link_libraries(${target} PRIVATE
    llmengine::llm_engine
    ${ARG_LINK_LIBRARIES})

  add_test(NAME ${ARG_NAME} COMMAND ${target})
  set_tests_properties(${ARG_NAME} PROPERTIES LABELS "${label}")

  if(ARG_REQUIRES_ENV)
    set_tests_properties(${ARG_NAME} PROPERTIES
      ENVIRONMENT_MODIFICATION ""    # placeholder — env vars are
                                     # consumed by the test binary itself
    )
    list(JOIN ARG_REQUIRES_ENV ", " req_list)
    message(STATUS "  test '${ARG_NAME}' (deep): skips unless ${req_list} set")
  endif()
endfunction()

function(add_smoke_test)
  _add_labeled_test_internal(smoke ${ARGN})
endfunction()

function(add_deep_test)
  _add_labeled_test_internal(deep ${ARGN})
endfunction()
