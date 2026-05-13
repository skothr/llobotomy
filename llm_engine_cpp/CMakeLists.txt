cmake_minimum_required(VERSION 3.20)
project(llm_engine CXX)

# llm_engine — a backend-abstraction library for LLM interpretability tooling.
#
# Provides:
#   - llmengine::Model              abstract data interface (every panel of
#                                   an interpretability UI pulls through it)
#   - llmengine::MockModel          deterministic-fake-data implementation
#   - llmengine::HFProxyEngine      thin HTTP/JSON wrapper over a Python
#                                   FastAPI backend (see testing/gui/backend/)
#   - llmengine::Severity           log primitives so engines can surface
#   - llmengine::LogEntry           messages without owning a sink
#
# Independent of any UI framework — consumers wire `drainEngineLogs()` into
# whatever logging system they use.  The reference consumer is
# `testing/gui_cpp/` (Dear-ImGui-based interpretability bench).
#
# Build options:
#   LLM_ENGINE_USE_MOCK_DATA  ON   MockModel returns deterministic fake data
#                                   (great for screenshots / dev iteration);
#                                   OFF turns every MockModel method into a
#                                   no-data sentinel return so a release
#                                   build can't silently ship mock numbers.

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 23)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

option(LLM_ENGINE_USE_MOCK_DATA
       "Have MockModel return deterministic mock data (vs. no-data sentinels)"
       ON)

# ---------------------------------------------------------------------------
# Vendored header-only deps — private to the engine's implementation.
# Both are MIT-licensed; see libs/<name>/LICENSE*.
# Marked SYSTEM so their warnings don't leak into consumers.
# ---------------------------------------------------------------------------
add_library(llm_engine_cpp_httplib INTERFACE)
target_include_directories(llm_engine_cpp_httplib SYSTEM INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/libs/cpp-httplib)

add_library(llm_engine_nlohmann_json INTERFACE)
target_include_directories(llm_engine_nlohmann_json SYSTEM INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/libs/nlohmann_json)

# ---------------------------------------------------------------------------
# llm_engine — the library itself.
# ---------------------------------------------------------------------------
add_library(llm_engine STATIC
  src/log.cpp
  src/mock_model.cpp
  src/hf_proxy_engine.cpp
  src/tensor_source.cpp
  src/tensor_handle.cpp
  src/derived_cache.cpp
  src/model_view.cpp
  src/gguf_parser.cpp
  src/gguf_source.cpp
  src/gguf_inspector_engine.cpp
  # LlamaCppEngine: always compiled; llama.cpp-specific code is guarded by
  # LLM_ENGINE_HAVE_LLAMA_CPP so the stub runs cleanly with the flag off.
  src/llama_cpp_engine.cpp
  src/llama_cpp_capture.cpp
)
add_library(llmengine::llm_engine ALIAS llm_engine)

target_include_directories(llm_engine
  PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  # src/ is PRIVATE so llama_cpp_internal.hpp is reachable by our own TUs
  # but not exported to consumers.
  PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)

# httplib + json are PRIVATE — they're implementation details.  Consumers
# never see them.
target_link_libraries(llm_engine
  PRIVATE llm_engine_cpp_httplib llm_engine_nlohmann_json
)

if(LLM_ENGINE_USE_MOCK_DATA)
  target_compile_definitions(llm_engine PRIVATE LLOB_USE_MOCK_DATA=1)
endif()

# Threads are needed by the mutex/cv work in HFProxyEngine.
find_package(Threads REQUIRED)
target_link_libraries(llm_engine PUBLIC Threads::Threads)

# Strict warnings on our own translation units — vendored deps stay quiet
# thanks to the SYSTEM keyword on their interface includes.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(llm_engine PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---------------------------------------------------------------------------
# Optional: LlamaCppEngine — links against a pre-built libllama.so.
#
# Enable with:
#   cmake -DLLM_ENGINE_BUILD_LLAMA_CPP=ON
#       [-DLLAMA_CPP_ROOT=/path/to/llama.cpp]
#
# The option is OFF by default; consumers that don't need the native runtime
# get a clean, dependency-free build.
# ---------------------------------------------------------------------------
option(LLM_ENGINE_BUILD_LLAMA_CPP
       "Build LlamaCppEngine (links pre-built libllama.so)"
       OFF)

if(LLM_ENGINE_BUILD_LLAMA_CPP)
  set(LLAMA_CPP_ROOT
      "/home/ai/ai-projects/llm/lib/llama.cpp"
      CACHE PATH "Root of the pre-built llama.cpp installation")

  set(_LLAMA_LIB_DIR "${LLAMA_CPP_ROOT}/build/bin")
  set(_LLAMA_INC_DIR "${LLAMA_CPP_ROOT}/include")
  set(_GGML_INC_DIR  "${LLAMA_CPP_ROOT}/ggml/include")

  # ── Imported targets for each .so ─────────────────────────────────────
  foreach(_lib IN ITEMS llama ggml ggml-base ggml-cpu)
    set(_tgt "llm_engine_${_lib}_imported")
    add_library(${_tgt} SHARED IMPORTED)
    set_target_properties(${_tgt} PROPERTIES
      IMPORTED_LOCATION "${_LLAMA_LIB_DIR}/lib${_lib}.so"
    )
  endforeach()

  # ── ggml-cuda is optional (not present on CPU-only builds) ────────────
  if(EXISTS "${_LLAMA_LIB_DIR}/libggml-cuda.so")
    add_library(llm_engine_ggml_cuda_imported SHARED IMPORTED)
    set_target_properties(llm_engine_ggml_cuda_imported PROPERTIES
      IMPORTED_LOCATION "${_LLAMA_LIB_DIR}/libggml-cuda.so"
    )
    set(_GGML_CUDA_TARGET llm_engine_ggml_cuda_imported)
  else()
    set(_GGML_CUDA_TARGET "")
  endif()

  # ── Compile definition so guarded code compiles in ────────────────────
  # (sources are always compiled; the guard macro enables the llama.cpp paths)
  target_compile_definitions(llm_engine PRIVATE LLM_ENGINE_HAVE_LLAMA_CPP=1)

  # ── Include dirs: SYSTEM so llama.cpp warnings don't pollute ──────────
  target_include_directories(llm_engine SYSTEM PRIVATE
    "${_LLAMA_INC_DIR}"
    "${_GGML_INC_DIR}"
  )

  # ── Link ──────────────────────────────────────────────────────────────
  target_link_libraries(llm_engine PRIVATE
    llm_engine_llama_imported
    llm_engine_ggml_imported
    llm_engine_ggml-base_imported
    llm_engine_ggml-cpu_imported
    ${_GGML_CUDA_TARGET}
  )

  message(STATUS "LLM_ENGINE_BUILD_LLAMA_CPP=ON: LlamaCppEngine enabled")
  message(STATUS "  LLAMA_CPP_ROOT = ${LLAMA_CPP_ROOT}")
endif()  # LLM_ENGINE_BUILD_LLAMA_CPP

# ---------------------------------------------------------------------------
# Tests — only built when this CMakeLists is the top-level project (i.e.
# someone building llm_engine directly, not consumers add_subdirectory'ing).
# Keeps gui_cpp's build clean of test binaries.
# ---------------------------------------------------------------------------
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  option(LLM_ENGINE_BUILD_TESTS "Build llm_engine unit tests" ON)
  if(LLM_ENGINE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
  endif()
endif()
