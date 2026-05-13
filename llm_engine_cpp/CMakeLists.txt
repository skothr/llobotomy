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
)
add_library(llmengine::llm_engine ALIAS llm_engine)

target_include_directories(llm_engine
  PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
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
