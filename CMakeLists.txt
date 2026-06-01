# llobotomy workspace root.
#
# Thin aggregator: the real targets live in gui_cpp/, which in turn pulls
# in the engine library via a relative add_subdirectory(../llm_engine_cpp).
# Configuring from this directory (cmake -S . -B build) makes that relative
# path resolve from the workspace root, so the two sibling directories
# (gui_cpp/ and llm_engine_cpp/) build together as one tree.
#
# gui_cpp/CMakeLists.txt declares project(llobotomy) and owns the build
# options (LLOB_USE_MOCK_DATA, plus the engine's LLM_ENGINE_BUILD_LLAMA_CPP);
# this file deliberately holds no logic of its own.

cmake_minimum_required(VERSION 3.20)
project(llobotomy_workspace CXX)

add_subdirectory(gui_cpp)

