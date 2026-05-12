#pragma once
//
// Logger — one entry point for every log line in the app.  Routes to:
//
//   1. AppState.logs — the in-memory ring shown in the logs workspace.
//   2. The on-disk log file (default: ./llobotomy.log, append mode).
//
// All severities flow through here.  No `std::fprintf(stderr, ...)` in
// our own code — anything that wants to surface a message to the user or
// to disk goes through LLOB_LOG_* below.
//
// Thread safety: AppState::pushLog acquires AppState::logs_mu and the
// internal file mutex itself, so calls from any thread are safe.
//
// Macros are printf-style.  `kind` is a short source tag (fwd / probe /
// glfw / font / init / engine / ...) used for the in-app source filter
// and the on-disk `[kind]` column.
//
//   LLOB_LOG_INFO ("init", "llobotomy %s starting", VERSION);
//   LLOB_LOG_WARN ("glfw", "error %d: %s", err, desc);
//   LLOB_LOG_ERROR("engine", "checkpoint load failed: %s", path.c_str());

#include "appstate.hpp"

#include <cstdarg>
#include <cstdio>
#include <string>

namespace llob {

// Initialize the global logger.  `app` is the AppState that owns the
// in-memory ring; `log_path` is where the on-disk log file lives.
// Safe to call multiple times — second call rotates the file.
void LoggerInit(AppState* app, std::string log_path);
void LoggerShutdown();

// Path of the current on-disk log file (empty before LoggerInit).
const std::string& LoggerPath();

// Push a single line.  Both sinks (memory + file) get it.
void LoggerPush(Severity sev, std::string kind, std::string msg);

// printf-style entry — used by the macros below.
void LoggerPushFmt(Severity sev, const char* kind, const char* fmt, ...)
    __attribute__((format(printf, 3, 4)));

}  // namespace llob

#define LLOB_LOG(SEV, KIND, ...) ::llob::LoggerPushFmt((SEV), (KIND), __VA_ARGS__)
#define LLOB_LOG_TRACE(KIND, ...) LLOB_LOG(::llob::Severity::Trace, KIND, __VA_ARGS__)
#define LLOB_LOG_DEBUG(KIND, ...) LLOB_LOG(::llob::Severity::Debug, KIND, __VA_ARGS__)
#define LLOB_LOG_INFO( KIND, ...) LLOB_LOG(::llob::Severity::Info,  KIND, __VA_ARGS__)
#define LLOB_LOG_WARN( KIND, ...) LLOB_LOG(::llob::Severity::Warn,  KIND, __VA_ARGS__)
#define LLOB_LOG_ERROR(KIND, ...) LLOB_LOG(::llob::Severity::Error, KIND, __VA_ARGS__)
#define LLOB_LOG_FATAL(KIND, ...) LLOB_LOG(::llob::Severity::Fatal, KIND, __VA_ARGS__)
