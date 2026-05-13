#pragma once
//
// Log primitives shared between the engine and any UI/host that wants to
// pump engine log lines through its own logger.  The engine produces
// LogEntry values internally and surfaces them via Model::drainEngineLogs;
// the host owns the actual log sinks (stdout, files, on-screen panels).
//
// Kept as a tiny, dependency-free header so consumers can include it
// without pulling in the rest of the model interface.

#include <cstdint>
#include <string>

namespace llmengine {

enum class Severity : int {
    Trace = 0, Debug, Info, Warn, Error, Fatal,
};

// "trace" / "debug" / "info" / "warn" / "error" / "fatal" — lowercase, untruncated.
const char* SeverityName(Severity s);

// "TRACE" / "DEBUG" / "INFO " / "WARN " / "ERROR" / "FATAL" — 5-char padded
// for fixed-width log lines.
const char* SeverityShort(Severity s);

struct LogEntry {
    std::int64_t  ts_ms;     // unix milliseconds
    Severity      sev;
    std::string   kind;      // source tag (e.g. "fwd", "hf", "ablate", "init")
    std::string   msg;
};

}  // namespace llmengine
