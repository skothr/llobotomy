#include "llm_engine/log.hpp"

namespace llmengine {

const char* SeverityName(Severity s) {
    switch (s) {
        case Severity::Trace: return "trace";
        case Severity::Debug: return "debug";
        case Severity::Info:  return "info";
        case Severity::Warn:  return "warn";
        case Severity::Error: return "error";
        case Severity::Fatal: return "fatal";
    }
    return "?";
}

const char* SeverityShort(Severity s) {
    switch (s) {
        case Severity::Trace: return "TRACE";
        case Severity::Debug: return "DEBUG";
        case Severity::Info:  return "INFO ";
        case Severity::Warn:  return "WARN ";
        case Severity::Error: return "ERROR";
        case Severity::Fatal: return "FATAL";
    }
    return "?";
}

}  // namespace llmengine
