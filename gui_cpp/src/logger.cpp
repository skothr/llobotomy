#include "logger.hpp"

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <mutex>
#include <string>

namespace llob {

namespace {

std::mutex   g_mu;
AppState*    g_app   = nullptr;
std::FILE*   g_file  = nullptr;
std::string  g_path;

std::string fmt_timestamp_iso(std::int64_t ts_ms) {
    using namespace std::chrono;
    const auto secs = ts_ms / 1000;
    const std::time_t t = static_cast<std::time_t>(secs);
    std::tm tm{};
    localtime_r(&t, &tm);
    char buf[40];
    std::snprintf(buf, sizeof buf, "%04d-%02d-%02dT%02d:%02d:%02d.%03lld",
                  tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                  tm.tm_hour, tm.tm_min, tm.tm_sec,
                  static_cast<long long>(ts_ms % 1000));
    return buf;
}

}  // namespace

void LoggerInit(AppState* app, std::string log_path) {
    std::lock_guard<std::mutex> lk(g_mu);
    g_app = app;
    if (g_file) { std::fclose(g_file); g_file = nullptr; }
    g_path = std::move(log_path);
    g_file = std::fopen(g_path.c_str(), "a");
    if (g_file) {
        using namespace std::chrono;
        const auto ts = duration_cast<milliseconds>(
            system_clock::now().time_since_epoch()).count();
        std::fprintf(g_file, "\n=== llobotomy boot · %s ===\n",
                     fmt_timestamp_iso(ts).c_str());
        std::fflush(g_file);
    } else {
        std::fprintf(stderr, "[logger] failed to open log file: %s\n",
                     g_path.c_str());
    }
}

void LoggerShutdown() {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_file) { std::fclose(g_file); g_file = nullptr; }
    g_app  = nullptr;
    g_path.clear();
}

const std::string& LoggerPath() { return g_path; }

void LoggerPush(Severity sev, std::string kind, std::string msg) {
    using namespace std::chrono;
    const auto ts_ms = duration_cast<milliseconds>(
        system_clock::now().time_since_epoch()).count();
    // 1. In-memory ring (AppState handles its own mutex).
    AppState* app = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        app = g_app;
    }
    if (app) app->pushLog(sev, kind, msg);
    // 2. On-disk file.
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_file) {
        std::fprintf(g_file, "%s [%s] [%s] %s\n",
                     fmt_timestamp_iso(ts_ms).c_str(),
                     SeverityShort(sev), kind.c_str(), msg.c_str());
        std::fflush(g_file);
    }
}

void LoggerPushFmt(Severity sev, const char* kind, const char* fmt, ...) {
    char buf[2048];
    va_list ap; va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    LoggerPush(sev, std::string(kind), std::string(buf));
}

}  // namespace llob
