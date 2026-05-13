// gguf_inspector_engine.cpp — Model backend for inspecting GGUF weight files.
//
// Lifecycle (mirroring hf_proxy_engine.cpp):
//
//   loadCheckpoint(path)    parse header → build GgufSource → fill view
//   unloadCheckpoint()      view.clear() → source released
//
// The heavy lifting is done by:
//   parse_gguf()  (gguf_parser.cpp)  — reads header, returns GgufHeader
//   GgufSource    (gguf_source.cpp)  — mmap/pread on the data section
//
// All "live" getters (getAttentionPattern, getMlpFeatures, …) inherit
// from MockModel without override.  The engine's only real overrides are
// view(), getCapabilities(), getModelInfo(), loadCheckpoint(), and
// unloadCheckpoint().  This keeps GgufInspectorEngine.cpp to ~200 lines
// while the entire Model interface is satisfied.

#include "llm_engine/gguf_inspector_engine.hpp"
#include "llm_engine/gguf_parser.hpp"
#include "llm_engine/gguf_source.hpp"
#include "llm_engine/model_view.hpp"

#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace llmengine {

namespace {

std::int64_t now_ms_gguf() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(
        system_clock::now().time_since_epoch()).count();
}

}  // namespace

// ── PIMPL state ───────────────────────────────────────────────────────────

struct GgufInspectorEngine::State {
    ModelView              view;
    mutable std::mutex     mu;
    std::vector<LogEntry>  pending_logs;

    void log(Severity sev, const char* kind, std::string msg) {
        std::lock_guard<std::mutex> lk(mu);
        pending_logs.push_back({now_ms_gguf(), sev, kind, std::move(msg)});
    }
};

// ── Construction / destruction ─────────────────────────────────────────────

GgufInspectorEngine::GgufInspectorEngine()
    : m_state(std::make_unique<State>()) {}

GgufInspectorEngine::~GgufInspectorEngine() = default;

// ── loadCheckpoint ────────────────────────────────────────────────────────

Model::CheckpointResult
GgufInspectorEngine::loadCheckpoint(std::string_view path) {
    return loadCheckpoint(path, LoadOptions{});
}

Model::CheckpointResult
GgufInspectorEngine::loadCheckpoint(std::string_view path,
                                    const LoadOptions& /*opts*/) {
    const std::string spath(path);
    if (spath.empty()) {
        const std::string err = "loadCheckpoint: empty path";
        m_state->log(Severity::Error, "gguf", err);
        return {false, err};
    }

    // ── Parse the GGUF header ────────────────────────────────────────────
    m_state->log(Severity::Info, "gguf", "parsing header: " + spath);
    auto hdr = parse_gguf(spath);
    if (!hdr) {
        const std::string err = "GGUF parse failed: " + spath;
        m_state->log(Severity::Error, "gguf", err);
        return {false, err};
    }

    // ── Construct the shared GgufSource ──────────────────────────────────
    auto source = std::make_shared<GgufSource>(spath, hdr->data_section_offset);

    // ── Build view.tensors ───────────────────────────────────────────────
    TensorRegistry registry;
    registry.all.reserve(hdr->tensors.size());
    for (const auto& ti : hdr->tensors) {
        TensorHandle h;
        h.source      = source;
        h.name        = ti.canonical;  // engine canonical (normalised)
        h.dtype       = ti.dtype;
        h.shape       = ti.shape;
        h.byte_offset = ti.byte_offset;
        h.byte_length = ti.byte_length;
        h.contiguous  = true;
        registry.insert(std::move(h));
    }

    // ── Populate view (under the mutex in case of concurrent reads) ───────
    {
        std::lock_guard<std::mutex> lk(m_state->mu);

        m_state->view.clear();   // discard any prior session

        m_state->view.provenance = {
            .path         = spath,
            .format       = "gguf",
            .content_hash = "",
            .source_label = "GGUF v" + std::to_string(hdr->version) +
                            " (" + (hdr->architecture.empty() ? "unknown" : hdr->architecture) + ")",
        };

        m_state->view.topology = {
            .name        = spath.substr(spath.find_last_of("/\\") + 1),
            .nLayers     = hdr->n_layers,
            .nHeads      = hdr->n_heads,
            .nKvHeads    = hdr->n_kv_heads,
            .dModel      = hdr->d_model,
            .dHead       = hdr->d_head,
            .dMlp        = hdr->d_mlp,
            .vocab       = hdr->vocab,
            .maxPos      = hdr->max_pos,
            .ropeTheta   = hdr->rope_theta,
            .totalParams = kNoSize,          // not computed here
            .chatTemplate = hdr->chat_template,
            .bosToken    = hdr->bos_token,
            .eosToken    = hdr->eos_token,
        };

        m_state->view.tokenizer.bos_token     = hdr->bos_token;
        m_state->view.tokenizer.eos_token     = hdr->eos_token;
        m_state->view.tokenizer.chat_template = hdr->chat_template;
        // encode / decode not wired — no tokenizer runtime available here

        m_state->view.tensors = std::move(registry);

        m_state->view.capabilities = Model::Capabilities{
            .has_topology      = true,
            .has_tokenizer     = false,
            .has_state_dict    = true,
            .has_attention     = false,
            .has_residual      = false,
            .has_logit_lens    = false,
            .has_token_stream  = false,
            .has_captures      = false,
            .has_intervention  = false,
            .has_weight_deltas = false,
            .has_training      = false,
        };
    }

    m_state->log(Severity::Info, "gguf",
                 "loaded " + std::to_string(hdr->tensors.size()) +
                 " tensors · arch=" + hdr->architecture +
                 " · layers=" + std::to_string(hdr->n_layers) +
                 " · d_model=" + std::to_string(hdr->d_model));

    return {true, ""};
}

// ── unloadCheckpoint ──────────────────────────────────────────────────────

void GgufInspectorEngine::unloadCheckpoint() {
    // clear() resets every field.  The shared_ptrs in TensorRegistry drop
    // their reference to the GgufSource; if this is the last owner the
    // mmap is released and the fd is closed.
    {
        std::lock_guard<std::mutex> lk(m_state->mu);
        m_state->view.clear();
        // Push the log entry into the buffer under the lock, then release
        // before returning so drainEngineLogs can acquire immediately.
        m_state->pending_logs.push_back(
            {now_ms_gguf(), Severity::Info, "gguf", "unloaded"});
    }
}

// ── view() / getCapabilities() / getModelInfo() ───────────────────────────

const ModelView& GgufInspectorEngine::view() const {
    // No lock — see ModelView threading contract (steady-state reads are safe).
    return m_state->view;
}

Model::Capabilities GgufInspectorEngine::getCapabilities() const {
    std::lock_guard<std::mutex> lk(m_state->mu);
    return m_state->view.capabilities;
}

ModelInfo GgufInspectorEngine::getModelInfo() {
    std::lock_guard<std::mutex> lk(m_state->mu);
    return m_state->view.topology;
}

// ── drainEngineLogs ───────────────────────────────────────────────────────

std::vector<LogEntry> GgufInspectorEngine::drainEngineLogs() {
    std::lock_guard<std::mutex> lk(m_state->mu);
    std::vector<LogEntry> out;
    out.swap(m_state->pending_logs);
    return out;
}

}  // namespace llmengine
