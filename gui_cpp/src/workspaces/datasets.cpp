// Datasets workspace — dataset list | sample browser | stats | [token ids].
// HANDOFF §3.7.
//
// All values come from the Model interface; nothing in this file is
// hardcoded.  The data hooks each section needs are documented inline as
// `// [DATA HOOK]` comments naming the Model::* method that supplies them.

#include "workspaces/workspaces.hpp"

#include "appstate.hpp"
#include "model/model.hpp"
#include "style.hpp"
#include "ui/chrome.hpp"
#include "ui/colormap.hpp"
#include "ui/fmt.hpp"
#include "ui/widgets.hpp"

#include <imgui.h>

#include <cstdio>
#include <span>
#include <string>
#include <vector>

namespace llob {

namespace {

void DrawDatasetList(int& sel, std::string& selName, Model& m) {
    DrawTitleBar("datasets", "≡", nullptr, "ds-list");
    if (!ImGui::BeginChild("##dl_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    // [DATA HOOK] Model::getDatasets() — list of available datasets with
    // size + token-count metadata.  Engine source: catalogue file or
    // HF datasets API enumeration.
    const auto dss = m.getDatasets();
    if (dss.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
        ImGui::TextUnformatted("// no datasets available");
        ImGui::PopStyleColor();
        ImGui::EndChild(); return;
    }
    if (ImGui::BeginTable("##ds", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInner)) {
        ImGui::TableSetupColumn("name");
        ImGui::TableSetupColumn("size", ImGuiTableColumnFlags_WidthFixed, 56);
        ImGui::TableSetupColumn("tok",  ImGuiTableColumnFlags_WidthFixed, 56);
        ImGui::TableHeadersRow();
        for (int i = 0; i < int(dss.size()); ++i) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            if (ImGui::Selectable(dss[i].name.c_str(), i == sel,
                                  ImGuiSelectableFlags_SpanAllColumns)) {
                sel = i;
                selName = dss[i].name;
            }
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(FmtSize(dss[i].size_bytes).c_str());
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(FmtTokens(dss[i].n_tokens).c_str());
        }
        ImGui::EndTable();
    }
    if (selName.empty() && !dss.empty()) selName = dss[0].name;
    ImGui::EndChild();
}

void DrawSampleBrowser(const std::string& dataset, int sample_id, Model& m) {
    char title[128]; std::snprintf(title, sizeof title, "%s · sample browser",
                                    dataset.empty() ? "?" : dataset.c_str());
    DrawTitleBar(title, "▦", nullptr, "ds-browser");
    if (!ImGui::BeginChild("##sb_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    static char filter[64] = {};
    ImGui::SetNextItemWidth(-80);
    ImGui::InputTextWithHint("##f", "filter examples (regex)…", filter, sizeof filter);
    ImGui::SameLine();
    if (ImGui::SmallButton("(reload)")) {}

    // [DATA HOOK] Model::getSample(dataset, sampleId) — text + highlight
    // span list for the sample at this position in the dataset.  Spans
    // mark interesting regions for inline emphasis (e.g. attention-target
    // tokens, key terms, ablation-affected positions).
    const auto sample = m.getSample(dataset, sample_id);
    char hdr[64]; std::snprintf(hdr, sizeof hdr, "Sample %d · doc_id=%s",
                                 sample.sample_id, Or(sample.doc_id).c_str());
    if (auto sec = BeginSection(hdr, true)) {
        ImGui::PushStyleColor(ImGuiCol_ChildBg, Sty().bg_input);
        ImGui::BeginChild("##sb_text", ImVec2(0, 240), ImGuiChildFlags_Borders);
        // Render text + spans.  Each span (begin, end, kind) marks an
        // accent-coloured / warn-coloured slice of the text.
        if (sample.text.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
            ImGui::TextUnformatted("// no sample text");
            ImGui::PopStyleColor();
        } else {
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
            ImGui::Text("[doc_id=%s | source=%s]",
                         sample.doc_id.c_str(), sample.source.c_str());
            ImGui::PopStyleColor();
            // Naïve span renderer: split the text into runs by span
            // boundaries and switch text color per run.
            int cursor = 0;
            const char* full = sample.text.c_str();
            for (const auto& sp : sample.spans) {
                if (sp.begin > cursor) ImGui::TextWrapped("%.*s", sp.begin - cursor, full + cursor);
                ImGui::PushStyleColor(ImGuiCol_Text, sp.kind == 0 ? Sty().accent : Sty().warn);
                ImGui::TextWrapped("%.*s", sp.end - sp.begin, full + sp.begin);
                ImGui::PopStyleColor();
                cursor = sp.end;
            }
            if (cursor < int(sample.text.size()))
                ImGui::TextWrapped("%s", full + cursor);
        }
        ImGui::EndChild();
        ImGui::PopStyleColor();
    }
    if (auto sec = BeginSection("Token statistics", false, "this sample")) {
        // [DATA HOOK] Model::getSampleStats(dataset, sampleId) — len,
        // base/ft perplexity, avg surprisal, top fired feature.
        const auto st = m.getSampleStats(dataset, sample_id);
        char len[32]; std::snprintf(len, sizeof len, "%s tok", FmtInt(st.len_tokens).c_str());
        char surp[24];
        if (std::isnan(st.avg_surprisal)) std::snprintf(surp, sizeof surp, "—");
        else                                std::snprintf(surp, sizeof surp, "%.2f nats", double(st.avg_surprisal));
        KV({
            { "len",            len,                              "" },
            { "ppl (base)",     FmtFloat(st.ppl_base, "%.2f"),    "" },
            { "ppl (ft)",       FmtFloat(st.ppl_ft,   "%.2f"),    "good" },
            { "avg surprisal",  surp,                              "" },
            { "top fired feat", Or(st.top_feature),                "accent" },
        });
    }
    ImGui::EndChild();
}

void DrawDatasetStats(const std::string& dataset, Model& m) {
    DrawTitleBar("dataset_stats", "∑", nullptr, "ds-stats");
    if (!ImGui::BeginChild("##stats_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    // [DATA HOOK] Model::getDatasetDistribution(dataset) — document length
    // histogram + source-mix bars.  Engine source: pre-computed catalogue
    // metadata (or sampled on first access).
    const auto dist = m.getDatasetDistribution(dataset);
    if (auto sec = BeginSection("distribution", true)) {
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
        ImGui::TextUnformatted("document length"); ImGui::PopStyleColor();
        if (dist.doc_length_histogram.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
            ImGui::TextUnformatted("// no length data");
            ImGui::PopStyleColor();
        } else {
            ActivationHistogram(std::span{dist.doc_length_histogram},
                                ImGui::GetContentRegionAvail().x - 8, 56, Sty().info);
        }
    }
    if (auto sec = BeginSection("source mix")) {
        if (dist.source_mix.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
            ImGui::TextUnformatted("// no source-mix breakdown");
            ImGui::PopStyleColor();
        } else {
            for (const auto& sm : dist.source_mix) {
                ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
                ImGui::Text("%-14s", sm.name.c_str()); ImGui::PopStyleColor();
                ImGui::SameLine(); Bar(sm.fraction, 120, 6, ToneColor(sm.tone));
                ImGui::SameLine(); ImGui::Text("%.0f%%", double(sm.fraction * 100));
            }
        }
    }
    ImGui::EndChild();
}

}  // namespace

void DrawDatasetsWorkspace(AppState& s, Model& m) {
    static int         dsSel = 0;
    static std::string dsName;
    static int         sampleId = 4182;

    const float W = ImGui::GetContentRegionAvail().x, H = ImGui::GetContentRegionAvail().y;
    const float gap = 1.0f;
    const bool  raw = s.showRaw;
    const float lw = 320.0f, rw = 280.0f, raw_w = raw ? 160.0f : 0.0f;
    const float cw = std::max(200.0f, W - lw - rw - raw_w - 3 * gap);

    ImGui::BeginChild("##ds_left",  { lw, H }, ImGuiChildFlags_Borders);
    DrawDatasetList(dsSel, dsName, m); ImGui::EndChild(); ImGui::SameLine(0, gap);

    ImGui::BeginChild("##ds_center",{ cw, H }, ImGuiChildFlags_Borders);
    DrawSampleBrowser(dsName, sampleId, m); ImGui::EndChild(); ImGui::SameLine(0, gap);

    ImGui::BeginChild("##ds_right", { rw, H }, ImGuiChildFlags_Borders);
    DrawDatasetStats(dsName, m); ImGui::EndChild();

    if (raw) {
        ImGui::SameLine(0, gap);
        ImGui::BeginChild("##ds_raw", { raw_w, H }, ImGuiChildFlags_Borders);
        DrawTitleBar("token_ids", "0x", "u16", "ds-tokens");
        if (ImGui::BeginChild("##tk_body", ImVec2(0, 0))) {
            // [DATA HOOK] Model::getTokenIds(dataset, sampleId, n) — the
            // raw token id sequence for hex display.  Real backend: pull
            // from the cached encoded sample.
            const auto buf = m.getTokenIds(dsName, sampleId, 200);
            if (buf.empty()) {
                ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
                ImGui::TextUnformatted("// no token ids");
                ImGui::PopStyleColor();
            } else {
                HexView(buf, 0, 3, 28, HexMode::Hex);
            }
            ImGui::EndChild();
        }
        ImGui::EndChild();
    }
}

}  // namespace llob
