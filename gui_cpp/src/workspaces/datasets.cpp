// Datasets workspace — dataset list | sample browser | stats | [token ids].
// HANDOFF §3.7.

#include "workspaces/workspaces.hpp"

#include "appstate.hpp"
#include "model/model.hpp"
#include "style.hpp"
#include "ui/chrome.hpp"
#include "ui/widgets.hpp"

#include <imgui.h>

#include <cstdio>
#include <vector>

namespace llob {

namespace {

void DrawDatasetList() {
    DrawTitleBar("datasets", "≡", nullptr, "ds-list");
    if (!ImGui::BeginChild("##dl_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    static int sel = 0;
    const struct { const char* n; const char* sz; const char* tk; } R[] = {
        { "the_pile_v2",          "412 GB", "82.4B" },
        { "fineweb_edu",          "132 GB", "28.1B" },
        { "sft/instruction_v3",   "2.4 GB", "128M"  },
        { "harmful_v2",           "42 MB",  "2.0M"  },
        { "mmlu_eval",            "12 MB",  "0.5M"  },
        { "truthful_qa",          "8 MB",   "0.3M"  },
    };
    if (ImGui::BeginTable("##ds", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInner)) {
        ImGui::TableSetupColumn("name");
        ImGui::TableSetupColumn("size", ImGuiTableColumnFlags_WidthFixed, 56);
        ImGui::TableSetupColumn("tok",  ImGuiTableColumnFlags_WidthFixed, 56);
        ImGui::TableHeadersRow();
        for (int i = 0; i < int(std::size(R)); ++i) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            if (ImGui::Selectable(R[i].n, i == sel, ImGuiSelectableFlags_SpanAllColumns)) sel = i;
            ImGui::TableNextColumn(); ImGui::TextUnformatted(R[i].sz);
            ImGui::TableNextColumn(); ImGui::TextUnformatted(R[i].tk);
        }
        ImGui::EndTable();
    }
    ImGui::EndChild();
}

void DrawSampleBrowser() {
    DrawTitleBar("the_pile_v2 · sample browser", "▦", nullptr, "ds-browser");
    if (!ImGui::BeginChild("##sb_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    static char filter[64] = {};
    ImGui::SetNextItemWidth(-80);
    ImGui::InputTextWithHint("##f", "filter examples (regex)…", filter, sizeof filter);
    ImGui::SameLine();
    if (ImGui::SmallButton("(reload)")) {}
    if (auto sec = BeginSection("Sample 4,182 · doc_id=82a17e", true)) {
        ImGui::PushStyleColor(ImGuiCol_ChildBg, Sty().bg_input);
        ImGui::BeginChild("##sb_text", ImVec2(0, 240), ImGuiChildFlags_Borders);
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
        ImGui::TextWrapped("[doc_id=82a17e | source=arxiv/cs.CL]");
        ImGui::PopStyleColor();
        ImGui::TextWrapped("When the transformer processes a sentence, each attention head attends to a subset of the previous tokens.");
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().accent);
        ImGui::TextWrapped("The residual stream");
        ImGui::PopStyleColor();
        ImGui::TextWrapped("carries information from layer to layer through the network, with each block adding its");
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().warn);
        ImGui::TextWrapped("contribution");
        ImGui::PopStyleColor();
        ImGui::TextWrapped("to the running sum.\n\nThe self-attention sublayer computes Q, K, V from the input residual:");
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().accent);
        ImGui::TextWrapped("  Q = LN(x) W_Q\n  K = LN(x) W_K\n  V = LN(x) W_V");
        ImGui::PopStyleColor();
        ImGui::EndChild();
        ImGui::PopStyleColor();
        EndSection(sec);
    }
    if (auto sec = BeginSection("Token statistics", false, "this sample")) {
        KV({
            { "len",            "482 tok",       "" },
            { "ppl (base)",     "4.21",          "" },
            { "ppl (ft)",       "3.84",          "good" },
            { "avg surprisal",  "2.07 nats",     "" },
            { "top fired feat", "f2381 attends_to_subset", "accent" },
        });
        EndSection(sec);
    }
    ImGui::EndChild();
}

void DrawDatasetStats() {
    DrawTitleBar("dataset_stats", "∑", nullptr, "ds-stats");
    if (!ImGui::BeginChild("##stats_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    if (auto sec = BeginSection("distribution", true)) {
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
        ImGui::TextUnformatted("document length"); ImGui::PopStyleColor();
        const int bins[] = {12,18,28,42,68,108,140,160,168,156,128,98,72,48,32,22,14,8,5,3};
        ActivationHistogram(bins, ImGui::GetContentRegionAvail().x - 8, 56, Sty().info);
        EndSection(sec);
    }
    if (auto sec = BeginSection("source mix")) {
        const struct { const char* n; float v; ImU32 c; } S[] = {
            { "common_crawl", 0.62f, Sty().accent },
            { "arxiv",        0.14f, Sty().info },
            { "github",       0.11f, Sty().warn },
            { "books",        0.08f, Sty().good },
            { "wiki",         0.05f, Sty().magenta },
        };
        for (auto& s : S) {
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
            ImGui::Text("%-14s", s.n); ImGui::PopStyleColor();
            ImGui::SameLine(); Bar(s.v, 120, 6, s.c);
            ImGui::SameLine(); ImGui::Text("%.0f%%", s.v * 100);
        }
        EndSection(sec);
    }
    ImGui::EndChild();
}

}  // namespace

void DrawDatasetsWorkspace(AppState& s, Model& m) {
    const float W = ImGui::GetContentRegionAvail().x, H = ImGui::GetContentRegionAvail().y;
    const float gap = 1.0f;
    const bool  raw = s.showRaw;
    const float lw = 320.0f, rw = 280.0f, raw_w = raw ? 160.0f : 0.0f;
    const float cw = std::max(200.0f, W - lw - rw - raw_w - 3 * gap);

    ImGui::BeginChild("##ds_left",  { lw, H }, ImGuiChildFlags_Borders);
    DrawDatasetList(); ImGui::EndChild(); ImGui::SameLine(0, gap);
    ImGui::BeginChild("##ds_center",{ cw, H }, ImGuiChildFlags_Borders);
    DrawSampleBrowser(); ImGui::EndChild(); ImGui::SameLine(0, gap);
    ImGui::BeginChild("##ds_right", { rw, H }, ImGuiChildFlags_Borders);
    DrawDatasetStats(); ImGui::EndChild();
    if (raw) {
        ImGui::SameLine(0, gap);
        ImGui::BeginChild("##ds_raw", { raw_w, H }, ImGuiChildFlags_Borders);
        DrawTitleBar("token_ids", "0x", "u16 [482]", "ds-tokens");
        if (ImGui::BeginChild("##tk_body", ImVec2(0, 0))) {
            const auto buf = m.getActivation(4182, 0, 200);
            HexView(buf, 0, 3, 28, HexMode::Hex); ImGui::EndChild();
        }
        ImGui::EndChild();
    }
}

}  // namespace llob
