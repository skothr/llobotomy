// Probes / SAE workspace — feature browser | feature card + SAE training |
// probe ops + export.  HANDOFF §3.4.

#include "workspaces/workspaces.hpp"

#include "appstate.hpp"
#include "model/model.hpp"
#include "style.hpp"
#include "ui/chrome.hpp"
#include "ui/colormap.hpp"
#include "ui/widgets.hpp"

#include <imgui.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <string>

namespace llob {

namespace {

struct Feature { int id; const char* label; int layer; float l0; float acts; const char* type; };

const std::array<Feature, 16>& Features() {
    static const std::array<Feature, 16> F = {{
        { 2381, "attends_to_subset",     8, 17.4f, 0.412f, "SAE" },
        {  471, "attention/head_pat",    7, 22.1f, 0.218f, "SAE" },
        { 1923, "induction-A→B",         9, 14.0f, 0.181f, "SAE" },
        { 3014, "self-reference",        6, 31.2f, 0.094f, "linear" },
        {   80, "syntax/clause",         5, 18.4f, 0.142f, "linear" },
        { 2557, "code/python_def",       3, 12.0f, 0.301f, "SAE" },
        { 1134, "json/key",              4, 41.1f, 0.061f, "mlp" },
        { 2202, "url/path",              7, 22.5f, 0.184f, "logit" },
        {   42, "refusal_dir",           8,  9.8f, 0.421f, "linear" },
        {  998, "sentiment_neg",        10, 26.8f, 0.114f, "linear" },
        {  111, "sentiment_pos",        10, 19.4f, 0.182f, "linear" },
        { 4119, "list-bullet",           4, 15.2f, 0.244f, "SAE" },
        { 8412, "sql/select",            6, 12.0f, 0.272f, "SAE" },
        { 9221, "paren-depth-3",         5, 33.4f, 0.091f, "mlp" },
        { 7100, "next-token-pred",      11, 21.8f, 0.171f, "logit" },
        { 6210, "multilingual",          9, 27.2f, 0.130f, "linear" },
    }};
    return F;
}

ImU32 TypeColor(const char* type) {
    if (!std::strcmp(type, "SAE"))    return Sty().accent;
    if (!std::strcmp(type, "linear")) return Sty().info;
    if (!std::strcmp(type, "logit"))  return Sty().good;
    (void)type;
    return Sty().warn;
}

void DrawFeatureBrowser(int& selected) {
    DrawTitleBar("features", "◈", "SAE-1M • 16 / 16384 shown", "feat-browser");
    if (!ImGui::BeginChild("##fb_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    static char filter[64] = {};
    ImGui::SetNextItemWidth(-1);
    ImGui::InputTextWithHint("##search", "search features… (e.g. 'refusal')", filter, sizeof filter);
    if (ImGui::BeginTable("##features", 5, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInner |
                                            ImGuiTableFlags_ScrollY)) {
        ImGui::TableSetupColumn("id",    ImGuiTableColumnFlags_WidthFixed, 38);
        ImGui::TableSetupColumn("label", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("L",     ImGuiTableColumnFlags_WidthFixed, 32);
        ImGui::TableSetupColumn("L0",    ImGuiTableColumnFlags_WidthFixed, 38);
        ImGui::TableSetupColumn("act",   ImGuiTableColumnFlags_WidthFixed, 48);
        ImGui::TableHeadersRow();
        for (auto& f : Features()) {
            ImGui::TableNextRow();
            ImGui::PushID(f.id);
            ImGui::TableNextColumn();
            const bool sel = (f.id == selected);
            if (ImGui::Selectable("##row", sel, ImGuiSelectableFlags_SpanAllColumns))
                selected = f.id;
            ImGui::SameLine(0, 0);
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
            ImGui::Text("%d", f.id); ImGui::PopStyleColor();
            ImGui::TableNextColumn();
            ImGui::PushStyleColor(ImGuiCol_Text, TypeColor(f.type));
            ImGui::Text("%c", f.type[0]); ImGui::PopStyleColor();
            ImGui::SameLine(); ImGui::Text("%s", f.label);
            ImGui::TableNextColumn(); ImGui::Text("L%02d", f.layer);
            ImGui::TableNextColumn(); ImGui::Text("%.0f", f.l0);
            ImGui::TableNextColumn(); ImGui::Text("%.3f", f.acts);
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    ImGui::EndChild();
}

void DrawFeatureCard(int feature, Model& m) {
    char title[32]; std::snprintf(title, sizeof title, "feature[%d]", feature);
    DrawTitleBar(title, "◉", "SAE/L08.mlp_out", "feat-detail");
    if (!ImGui::BeginChild("##fc_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    char idl[16]; std::snprintf(idl, sizeof idl, "f%d", feature);
    if (auto sec = BeginSection("Feature card", true)) {
        KV({
            { "id",                   idl,                                     "accent" },
            { "layer",                "L08.mlp_out",                            "accent" },
            { "type",                 "SAE feature (top-K, k=64)",              "" },
            { "L0 sparsity",          "17.4",                                   "" },
            { "fires/M tok",          "4,182",                                  "" },
            { "mean act when firing", "+2.184",                                 "warn" },
            { "decoder ↓ logits",     "+the (+0.34)  +to (+0.21)  +of (+0.18)", "good" },
            { "decoder ↑ logits",     "-of (-0.41)  -if (-0.22)  -in (-0.18)",  "bad"  },
        });
        EndSection(sec);
    }
    if (auto sec = BeginSection("Top activating examples", false, "from 100M tokens")) {
        const struct { const char* pre; const char* hl; const char* post; float score; } exs[] = {
            {"…each attention head ", "attends", " to a subset of the previous tokens.", 4.21f},
            {"…the model ",           "attends", " selectively when computing…",         3.84f},
            {"…the operator ",        "attends", " to relevant context windows…",        3.62f},
        };
        for (auto& e : exs) {
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
            ImGui::Text("%.2f  %s", e.score, e.pre);
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().warn);
            ImGui::TextUnformatted(e.hl);
            ImGui::PopStyleColor();
            ImGui::SameLine();
            ImGui::TextUnformatted(e.post);
            ImGui::PopStyleColor();
        }
        EndSection(sec);
    }
    if (auto sec = BeginSection("Activation across layers", false, "logit-lens")) {
        const auto bins = m.getWeightHistogram(8, "feature", 40);
        const LensAnnotation an[] = {
            { 0.55f, "firing thresh", Sty().accent },
            { 0.84f, "top-1",         Sty().warn   },
        };
        ActivationHistogram(bins, ImGui::GetContentRegionAvail().x - 8, 84, Sty().info,
                            std::span{an, std::size(an)});
        EndSection(sec);
    }
    ImGui::EndChild();
}

void DrawProbeOps(AppState& s) {
    DrawTitleBar("probe_ops", "◎", nullptr, "probe-ops");
    if (!ImGui::BeginChild("##po_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    static char probeName[64] = "refusal_v2";
    static int  kindIdx = 0;
    static bool training = false;
    static int  step = 256;

    if (auto sec = BeginSection("Train new probe", true)) {
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted); ImGui::TextUnformatted("target"); ImGui::PopStyleColor();
        ImGui::SameLine(80); ImGui::SetNextItemWidth(160);
        ImGui::InputText("##pname", probeName, sizeof probeName);
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted); ImGui::TextUnformatted("at"); ImGui::PopStyleColor();
        ImGui::SameLine(80);
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().accent); ImGui::TextUnformatted("L08.resid_post"); ImGui::PopStyleColor();
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted); ImGui::TextUnformatted("type"); ImGui::PopStyleColor();
        ImGui::SameLine(80);
        ImGui::RadioButton("linear",  &kindIdx, 0); ImGui::SameLine();
        ImGui::RadioButton("logistic",&kindIdx, 1);

        if (ImGui::Button(training ? "|| pause" : "> train probe")) {
            training = !training;
            s.pushLog("probe", training ? "training probe" : "paused probe training");
            if (training) step = 0;
        }
        ImGui::SameLine();
        if (ImGui::Button("save")) s.pushLog("probe", "saved probe");
        if (training) {
            step = std::min(512, step + 8);
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().accent);
            ImGui::Text("step %d/512", step); ImGui::PopStyleColor();
            if (step >= 512) { training = false; s.pushLog("probe", "probe train complete · acc 0.91"); }
        }
        EndSection(sec);
    }
    if (auto sec = BeginSection("Validation curves", true)) {
        const float vd[] = {0.51f,0.55f,0.6f,0.65f,0.71f,0.78f,0.83f,0.86f,0.88f,0.9f,0.91f,0.91f};
        SparkOpts so{}; so.color = Sty().good; so.fill = true; so.width = 140; so.height = 36;
        Sparkline(vd, so);
        ImGui::SameLine();
        KV({ {"train","0.92","good"}, {"val","0.89","good"}, {"gap","0.03",""} }, true);
        EndSection(sec);
    }
    if (auto sec = BeginSection("Probe library", false, "14")) {
        if (ImGui::BeginTable("##lib", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInner)) {
            ImGui::TableSetupColumn(""); ImGui::TableSetupColumn("name"); ImGui::TableSetupColumn("acc", ImGuiTableColumnFlags_WidthFixed, 50);
            ImGui::TableHeadersRow();
            const struct { const char* p; const char* tone; const char* n; const char* a; bool sel; } R[] = {
                { "L", "accent", "refusal_dir",      "0.91", true },
                { "L", "accent", "truthfulness",     "0.81", false },
                { "L", "accent", "code_v_prose",     "0.94", false },
                { "S", "warn",   "SAE/feature_2381", "0.62", false },
                { "P", "dim",    "multilingual",     "0.74", false },
            };
            for (auto& r : R) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn(); Pill(r.p, r.tone, true);
                ImGui::TableNextColumn();
                if (r.sel) { ImGui::PushStyleColor(ImGuiCol_Text, Sty().accent); ImGui::TextUnformatted(r.n); ImGui::PopStyleColor(); }
                else        ImGui::TextUnformatted(r.n);
                ImGui::TableNextColumn(); ImGui::TextUnformatted(r.a);
            }
            ImGui::EndTable();
        }
        EndSection(sec);
    }
    ImGui::EndChild();
}

void DrawSAETrain() {
    DrawTitleBar("sae_training", "∑", nullptr, "sae-train");
    if (!ImGui::BeginChild("##sae_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    if (auto sec = BeginSection("SAE training run · L08.mlp_out · step 412k/1M", true)) {
        static const float DRECON[] = {1.2f,1.0f,0.84f,0.71f,0.62f,0.55f,0.50f,0.47f,0.45f,0.43f,0.42f,0.41f};
        static const float DL0   [] = {80,72,65,58,52,47,43,40,38,36,35,34};
        static const float DDEAD [] = {120,82,61,42,38,31,28,26,25,24,23,22};
        static const float DEXPL [] = {0.41f,0.52f,0.61f,0.69f,0.74f,0.78f,0.81f,0.83f,0.84f,0.85f,0.86f,0.86f};
        const struct { const char* lbl; const float* d; std::size_t n; ImU32 col; const char* val; const char* tone; }
            cards[] = {
                { "RECON LOSS",    DRECON, std::size(DRECON), Sty().accent, "0.412",      "" },
                { "L0 SPARSITY",   DL0,    std::size(DL0),    Sty().warn,   "34.2",       "warn" },
                { "DEAD FEATURES", DDEAD,  std::size(DDEAD),  Sty().bad,    "22 / 16384", "bad"  },
                { "EXPL VAR",      DEXPL,  std::size(DEXPL),  Sty().good,   "0.864",      "good" },
            };
        const float card_w = (ImGui::GetContentRegionAvail().x - 24) / 4.0f;
        for (int i = 0; i < 4; ++i) {
            ImGui::BeginGroup();
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
            ImGui::TextUnformatted(cards[i].lbl); ImGui::PopStyleColor();
            SparkOpts so{}; so.color = cards[i].col; so.fill = true; so.width = card_w - 8; so.height = 50;
            Sparkline(std::span{cards[i].d, cards[i].n}, so);
            ImGui::PushStyleColor(ImGuiCol_Text, cards[i].tone[0] ? ::llob::ToneColor(cards[i].tone) : Sty().text_bright);
            ImGui::TextUnformatted(cards[i].val); ImGui::PopStyleColor();
            ImGui::EndGroup();
            if (i < 3) ImGui::SameLine();
        }
        EndSection(sec);
    }
    ImGui::EndChild();
}

void DrawExport(AppState& s) {
    DrawTitleBar("export", "↗", nullptr, "export");
    if (!ImGui::BeginChild("##ex_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    char ab[16]; std::snprintf(ab, sizeof ab, "%zu", s.ablatedHeads.size());
    char pr[16]; std::snprintf(pr, sizeof pr, "%zu", s.probedHeads.size());
    if (auto sec = BeginSection("Snapshot", true)) {
        KV({
            { "active probes",   pr,            "accent" },
            { "active features", "60",          "accent" },
            { "ablations",       ab,            "warn" },
            { "steering",        "on (α=1.40)", "warn" },
        });
        EndSection(sec);
    }
    if (ImGui::Button("↗ export state")) s.pushLog("export", "state snapshot exported → ./out/state.json");
    ImGui::SameLine();
    if (ImGui::Button("save preset"))   s.pushLog("export", "preset saved");
    ImGui::EndChild();
}

}  // namespace

void DrawProbesWorkspace(AppState& s, Model& m) {
    static int feature = 2381;
    const float W = ImGui::GetContentRegionAvail().x;
    const float H = ImGui::GetContentRegionAvail().y;
    const float gap = 1.0f, lw = 320.0f, rw = 320.0f;
    const float cw = std::max(200.0f, W - lw - rw - 2 * gap);
    const float bot_h = std::min(220.0f, H * 0.30f);
    const float top_h = H - bot_h - gap;

    ImGui::BeginChild("##pr_left", { lw, H }, ImGuiChildFlags_Borders);
    DrawFeatureBrowser(feature); ImGui::EndChild(); ImGui::SameLine(0, gap);

    ImGui::BeginChild("##pr_center", { cw, H });
    ImGui::BeginChild("##pr_card", { cw, top_h }, ImGuiChildFlags_Borders);
    DrawFeatureCard(feature, m); ImGui::EndChild();
    ImGui::BeginChild("##pr_sae",  { cw, bot_h }, ImGuiChildFlags_Borders);
    DrawSAETrain(); ImGui::EndChild();
    ImGui::EndChild(); ImGui::SameLine(0, gap);

    ImGui::BeginChild("##pr_right", { rw, H });
    ImGui::BeginChild("##pr_ops", { rw, top_h }, ImGuiChildFlags_Borders);
    DrawProbeOps(s); ImGui::EndChild();
    ImGui::BeginChild("##pr_exp", { rw, bot_h }, ImGuiChildFlags_Borders);
    DrawExport(s); ImGui::EndChild();
    ImGui::EndChild();
}

}  // namespace llob
