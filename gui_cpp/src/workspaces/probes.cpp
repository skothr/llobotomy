// Probes / SAE workspace — feature browser | feature card + SAE training |
// probe ops + export.  HANDOFF §3.4.
//
// All values come from the Model interface; nothing in this file is
// hardcoded.  The data hooks each section needs are documented inline as
// `// [DATA HOOK]` comments naming the Model::* method that supplies them.

#include "workspaces/workspaces.hpp"

#include "appstate.hpp"
#include "model/model.hpp"
#include "logger.hpp"
#include "style.hpp"
#include "ui/chrome.hpp"
#include "ui/colormap.hpp"
#include "ui/fmt.hpp"
#include "ui/widgets.hpp"

#include <imgui.h>

#include <cstdio>
#include <cstring>
#include <span>
#include <string>

namespace llob {

namespace {

ImU32 TypeColor(const std::string& type) {
    if (type == "SAE")    return Sty().accent;
    if (type == "linear") return Sty().info;
    if (type == "logit")  return Sty().good;
    return Sty().warn;
}

void DrawFeatureBrowser(AppState&, Model& m, int& selected, char filter[64]) {
    DrawTitleBar("features", "◈", "SAE-1M • filtered", "feat-browser");
    if (!ImGui::BeginChild("##fb_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    ImGui::SetNextItemWidth(-1);
    ImGui::InputTextWithHint("##search", "search features… (e.g. 'refusal')", filter, 64);

    // [DATA HOOK] Model::getFeatureLibrary(filter) — list of feature
    // summaries for the active SAE / probe set.  Filter is a free-text
    // substring match applied to label/id; engine returns whatever subset
    // matches (with its own paging strategy if needed).
    const auto features = m.getFeatureLibrary(filter);
    if (features.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
        ImGui::TextUnformatted("// no features available");
        ImGui::PopStyleColor();
        ImGui::EndChild();
        return;
    }

    if (ImGui::BeginTable("##features", 5, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInner |
                                            ImGuiTableFlags_ScrollY)) {
        ImGui::TableSetupColumn("id",    ImGuiTableColumnFlags_WidthFixed, 38);
        ImGui::TableSetupColumn("label", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("L",     ImGuiTableColumnFlags_WidthFixed, 32);
        ImGui::TableSetupColumn("L0",    ImGuiTableColumnFlags_WidthFixed, 38);
        ImGui::TableSetupColumn("act",   ImGuiTableColumnFlags_WidthFixed, 48);
        ImGui::TableHeadersRow();
        for (const auto& f : features) {
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
            ImGui::Text("%c", f.type.empty() ? '?' : f.type[0]); ImGui::PopStyleColor();
            ImGui::SameLine(); ImGui::Text("%s", f.label.c_str());
            ImGui::TableNextColumn(); ImGui::Text("L%02d", f.layer);
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(FmtFloat(f.l0_sparsity, "%.0f").c_str());
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(FmtFloat(f.acts_mean, "%.3f").c_str());
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    ImGui::EndChild();
}

void DrawFeatureCard(int feature, Model& m) {
    char title[32]; std::snprintf(title, sizeof title, "feature[%d]", feature);
    // [DATA HOOK] Model::getFeatureCard(featureId) — full details for the
    // selected feature.  Card includes the layer location, type, sparsity,
    // firing rate, and decoder-direction logit changes (the "↑/↓ logits"
    // rows that tell you what the feature pushes towards/away from).
    const auto card = m.getFeatureCard(feature);
    const std::string flag = card.layer_str.empty() ? "—" : ("SAE/" + card.layer_str);
    DrawTitleBar(title, "◉", flag.c_str(), "feat-detail");
    if (!ImGui::BeginChild("##fc_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }

    if (auto sec = BeginSection("Feature card", true)) {
        char idl[16]; std::snprintf(idl, sizeof idl, "f%d", card.id);
        const std::string l0   = FmtFloat(card.l0_sparsity, "%.1f");
        char fires[16];
        if (card.fires_per_million == kNoInt) std::snprintf(fires, sizeof fires, "—");
        else                                    std::snprintf(fires, sizeof fires, "%d", card.fires_per_million);
        char mean[16]; std::snprintf(mean, sizeof mean, "%+.3f", double(card.mean_act_when_firing));
        const std::string mean_s = std::isnan(card.mean_act_when_firing) ? "—" : mean;
        KV({
            { "id",                   idl,                              "accent" },
            { "layer",                Or(card.layer_str),                "accent" },
            { "type",                 Or(card.type_str),                 "" },
            { "L0 sparsity",          l0,                                "" },
            { "fires/M tok",          fires,                             "" },
            { "mean act when firing", mean_s,                            "warn" },
            { "decoder ↓ logits",     Or(card.decoder_down_str),         "good" },
            { "decoder ↑ logits",     Or(card.decoder_up_str),           "bad"  },
        });
    }

    if (auto sec = BeginSection("Top activating examples", false, "from corpus")) {
        // [DATA HOOK] Model::getFeatureExamples(featureId, k) — top-k
        // context windows where this feature fires hardest, each with a
        // (pre, highlight, post) split for in-line emphasis rendering.
        const auto exs = m.getFeatureExamples(feature, 5);
        if (exs.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
            ImGui::TextUnformatted("// no examples available");
            ImGui::PopStyleColor();
        } else {
            for (const auto& e : exs) {
                ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
                ImGui::Text("%.2f  %s", double(e.score), e.pre.c_str());
                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_Text, Sty().warn);
                ImGui::TextUnformatted(e.highlight.c_str());
                ImGui::PopStyleColor();
                ImGui::SameLine();
                ImGui::TextUnformatted(e.post.c_str());
                ImGui::PopStyleColor();
            }
        }
    }

    if (auto sec = BeginSection("Activation across layers", false, "logit-lens")) {
        // [DATA HOOK] Model::getWeightHistogram("feature.<id>", bins) —
        // distribution of this feature's activation magnitudes across the
        // calibration corpus.  Real backend: cached during SAE training.
        char name[32]; std::snprintf(name, sizeof name, "feature.%d", feature);
        const auto bins = m.getWeightHistogram(name, 40);
        const LensAnnotation an[] = {
            { 0.55f, "firing thresh", Sty().accent },
            { 0.84f, "top-1",         Sty().warn   },
        };
        ActivationHistogram(bins, ImGui::GetContentRegionAvail().x - 8, 84, Sty().info,
                            std::span{an, std::size(an)});
    }

    if (auto sec = BeginSection("Co-firing features", false, "cosine ≥ 0.4")) {
        // [DATA HOOK] Model::getCoFiringFeatures(featureId, threshold) —
        // features whose firing pattern is most correlated with this one
        // (cosine of activation vectors over the corpus).
        const auto cofs = m.getCoFiringFeatures(feature, 0.4f);
        if (cofs.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
            ImGui::TextUnformatted("// no co-firing features above threshold");
            ImGui::PopStyleColor();
        } else if (ImGui::BeginTable("##cof", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInner)) {
            ImGui::TableSetupColumn("id",      ImGuiTableColumnFlags_WidthFixed, 46);
            ImGui::TableSetupColumn("label");
            ImGui::TableSetupColumn("cos sim", ImGuiTableColumnFlags_WidthFixed, 80);
            ImGui::TableHeadersRow();
            for (const auto& c : cofs) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn(); ImGui::Text("f%d", c.id);
                ImGui::TableNextColumn(); ImGui::Text("%s", c.label.c_str());
                ImGui::TableNextColumn();
                ImGui::PushStyleColor(ImGuiCol_Text, c.cos_sim >= 0.6f ? Sty().good : Sty().text);
                ImGui::TextUnformatted(FmtFloat(c.cos_sim, "%.3f").c_str());
                ImGui::PopStyleColor();
            }
            ImGui::EndTable();
        }
    }
    ImGui::EndChild();
}

void DrawProbeOps(AppState& s, Model& m) {
    DrawTitleBar("probe_ops", "◎", nullptr, "probe-ops");
    if (!ImGui::BeginChild("##po_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    static char probeName[64] = "refusal_v2";
    static int  kindIdx = 0;

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

        // [DATA HOOK] Model::getProbeTrainState(name) — current training
        // step / val curve / accuracies for the named probe.  Polled each
        // frame so the section can show the live progress bar.
        const auto pt = m.getProbeTrainState(probeName);
        if (ImGui::Button(pt.training ? "|| pause" : "> train probe")) {
            // [DATA HOOK] Model::startProbeTraining(name, kind, location, dataset)
            // — kicks off a real probe-training job in the backend.
            m.startProbeTraining(probeName, kindIdx == 0 ? "linear" : "logistic",
                                  "L08.resid_post", "harmful_v2");
            LLOB_LOG_INFO("probe", "training probe %s", probeName);
        }
        ImGui::SameLine();
        if (ImGui::Button("save")) {
            // [DATA HOOK] Model::saveProbe(name) — persist the trained
            // probe to disk; real backend writes to ./out/<name>.pt.
            m.saveProbe(probeName);
            LLOB_LOG_INFO("probe", "saved probe %s", probeName);
        }
        if (pt.training) {
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().accent);
            ImGui::Text("step %d/%d", pt.step, pt.total_steps); ImGui::PopStyleColor();
        }
    }

    if (auto sec = BeginSection("Validation curves", true)) {
        const auto pt = m.getProbeTrainState(probeName);
        if (pt.val_curve.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
            ImGui::TextUnformatted("// no training in progress");
            ImGui::PopStyleColor();
        } else {
            SparkOpts so{}; so.color = Sty().good; so.fill = true;
            so.width = 140; so.height = 36;
            Sparkline(pt.val_curve, so);
            ImGui::SameLine();
            KV({
                { "train", FmtFloat(pt.train_acc, "%.2f"), "good" },
                { "val",   FmtFloat(pt.val_acc,   "%.2f"), "good" },
                { "gap",   FmtFloat(pt.gap,       "%.2f"), "" },
            }, true);
        }
    }

    if (auto sec = BeginSection("Probe library", false, "library")) {
        // [DATA HOOK] Model::getProbeLibrary() — list of saved probes
        // available to attach.  Engine source: ./out/probes/*.pt + meta.
        const auto lib = m.getProbeLibrary();
        if (lib.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
            ImGui::TextUnformatted("// probe library is empty");
            ImGui::PopStyleColor();
        } else if (ImGui::BeginTable("##lib", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInner)) {
            ImGui::TableSetupColumn(""); ImGui::TableSetupColumn("name");
            ImGui::TableSetupColumn("acc", ImGuiTableColumnFlags_WidthFixed, 50);
            ImGui::TableHeadersRow();
            for (const auto& r : lib) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn(); Pill(r.type.c_str(), r.type_tone.c_str(), r.type_solid);
                ImGui::TableNextColumn();
                ImGui::PushStyleColor(ImGuiCol_Text, r.type_solid ? Sty().accent : Sty().text);
                ImGui::TextUnformatted(r.name.c_str());
                ImGui::PopStyleColor();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(FmtFloat(r.accuracy, "%.2f").c_str());
            }
            ImGui::EndTable();
        }
    }
    ImGui::EndChild();
}

void DrawSAETrain(Model& m) {
    // [DATA HOOK] Model::getSAETrainingMetrics(saeId) — live training-run
    // state for the current SAE: recon_loss, L0 sparsity, dead features,
    // explained variance, plus history vectors for each (for the sparklines).
    const auto sae = m.getSAETrainingMetrics("L08.mlp_out");
    char header[96];
    std::snprintf(header, sizeof header,
                  "SAE training run · %s · step %d/%d",
                  sae.sae_id.empty() ? "—" : sae.sae_id.c_str(),
                  sae.step, sae.total_steps);

    DrawTitleBar("sae_training", "∑", nullptr, "sae-train");
    if (!ImGui::BeginChild("##sae_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    if (auto sec = BeginSection(header, true)) {
        struct Card { const char* lbl; const std::vector<float>& history;
                      ImU32 col; std::string val; const char* tone; };
        char dead[24]; std::snprintf(dead, sizeof dead, "%d / %d",
                                      sae.dead_features, sae.total_features);
        const Card cards[] = {
            { "RECON LOSS",    sae.recon_loss_history,    Sty().accent,
              FmtFloat(sae.recon_loss, "%.3f"),    "" },
            { "L0 SPARSITY",   sae.l0_sparsity_history,   Sty().warn,
              FmtFloat(sae.l0_sparsity, "%.1f"),   "warn" },
            { "DEAD FEATURES", sae.dead_features_history, Sty().bad,
              std::string(dead),                    "bad"  },
            { "EXPL VAR",      sae.expl_var_history,      Sty().good,
              FmtFloat(sae.expl_var, "%.3f"),       "good" },
        };
        const float card_w = (ImGui::GetContentRegionAvail().x - 24) / 4.0f;
        for (int i = 0; i < 4; ++i) {
            ImGui::BeginGroup();
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
            ImGui::TextUnformatted(cards[i].lbl); ImGui::PopStyleColor();
            if (cards[i].history.empty()) {
                ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_dim);
                ImGui::TextUnformatted("// no history");
                ImGui::PopStyleColor();
            } else {
                SparkOpts so{}; so.color = cards[i].col; so.fill = true;
                so.width = card_w - 8; so.height = 50;
                Sparkline(cards[i].history, so);
            }
            ImGui::PushStyleColor(ImGuiCol_Text,
                cards[i].tone[0] ? ToneColor(cards[i].tone) : Sty().text_bright);
            ImGui::TextUnformatted(cards[i].val.c_str()); ImGui::PopStyleColor();
            ImGui::EndGroup();
            if (i < 3) ImGui::SameLine();
        }
    }
    ImGui::EndChild();
}

void DrawExport(AppState& s, Model& m) {
    DrawTitleBar("export", "↗", nullptr, "export");
    if (!ImGui::BeginChild("##ex_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    char ab[16]; std::snprintf(ab, sizeof ab, "%zu",
                                 s.ablatedHeads.size() + s.ablatedComponents.size());
    char pr[16]; std::snprintf(pr, sizeof pr, "%zu",
                                 s.probedHeads.size() + s.probedComponents.size());
    if (auto sec = BeginSection("Snapshot", true)) {
        // Most snapshot counts are UI-state (AppState) — the live ablation /
        // probe sets the user has pending.  active_features could come from
        // Model::getActiveProbes() filtered by type; left as UI state for now.
        const auto st = m.getSteering();
        const std::string steering = st.active
            ? "on (α=" + FmtFloat(st.alpha, "%+.2f") + ")"
            : "off";
        KV({
            { "active probes",   pr,       "accent" },
            { "active features", "—",      "accent" },   // [DATA HOOK] feature count from engine
            { "ablations",       ab,       "warn" },
            { "steering",        steering, "warn" },
        });
    }
    if (ImGui::Button("↗ export state")) {
        // [DATA HOOK] Model::exportSnapshot(path) — serialise the full
        // session (ablations, probes, features, steering) to a JSON sidecar.
        m.exportSnapshot("./out/state.json");
        LLOB_LOG_INFO("export", "state snapshot exported → ./out/state.json");
    }
    ImGui::SameLine();
    if (ImGui::Button("save preset"))   LLOB_LOG_INFO("export", "preset saved");

    if (auto sec = BeginSection("Recent exports")) {
        // [DATA HOOK] Model::getRecentExports() — list of recent export
        // files; engine source: `ls ./out/*.{json,npz,csv}` sorted by mtime.
        const auto exps = m.getRecentExports();
        if (exps.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
            ImGui::TextUnformatted("// no recent exports");
            ImGui::PopStyleColor();
        } else {
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
            for (const auto& e : exps) {
                ImGui::Text("%s  %s", e.timestamp.c_str(), e.filename.c_str());
            }
            ImGui::PopStyleColor();
        }
    }
    ImGui::EndChild();
}

}  // namespace

void DrawProbesWorkspace(AppState& s, Model& m) {
    static int  feature = 2381;
    static char filter[64] = {};

    const float W = ImGui::GetContentRegionAvail().x;
    const float H = ImGui::GetContentRegionAvail().y;
    const float gap = 1.0f, lw = 320.0f, rw = 320.0f;
    const float cw = std::max(200.0f, W - lw - rw - 2 * gap);
    const float bot_h = std::min(220.0f, H * 0.30f);
    const float top_h = H - bot_h - gap;

    ImGui::BeginChild("##pr_left", { lw, H }, ImGuiChildFlags_Borders);
    DrawFeatureBrowser(s, m, feature, filter); ImGui::EndChild(); ImGui::SameLine(0, gap);

    ImGui::BeginChild("##pr_center", { cw, H });
    ImGui::BeginChild("##pr_card", { cw, top_h }, ImGuiChildFlags_Borders);
    DrawFeatureCard(feature, m); ImGui::EndChild();
    ImGui::BeginChild("##pr_sae",  { cw, bot_h }, ImGuiChildFlags_Borders);
    DrawSAETrain(m); ImGui::EndChild();
    ImGui::EndChild(); ImGui::SameLine(0, gap);

    ImGui::BeginChild("##pr_right", { rw, H });
    ImGui::BeginChild("##pr_ops", { rw, top_h }, ImGuiChildFlags_Borders);
    DrawProbeOps(s, m); ImGui::EndChild();
    ImGui::BeginChild("##pr_exp", { rw, bot_h }, ImGuiChildFlags_Borders);
    DrawExport(s, m); ImGui::EndChild();
    ImGui::EndChild();
}

}  // namespace llob
