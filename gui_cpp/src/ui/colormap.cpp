#include "ui/colormap.hpp"

#include "style.hpp"

#include <cstring>

namespace llob {

ImU32 TextOverHeat(float v) {
    return v > 0.5f ? IM_COL32(0, 0, 0, 255) : Sty().text_bright;
}

ImU32 ToneColor(const char* tone) {
    if (!tone || !*tone) return Sty().text;
    if (std::strcmp(tone, "accent") == 0) return Sty().accent;
    if (std::strcmp(tone, "good")   == 0) return Sty().good;
    if (std::strcmp(tone, "warn")   == 0) return Sty().warn;
    if (std::strcmp(tone, "bad")    == 0) return Sty().bad;
    if (std::strcmp(tone, "info")   == 0) return Sty().info;
    if (std::strcmp(tone, "muted")  == 0) return Sty().text_muted;
    if (std::strcmp(tone, "dim")    == 0) return Sty().text_dim;
    if (std::strcmp(tone, "magenta")== 0) return Sty().magenta;
    if (std::strcmp(tone, "yellow") == 0) return Sty().yellow;
    return Sty().text;
}

}  // namespace llob
