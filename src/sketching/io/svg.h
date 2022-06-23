#pragma once

#include "../types.h"

namespace sketching {

struct Stroke;

void save_svg(span<const Stroke> strokes, const std::string& path);

} // namespace sketching
