#pragma once

#include "stroke_graph.h"

namespace sketching {
bool is_face_collapsible_clipping(const StrokeGraph& stroke_graph,
                                  const size_t face_index, Float collapsing_threshold,
                                  Float stroke_width_scale);
} // namespace sketching
