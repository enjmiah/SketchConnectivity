#pragma once

#include "../types.h"

namespace sketching {

struct Stroke;

/**
 * Determine the index of the first or last vertex (depending on `head`) to not be
 * included in the deformation range.
 *
 * @param head True if and only if the head of the stroke is to be deformed.
 */
size_t determine_deformation_substroke(const Stroke& stroke2, bool head);

#ifdef HAS_GUROBI
bool snap_endpoints(const Stroke& stroke, Vec2 head_pos, Vec2 tail_pos,
                    Stroke& stroke_snapped, bool use_default_sampling = false);

bool snap_endpoints_divided(Stroke& stroke, Vec2 head_pos, Vec2 tail_pos,
                            bool use_default_sampling);

bool snap_endpoints_adaptive(Stroke& stroke, Vec2 head_pos, Vec2 tail_pos,
                             bool use_default_sampling, int& deformation_size,
                             bool& forward);
#endif

} // namespace sketching
