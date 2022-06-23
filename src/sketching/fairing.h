#pragma once

#include "stroke_view.h"
#include "types.h"

namespace sketching {

struct Stroke;

void smooth_stroke_box3(Stroke& stroke);

std::vector<Stroke> remove_strokes_visual(span<const Stroke> strokes,
                                          Float average_area_threshold_proportion,
                                          Float stroke_area_threshold_proportion);

/**
 * Remove hooks based on difference in proportion of stroke envelope area.
 *
 * `area_threshold_proportion` should be in the range [0, 1).  An
 * `area_threshold_proportion` of p/100 means to only remove vertices that contribute less
 * than p% of the total area.
 */
ConstStrokeView remove_hooks_visual(const Stroke& s, Float area_threshold_proportion);

/**
 * Remove hooks from a collection of strokes.
 *
 * The hook removal criteria depends on a stroke's relation with other strokes, thus
 * calling this function on each stroke individually is not equivalent to calling this
 * function on all the strokes together.
 *
 * `factor` determines how aggressive to be with dehooking.
 */
void dehook_strokes(span<Stroke> strokes, Float factor = 1);

/**
 * Heuristically determine the non-hook range of the input stroke.
 *
 * You can actually trim the hook like so:
 *
 *     auto [start, end] = dehooked_range(stroke);
 *     stroke.trim(start, end);
 *
 * `factor` determines how aggressive to be with dehooking.
 */
[[nodiscard]] std::pair<Index, Index> dehooked_range(const Stroke& stroke,
                                                     Float factor = 1);

void cut_at_corners(span<const Stroke> strokes, std::vector<Stroke>& out_cut_strokes);

} // namespace sketching
