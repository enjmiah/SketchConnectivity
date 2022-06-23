#pragma once

#include "types.h"

#include <polyclipping/clipper.hpp>

#include <memory>

namespace sketching {

// Clipper only works with integer coordinates, so we must convert.
constexpr auto clip_precision_factor = 1e8;

std::unique_ptr<ClipperLib::Paths> to_clip_paths(span<const CoordMat> polygons);

std::vector<CoordMat> from_clip_paths(const ClipperLib::Paths& paths);

CoordMat from_clip_path(const ClipperLib::Path& path);

std::unique_ptr<ClipperLib::Paths> boolean_union(const ClipperLib::Paths& polygons);

std::unique_ptr<ClipperLib::Paths> boolean_difference(const ClipperLib::Paths& a,
                                                      const ClipperLib::Paths& b);

std::unique_ptr<ClipperLib::Paths> boolean_intersection(const ClipperLib::Paths& a,
                                                        const ClipperLib::Paths& b);

/**
 * This is not the actual area, but relative comparisons should work.
 */
Float clip_area_scaled(const ClipperLib::Paths& paths);

inline Float clip_area_to_real_area(Float clip_area) {
  static constexpr double inv_precision_factor = 1 / clip_precision_factor;
  return clip_area * (inv_precision_factor * inv_precision_factor);
}

} // namespace sketching
