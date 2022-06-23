#include "is_face_collapsible.h"

#include "bvh.h"
#include "clipping.h"
#include "intersect.h"
#include "render.h"

namespace sketching {

bool is_face_collapsible_clipping(const StrokeGraph& stroke_graph,
                                  const size_t face_index, Float collapsing_threshold,
                                  Float stroke_width_scale) {
  auto f = stroke_graph.face(face_index);

  // 1. Determine the boundary box of f
  std::unordered_set<size_t> stroke_indices;
  std::vector<size_t> empty_area_face;
  constexpr auto infinity_ = std::numeric_limits<Float>::infinity();
  for (const auto hi : f.cycles()) {
    const auto he = stroke_graph.hedge(hi);
    // For each boundary loop
    auto it = he;
    do {
      stroke_indices.emplace(it.stroke_idx());
      if (empty_area_face.empty() || empty_area_face.back() != it.stroke_idx())
        empty_area_face.emplace_back(it.stroke_idx());
      else
        empty_area_face.pop_back();

      it = it.next();
    } while (it != he);
  }

  // 1.5 Check if this face has no area (this is the case when it's a cut inside another
  // face). This face has no area but still we don't need/want to collapse it
  if (empty_area_face.empty())
    return false;

  // 2. Clip to only include the uncovered part
  std::vector<Vec2> poly = f.graph_->face_positions(f.index_);
  CoordMat poly_coords;
  poly_coords.resize(poly.size(), 2);
  for (size_t i = 0; i < poly.size(); ++i) {
    poly_coords.row(i) << poly[i].x(), poly[i].y();
  }

  auto face_polygon = to_clip_paths(span<const CoordMat>{&poly_coords, 1});
  Float check_area = clip_area_scaled(*face_polygon);

  // Rare case of having a single stroke in the cycle and it overlaps itself perfectly
  if (check_area < std::numeric_limits<Float>::epsilon())
    return false;

  std::vector<CoordMat> coords;
  for (size_t s_idx : stroke_indices) {
    auto stroke = stroke_graph.strokes_[s_idx].clone();
    for (Index i = 0; i < stroke.size(); ++i) {
      stroke.width(i) *= stroke_width_scale;
    }
    if (stroke.size() > 1) {
      outline_to_polygons(stroke, coords);
    }
  }
  const auto clipped = to_clip_paths(coords);
  auto diff = boolean_difference(*face_polygon, *clipped);
  Float visible_area = clip_area_scaled(*diff);

  // SPDLOG_INFO("Area: {} vs {}, {}", face_area, check_area, visible_area);

  if (check_area < std::numeric_limits<Float>::epsilon())
    check_area = 1.0;
  Float visible_ratio = visible_area / check_area;
  return visible_ratio < collapsing_threshold;
}

} // namespace sketching
