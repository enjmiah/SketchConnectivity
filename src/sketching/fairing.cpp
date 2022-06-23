#include "fairing.h"

#include "bvh.h"
#include "clipping.h"
#include "degrees.h"
#include "detail/alloca.h"
#include "detail/util.h"
#include "fitting.h"
#include "force_assert.h"
#include "intersect.h"
#include "render.h"
#include "resample.h"
#include "sketching.h"
#include "stroke_view.h"

namespace sketching {

void smooth_stroke_box3(Stroke& stroke) {
  const auto n = stroke.size();
  auto temp_buf = std::vector<Float>(n);
  for (auto* buf : {stroke.x_, stroke.y_}) {
    temp_buf[0] = buf[0];
    for (Index i = 1; i < n - 1; ++i) {
      temp_buf[i] = (buf[i - 1] + buf[i] + buf[i + 1]) / 3;
    }
    temp_buf[n - 1] = buf[n - 1];

    for (Index i = 0; i < n; ++i) {
      buf[i] = temp_buf[i];
    }
  }
}

static int64_t bitmap_area(const Bitmap& bitmap) {
  const auto n = bitmap.size();
  auto area = int64_t(0);
  for (Index i = 0; i < n; ++i) {
    if (bitmap.data()[i]) {
      area++;
    }
  }
  return area;
}

std::vector<Stroke> remove_strokes_visual(const span<const Stroke> strokes,
                                          const Float average_area_threshold_proportion,
                                          const Float stroke_area_threshold_proportion) {
  const auto old_n_cap_vertices = get_render_n_cap_vertices();
  const auto old_decimation_status = get_render_decimation();
  // Use high quality polygons to be accurate.
  set_render_n_cap_vertices(10);
  set_render_decimation(false);

  const auto bb = visual_bounds(strokes);

  auto average_pen_width = Float(0);
  auto average_pen_width_denom = Float(0);
  for (size_t si = 0; si < strokes.size(); ++si) {
    average_pen_width += strokes[si].pen_width() * strokes[si].length();
    average_pen_width_denom += strokes[si].length();
  }
  average_pen_width /= average_pen_width_denom;
  spdlog::debug("average_pen_width is {}", average_pen_width);
  spdlog::debug("average_pen_width/bb is {:.3f}",
                average_pen_width / std::max(bb.width(), bb.height()));
  const auto raster_bounds =
    std::min(int(12 * std::max(bb.width(), bb.height()) / average_pen_width), 2048);

  auto stroke_bitmaps = std::vector<Bitmap>();
  auto areas = std::vector<int64_t>();
  stroke_bitmaps.reserve(strokes.size());
  areas.reserve(strokes.size());
  for (size_t si = 0; si < strokes.size(); ++si) {
    auto& bitmap = stroke_bitmaps.emplace_back(
      rasterize({&strokes[si], 1}, raster_bounds, raster_bounds, bb));
    assert(bitmap.rows() == raster_bounds);
    assert(bitmap.cols() == raster_bounds);
    areas.push_back(bitmap_area(bitmap));
  }

  const auto scale = raster_bounds / std::max(bb.width(), bb.height());
  const auto pen_area = M_PI * square(0.5 * average_pen_width * scale);
  const auto threshold = pen_area;
  force_assert(threshold >= 1);

  auto shared_bitmap = Bitmap(raster_bounds, raster_bounds);
  int64_t total_area = 0;
  const auto n = shared_bitmap.size();
  assert(n == raster_bounds * raster_bounds);
  {
    shared_bitmap.fill(false);
    for (size_t si = 0; si < strokes.size(); ++si) {
      assert(shared_bitmap.size() == stroke_bitmaps[si].size());
      const auto* other_bitmap = stroke_bitmaps[si].data();
      for (Index i = 0; i < n; ++i) {
        if (other_bitmap[i])
          shared_bitmap.data()[i] = true;
      }
    }
    total_area = bitmap_area(shared_bitmap);
  }
  spdlog::debug("Total area: {:.3f}%", 100 * Float(total_area) / n);

  std::vector<Stroke> out_strokes;
  auto removed_stroke_indices = std::vector<size_t>();
  for (size_t si = 0; si < strokes.size(); ++si) {
    shared_bitmap.fill(false);
    for (size_t other_si = 0; other_si < strokes.size(); ++other_si) {
      if (si != other_si &&
          std::find(removed_stroke_indices.begin(), removed_stroke_indices.end(),
                    other_si) == removed_stroke_indices.end()) {
        assert(shared_bitmap.size() == stroke_bitmaps[other_si].size());
        const auto* other_bitmap = stroke_bitmaps[other_si].data();
        for (Index i = 0; i < n; ++i) {
          if (other_bitmap[i])
            shared_bitmap.data()[i] = true;
        }
      }
    }

    const auto area_without_si = bitmap_area(shared_bitmap);
    const auto loss = total_area - area_without_si;
    assert(loss <= areas[si]);
    assert(loss >= 0);
    if (Float(loss) <= average_area_threshold_proportion * threshold &&
        Float(loss) <= stroke_area_threshold_proportion * areas[si]) {
      spdlog::debug("Removing stroke {} with loss of {:.3f}x, {:.3f}", si,
                    loss / threshold, Float(loss) / areas[si]);
      removed_stroke_indices.push_back(si);
      total_area = area_without_si;
    } else {
      spdlog::debug("Stroke {}: Loss of {:.3f}x, {:.3f}", si, loss / threshold,
                    Float(loss) / areas[si]);
      out_strokes.emplace_back(strokes[si].clone());
    }
  }

  set_render_n_cap_vertices(old_n_cap_vertices);
  set_render_decimation(old_decimation_status);

  return out_strokes;
}

ConstStrokeView remove_hooks_visual(const Stroke& s,
                                    const Float area_threshold_proportion) {
  if (s.size() <= 2) {
    return ConstStrokeView(s);
  }

  std::vector<CoordMat> total_polygon_coords;
  outline_to_polygons(s, total_polygon_coords);
  const auto total_polygon = to_clip_paths(total_polygon_coords);
  const auto total_area = clip_area_scaled(*total_polygon);

  const auto area_threshold = area_threshold_proportion * total_area;

  std::vector<CoordMat> coords;
  Index slice_start = 0;
  Float last_carea = 0.0;
  for (auto i = 1; i < s.size(); ++i) {
    coords.clear();
    outline_to_polygons(ConstStrokeView(s, i, s.size()), coords);
    const auto clipped = to_clip_paths(coords);
    const auto diff = boolean_difference(*total_polygon, *clipped);
    const auto carea = clip_area_scaled(*diff);
    if (carea > area_threshold) {
      spdlog::debug(
        "Head: rejected area change of {:.3f}%, using {:.3f}% (removed {:d} verts)",
        100 * carea / total_area, 100 * last_carea / total_area, slice_start);
      break;
    }
    slice_start = i;
    last_carea = carea;
  }
  Index slice_end = s.size();
  last_carea = 0.0;
  for (auto j = s.size() - 1; j > slice_start + 1; --j) {
    coords.clear();
    outline_to_polygons(ConstStrokeView(s, slice_start, j), coords);
    const auto clipped = to_clip_paths(coords);
    const auto diff = boolean_difference(*total_polygon, *clipped);
    const auto carea = clip_area_scaled(*diff);
    if (carea > area_threshold) {
      spdlog::debug(
        "Tail: rejected area change of {:.3f}%, using {:.3f}% (removed {:d} verts)",
        100 * carea / total_area, 100 * last_carea / total_area, s.size() - slice_end);
      break;
    }
    last_carea = carea;
    slice_end = j;
  }
  return {s, slice_start, slice_end};
}

void dehook_strokes(const span<Stroke> strokes, const Float factor) {
  auto bvh = EnvelopeBVH(strokes);
  for (const auto& stroke : strokes) {
    stroke.ensure_arclengths();
  }
  for (size_t i = 0; i < strokes.size(); ++i) {
    auto& this_stroke = strokes[i];
    auto [begin, end] = dehooked_range(this_stroke, factor);

    // Remove hook only if it does not destroy an existing connection.
    if (end != this_stroke.size()) {
      for (size_t j = 0; j < strokes.size(); ++j) {
        Float env_dist;
        const auto [dist, arclen] =
          find_snap_point(this_stroke, /*head=*/false, bvh.nodes[j], &env_dist);
        if (dist < INFINITY) {
          // The endpoint is connected to something else, but it may still be possible to
          // remove the hook and still have it connect.
          const auto [new_dist, new_arclen] =
            find_snap_point(ConstStrokeView(this_stroke, 0, end), /*head=*/false,
                            bvh.nodes[j], &env_dist);
          if (!(new_dist < INFINITY) ||
              std::abs(new_arclen - arclen) > strokes[j].width_at(arclen)) {
            // Don't dehook this end; we would destroy an existing connection.
            end = this_stroke.size();
            break;
          }
        }
      }
    }
    if (end != this_stroke.size()) {
      // Try not to destroy almost-connections either.
      auto closest_stroke = size_t(-1);
      auto closest_env_dist = Float(INFINITY);
      for (size_t j = 0; j < strokes.size(); ++j) {
        Float this_env_dist;
        find_snap_point(this_stroke, /*head=*/false, bvh.nodes[j], &this_env_dist);
        const auto env_dist_before = this_env_dist;
        if (env_dist_before < closest_env_dist) {
          closest_stroke = j;
          closest_env_dist = env_dist_before;
        }
      }
      if (closest_stroke != size_t(-1)) {
        Float this_env_dist;
        find_snap_point(ConstStrokeView(this_stroke, 0, end),
                        /*head=*/false, bvh.nodes[closest_stroke], &this_env_dist);
        const auto env_dist_after = std::max(this_env_dist, 0.0);
        if (env_dist_after > 2 * std::max(closest_env_dist, 0.0)) {
          end = this_stroke.size();
        }
      }
    }

    // Trim one end before assessing the other.
    // This prevents us destroying a self-connection.
    this_stroke.trim(0, end);
    this_stroke.ensure_arclengths();
    // Very important: keep in mind the pointer to `this_stroke` in bvh.nodes: make sure
    // we are changing the same stroke.

    // Same logic for the head.
    if (begin != 0) {
      for (size_t j = 0; j < strokes.size(); ++j) {
        Float env_dist;
        const auto [dist, arclen] =
          find_snap_point(this_stroke, /*head=*/true, bvh.nodes[j], &env_dist);
        if (dist < INFINITY) {
          const auto [new_dist, new_arclen] =
            find_snap_point(ConstStrokeView(this_stroke, begin, this_stroke.size()),
                            /*head=*/true, bvh.nodes[j], &env_dist);
          if (!(new_dist < INFINITY) ||
              std::abs(new_arclen - arclen) > strokes[j].width_at(arclen)) {
            begin = 0;
            break;
          }
        }
      }
    }
    if (begin != 0) {
      // Try not to destroy almost-connections either.
      auto closest_stroke = size_t(-1);
      auto closest_env_dist = Float(INFINITY);
      for (size_t j = 0; j < strokes.size(); ++j) {
        Float this_env_dist;
        find_snap_point(this_stroke, /*head=*/true, bvh.nodes[j], &this_env_dist);
        const auto env_dist_before = this_env_dist;
        if (env_dist_before < closest_env_dist) {
          closest_stroke = j;
          closest_env_dist = env_dist_before;
        }
      }
      if (closest_stroke != size_t(-1)) {
        Float this_env_dist;
        find_snap_point(ConstStrokeView(this_stroke, begin, this_stroke.size()),
                        /*head=*/true, bvh.nodes[closest_stroke], &this_env_dist);
        const auto env_dist_after = std::max(this_env_dist, 0.0);
        if (env_dist_after > 2 * std::max(closest_env_dist, 0.0)) {
          begin = 0;
        }
      }
    }

    this_stroke.trim(begin, this_stroke.size());
    this_stroke.ensure_arclengths();
  }
}

std::pair<Index, Index> dehooked_range(const Stroke& stroke, const Float factor) {
  const auto n = stroke.size();
  if (n < 3) {
    return {0, n};
  }
  const auto pw = stroke.pen_width();
  auto corner = ALLOCA(bool, n);
  corners_and_hooks(stroke, 0.5 * factor * pw, corner);
  stroke.ensure_arclengths();

  auto first_corner = Index(0);
  for (Index i = 1; i < n; ++i) {
    if (corner[i]) {
      first_corner = i;
      break;
    }
  }
  assert(first_corner != 0);
  if (first_corner == n - 1) {
    return {0, n}; // Line is approximately straight.
  }

  auto second_corner = first_corner + 1;
  for (; second_corner < n; ++second_corner) {
    if (corner[second_corner]) {
      break;
    }
  }

  auto trim_start = Index(0);
  auto trim_end = n;

  const auto max_len_pw = 1.5 * pw * factor;
  constexpr auto max_len_fac = 0.5;

  if (stroke.arclength(first_corner) + 0.5 * stroke.width(0) <
      std::min(max_len_pw, max_len_fac * (stroke.arclength(second_corner) -
                                          stroke.arclength(first_corner)))) {
    const auto xy = stroke.xy(0);

    const auto p = stroke.xy(first_corner);
    const auto q = stroke.xy(second_corner);
    const Float nsq = (q - p).squaredNorm();
    const auto t = (xy - p).dot(q - p) / nsq;
    if (t > 0.0 && t < 1.0) {
      const Vec2 proj = lerp(p, q, t);
      if ((proj - xy).norm() <= factor * pw) { // Endpoint is close to own stroke.
        trim_start = first_corner;
      }
    }
  }

  auto last_corner = n - 1;
  for (Index i = n - 2; i >= 0; --i) {
    if (corner[i]) {
      last_corner = i;
      break;
    }
  }
  if (last_corner == trim_start) {
    return {trim_start, n};
  }

  auto second_last_corner = last_corner - 1;
  for (; second_last_corner >= 0; --second_last_corner) {
    if (corner[second_last_corner]) {
      break;
    }
  }

  if (stroke.length() - stroke.arclength(last_corner) + 0.5 * stroke.width(Back) <
      std::min(max_len_pw, max_len_fac * (stroke.arclength(last_corner) -
                                          stroke.arclength(second_last_corner)))) {
    const auto xy = stroke.xy(Back);

    const auto p = stroke.xy(second_last_corner);
    const auto q = stroke.xy(last_corner);
    const auto nsq = (q - p).squaredNorm();
    const auto t = (xy - p).dot(q - p) / nsq;
    if (t > 0.0 && t < 1.0) {
      const auto proj = lerp(p, q, t);
      if ((proj - xy).norm() <= factor * pw) { // Endpoint is close to own stroke.
        trim_end = last_corner + 1;
      }
    }
  }

  force_assert(trim_start < trim_end);
  return {trim_start, trim_end};
}

void cut_at_corners(span<const Stroke> strokes, std::vector<Stroke>& out_cut_strokes) {
  auto max_vertices = Index(0);
  for (const auto& stroke : strokes) {
    max_vertices = std::max(max_vertices, stroke.size());
  }
  out_cut_strokes.reserve(size_t(1.5 * strokes.size()));

  auto include = std::make_unique<bool[]>(max_vertices);
  auto arcl = std::make_unique<Float[]>(max_vertices);
  auto proj = std::make_unique<Float[]>(max_vertices);
  auto indices = std::vector<Index>();

  for (const auto& stroke : strokes) {
    ramer_douglas_peucker(stroke, 0.5 * stroke.pen_width(), &include[0]);
    indices.push_back(0);
    for (Index i = 1; i < stroke.size() - 1; ++i) {
      if (include[i]) {
        indices.push_back(i);
      }
    }
    indices.push_back(stroke.size() - 1);

    // A curve can have corners where the stroke "folds back" on itself, but RDP will not
    // detect these because it only uses projection distance.
    // So we run RDP on arclength vs projection for each RDP segment.
    stroke.ensure_arclengths();
    const auto n = indices.size(); // Keep fixed, we'll be appending to `indices` in the
                                   // loop.
    for (size_t i = 1; i < n; ++i) {
      const auto start = indices[i - 1];
      const auto end = indices[i];
      const auto size = end - start + 1;
      const auto ab = (stroke.xy(end) - stroke.xy(start)).normalized();
      for (Index j = 0; j < size; ++j) {
        arcl[j] = stroke.arclength(start + j);
        const auto ac = stroke.xy(start + j) - stroke.xy(start);
        proj[j] = ac.dot(ab);
      }
      ramer_douglas_peucker(&arcl[0], &proj[0], size, stroke.pen_width(), &include[0]);
      for (Index j = 1; j < size - 1; ++j) {
        if (include[j]) {
          indices.push_back(start + j);
        }
      }
    }
    std::sort(indices.begin(), indices.end());

    auto start = Index(0);
    for (size_t i = 1; i < indices.size() - 1; ++i) {
      auto u = stroke.xy(indices[i]) - stroke.xy(indices[i - 1]);
      u.normalize();
      auto v = stroke.xy(indices[i + 1]) - stroke.xy(indices[i]);
      v.normalize();
      const auto angle = std::acos(std::clamp(u.dot(v), -1.0, 1.0));
      if (angle >= 90_deg) {
        out_cut_strokes.emplace_back(
          ConstStrokeView(stroke, start, indices[i] + 1).slice());
        start = indices[i];
      }
    }
    out_cut_strokes.emplace_back(ConstStrokeView(stroke, start, stroke.size()).slice());
    indices.clear();
  }
}

} // namespace sketching
