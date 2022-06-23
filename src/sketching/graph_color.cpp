#include "graph_color.h"

#include "clipping.h"
#include "force_assert.h"
#include "stroke_graph.h"

#include <polyclipping/clipper.hpp>

#include <unordered_set>

namespace cl = ClipperLib;

namespace sketching {

namespace {

inline bool contains(const std::unordered_set<size_t>& s, const size_t v) {
  return s.find(v) != s.end();
}

// Maps face index to a collection of face indices.
using Connectivity = std::vector<std::unordered_set<size_t>>;

ClipperLib::Paths face_polygon(const StrokeGraph::FaceView f) {
  auto out = std::vector<std::vector<cl::IntPoint>>();

  for (const auto hi : f.cycles()) {
    auto& cycle = out.emplace_back();

    const auto he = f.graph_->hedge(hi);
    auto iteration = 0;
    auto it = he;
    do {
      const auto& s = it.stroke();
      if (it.forward()) {
        const auto stop = s.size() - 1;
        for (Index i = 0; i < stop; ++i) {
          cycle.emplace_back(cl::IntPoint(cl::cInt(clip_precision_factor * s.x(i)),
                                          cl::cInt(clip_precision_factor * s.y(i))));
        }
      } else {
        for (Index i = s.size() - 1; i > 0; --i) {
          cycle.emplace_back(cl::IntPoint(cl::cInt(clip_precision_factor * s.x(i)),
                                          cl::cInt(clip_precision_factor * s.y(i))));
        }
      }
      it = it.next();
      iteration++;
      assert(iteration < 1048576 && "likely infinite loop found");
    } while (it != he);
  }

  return out;
}

bool welsh_powell(const int max_colors, const Connectivity& connectivity,
                  const span<int> out_coloring) {
  const auto nf = out_coloring.size();
  auto face_indices_by_dec_deg = std::vector<size_t>();
  face_indices_by_dec_deg.reserve(nf);
  for (size_t fi = 0; fi < nf; ++fi)
    face_indices_by_dec_deg.push_back(fi);
  std::sort(
    face_indices_by_dec_deg.begin(), face_indices_by_dec_deg.end(),
    [&](size_t a, size_t b) { return connectivity[a].size() > connectivity[b].size(); });

  // Perform Welsh-Powell.
  std::fill(out_coloring.begin(), out_coloring.end(), -1);
  int current_color = 0;
  auto colored_this_round = std::unordered_set<size_t>();
  while (current_color < max_colors) {
    colored_this_round.clear();
    for (const auto fi : face_indices_by_dec_deg) {
      if (out_coloring[fi] == -1) {
        auto skip = false;
        for (const auto neigh : colored_this_round) {
          if (contains(connectivity[neigh], fi)) {
            skip = true;
            break;
          }
        }
        if (!skip) {
          out_coloring[fi] = current_color;
          colored_this_round.insert(fi);
        }
      }
    }
    if (colored_this_round.empty())
      break;
    current_color++;
  }

  if (current_color == max_colors) {
    // We failed to find a coloring (though one may exist).
    for (size_t fi = 0; fi < nf; ++fi) {
      if (out_coloring[fi] == -1) {
        out_coloring[fi] = max_colors - 1;
      }
    }
  }

  return current_color != max_colors;
}

void shuffle_colors(const int max_colors, const Connectivity& connectivity,
                    const span<int> inout_coloring) {
  // Welsh-Powell will try to use as few colors as possible, but we would like to use as
  // many as possible to make the regions easier to distinguish.  So we try to replace
  // each color with a random one that doesn't violate connectivity constraints.
  // If we started with a valid coloring, our coloring will remain valid after each
  // replacement.
  const auto nf = inout_coloring.size();
  auto color_can_be_assigned = std::vector<bool>(max_colors);
  auto valid_colors = std::vector<int>();
  valid_colors.reserve(max_colors);
  for (size_t fi = 0; fi < nf; ++fi) {
    for (int color = 0; color < max_colors; ++color)
      color_can_be_assigned[color] = true;
    valid_colors.clear();

    for (const auto neighbor : connectivity[fi]) {
      if (neighbor != fi) {
        const auto neighbor_color = inout_coloring[neighbor];
        color_can_be_assigned[neighbor_color] = false;
      }
    }

    for (int color = 0; color < max_colors; ++color) {
      if (color_can_be_assigned[color]) {
        valid_colors.push_back(color);
      }
    }

    if (!valid_colors.empty()) {
      const auto pick = (97 * fi) % valid_colors.size();
      inout_coloring[fi] = valid_colors[pick];
    }
  }
}

} // namespace

Connectivity compute_connectivity(const StrokeGraph& graph) {
  auto connectivity = Connectivity();
  const auto nf = graph.faces_.size();
  connectivity.resize(nf);
  for (size_t fi = 0; fi < nf; ++fi) {
    // For consistency, we'll say that every face is connected to itself.
    connectivity[fi].insert(fi);
    if (fi != graph.boundary_face_) {
      for (const auto hi : graph.faces_[fi].cycles_) {
        const auto he = graph.hedge(hi);
        auto it = he;
        auto iters = 0;
        do {
          auto it2 = it;
          do {
            const auto other = it2.twin().face_idx();
            if (other != graph.boundary_face_) {
              connectivity[fi].insert(other);
              connectivity[other].insert(fi);
            }
            iters++;
            force_assert(iters < 1024 && "likely infinite loop found");
            it2 = it2.twin().next();
          } while (it2 != it);
          it = it.next();
        } while (it != he);
      }
    }
  }
  return connectivity;
}

Connectivity compute_connectivity(const Image<const int32_t>& label_img,
                                  const int n_labels) {
  auto connectivity = Connectivity();
  connectivity.resize(n_labels);

  const auto w = label_img.width_;
  const auto h = label_img.height_;
  const auto* data = label_img.data_;
  for (Index y = 1; y < h - 1; ++y) {
    for (Index x = 1; x < w - 1; ++x) {
      const auto label = data[y * w + x];
      assert(label < n_labels);
      for (const auto neigh : {
             data[(y - 1) * w + x - 1], data[(y - 1) * w + x],
             data[(y - 1) * w + x + 1], //
             data[y * w + x - 1], data[y * w + x + 1], //
             data[(y + 1) * w + x - 1], data[(y + 1) * w + x],
             data[(y + 1) * w + x + 1], //
           }) {
        if (label != neigh) {
          connectivity[label].insert(neigh);
          connectivity[neigh].insert(label);
        }
      }
    }
  }

  // For consistency.
  for (Index i = 0; i < n_labels; ++i) {
    connectivity[i].insert(i);
  }

  return connectivity;
}

bool map_color(const StrokeGraph& graph, const int max_colors,
               const span<int> out_coloring) {
  const auto nf = graph.faces_.size();
  force_assert(out_coloring.size() == nf);
  if (nf == 0) {
    return true;
  }

  auto success = true;
  const auto connectivity = compute_connectivity(graph);
  if ((int)nf <= max_colors + 1) {
    // Simple case: each face can get its own color.  (Boundary face doesn't count.)
    auto current_color = 0;
    for (size_t i = 0; i < out_coloring.size(); ++i) {
      out_coloring[i] = (i == graph.boundary_face_ ? max_colors : current_color++);
    }
    success = true;
  } else {
    assert(connectivity[graph.boundary_face_].size() == 1 &&
           connectivity[graph.boundary_face_].count(graph.boundary_face_));
    for (size_t i = 0; i < connectivity.size(); ++i) {
      if (i != graph.boundary_face_) {
        assert(connectivity[i].count(graph.boundary_face_) == 0);
      }
    }
    success = welsh_powell(max_colors, connectivity, out_coloring);
    shuffle_colors(max_colors, connectivity, out_coloring);
  }
  out_coloring[graph.boundary_face_] = max_colors; // To detect logic errors downstream.

  return success;
}

bool map_color_raster(const Image<const int32_t> label_img, const int max_colors,
                      span<int> out_coloring) {
  const auto n_labels = (int)out_coloring.size();
  if (n_labels == 0) {
    return true;
  }
  const auto connectivity = compute_connectivity(label_img, n_labels);
  auto success = false;
  if (n_labels <= max_colors) {
    // Simple case: each label can get its own color.
    for (size_t i = 0; i < out_coloring.size(); ++i) {
      out_coloring[i] = (int)i;
    }
    success = true;
  } else {
    success = welsh_powell(max_colors, connectivity, out_coloring);
    shuffle_colors(max_colors, connectivity, out_coloring);
  }
  out_coloring[0] = max_colors; // To detect logic errors downstream.
  return success;
}

bool color_by_reference(const int max_colors, //
                        const StrokeGraph& ref_graph, const span<const int> ref_coloring,
                        const StrokeGraph& new_graph, const span<int> out_coloring) {
  const auto n_ref_faces = ref_graph.faces_.size();
  const auto n_new_faces = new_graph.faces_.size();
  force_assert(ref_coloring.size() == n_ref_faces);
  force_assert(out_coloring.size() == n_new_faces);
  // TODO: Should really check for *any* overlap.
  force_assert(ref_coloring.data() != out_coloring.data());

  // There's no face/color in the reference graph.
  if (ref_coloring.empty()) {
    return map_color(new_graph, max_colors, out_coloring);
  }

  //
  // Pre-computation.
  //

  // Get face connectivities for the graph color problem.
  const auto new_connectivity = compute_connectivity(new_graph);

  // Create polygons to use for clipping.
  auto ref_polygons = std::vector<cl::Paths>(n_ref_faces);
  for (size_t fi = 0; fi < n_ref_faces; ++fi) {
    if (fi != ref_graph.boundary_face_)
      ref_polygons[fi] = face_polygon(ref_graph.face(fi));
  }
  auto new_polygons = std::vector<cl::Paths>(n_new_faces);
  for (size_t fi = 0; fi < n_new_faces; ++fi) {
    if (fi != new_graph.boundary_face_)
      new_polygons[fi] = face_polygon(new_graph.face(fi));
  }

  // Compute areas.
  auto ref_face_areas = std::vector<Float>(n_ref_faces);
  for (size_t fi = 0; fi < n_ref_faces; ++fi) {
    if (fi == ref_graph.boundary_face_) {
      ref_face_areas[fi] = 0;
    } else {
      ref_face_areas[fi] = clip_area_to_real_area(clip_area_scaled(ref_polygons[fi]));
    }
  }
  auto new_faces_by_area = std::vector<std::pair<int, Float>>();
  new_faces_by_area.reserve(n_new_faces - 1);
  for (size_t fi = 0; fi < n_new_faces; ++fi) {
    if (fi != new_graph.boundary_face_) {
      new_faces_by_area.emplace_back(
        (int)fi, clip_area_to_real_area(clip_area_scaled(new_polygons[fi])));
    }
  }
  std::sort(new_faces_by_area.begin(), new_faces_by_area.end(),
            [](const std::pair<int, Float>& a, const std::pair<int, Float>& b) {
              return a.second > b.second;
            });

  //
  // Find correspondences between the faces.
  //

  // Maps face indices in new_graph to face indices in old_graph.
  auto new2ref = std::vector<int>(n_new_faces, -1);
  auto ref2new = std::vector<int>(n_ref_faces, -1);
  new2ref[new_graph.boundary_face_] = (int)ref_graph.boundary_face_;
  ref2new[ref_graph.boundary_face_] = (int)new_graph.boundary_face_;

  // We do this by using the amount of overlap, starting with the biggest faces in
  // new_graph.  The matching is greedy.
  for (const auto& [new_fi, new_area] : new_faces_by_area) {
    const auto& new_polygon = new_polygons[new_fi];

    // Enforce a minimum overlap of 20% to get a match.  This ensures that a large area
    // does not greedily take a negligibly overlapping region away from a smaller area
    // with good overlap.
    auto best_overlap = 0.2 * new_area;
    auto best_ref_fi = -1;
    // Find the most similar face in ref_graph.
    for (size_t ref_fi = 0; ref_fi < n_ref_faces; ++ref_fi) {
      if (ref2new[ref_fi] != -1)
        continue; // Ref face already matched.

      const auto max_overlap = std::min(ref_face_areas[ref_fi], new_area);
      if (max_overlap <= best_overlap)
        continue; // The best we can do is not enough.

      const auto& ref_polygon = ref_polygons[ref_fi];
      const auto intersection = boolean_intersection(ref_polygon, new_polygon);
      const auto overlap_area = clip_area_to_real_area(clip_area_scaled(*intersection));
      if (overlap_area > best_overlap) {
        best_overlap = overlap_area;
        best_ref_fi = (int)ref_fi;

        if (best_overlap >= 0.5 * new_area)
          break; // Faces are disjoint, so if we get a majority, it must be the best one.
      }
    }
    if (best_ref_fi != -1) {
      new2ref[new_fi] = best_ref_fi;
      ref2new[best_ref_fi] = new_fi;
    }
  }

  //
  // Assign colors based on correspondences.
  //

  std::fill(out_coloring.begin(), out_coloring.end(), -1);
  out_coloring[new_graph.boundary_face_] = max_colors;
  // First try to assign faces with correspondences.
  for (const auto& [new_fi, _] : new_faces_by_area) {
    assert(out_coloring[new_fi] == -1);
    const auto ref_fi = new2ref[new_fi];
    if (ref_fi == -1)
      continue;

    const auto proposed_color = ref_coloring[ref_fi];
    assert(proposed_color != -1);
    auto skip = false;
    for (const auto neigh : new_connectivity[new_fi]) {
      if (out_coloring[neigh] == proposed_color) {
        skip = true; // A neighbour has a color of proposed_color.
        break;
      }
    }
    if (!skip) {
      // No neighbours have a color of proposed_color; this is safe.
      out_coloring[new_fi] = proposed_color;
    }
  }

  {
    // Determine if we are in the simple case where we can assign each remaining face a
    // unique color.
    auto largest_color_used = -1;
    auto n_faces_left_uncolored = 0;
    for (size_t fi = 0; fi < n_new_faces; ++fi) {
      if (out_coloring[fi] == -1) {
        n_faces_left_uncolored++;
      } else if (fi != new_graph.boundary_face_) {
        largest_color_used = std::max(largest_color_used, out_coloring[fi]);
      }
    }
    if (largest_color_used + n_faces_left_uncolored < max_colors) {
      // It's safe to give each face its own color.
      for (size_t fi = 0; fi < n_new_faces; ++fi) {
        if (out_coloring[fi] == -1) {
          out_coloring[fi] = ++largest_color_used;
          assert(out_coloring[fi] < max_colors);
        }
      }
      return true;
    }
  }

  //
  // Try to assign any faces left unassigned using a Welsh-Powell-ish heuristic.
  //

  auto face_indices_by_dec_deg = std::vector<size_t>();
  face_indices_by_dec_deg.reserve(n_new_faces);
  for (size_t fi = 0; fi < n_new_faces; ++fi)
    face_indices_by_dec_deg.push_back(fi);
  std::sort(face_indices_by_dec_deg.begin(), face_indices_by_dec_deg.end(),
            [&](size_t a, size_t b) {
              return new_connectivity[a].size() > new_connectivity[b].size();
            });

  int current_color = 0;
  while (current_color < max_colors) {
    auto all_done = true;
    for (const auto fi : face_indices_by_dec_deg) {
      if (out_coloring[fi] == -1) {
        auto skip = false;
        for (const auto neigh : new_connectivity[fi]) {
          if (out_coloring[neigh] == current_color) {
            skip = true;
            break;
          }
        }
        if (!skip) {
          out_coloring[fi] = current_color;
        } else {
          all_done = false;
        }
      }
    }
    if (all_done)
      break;
    current_color++;
  }

  if (current_color == max_colors) {
    // We failed to find a coloring (though one may exist).
    for (size_t fi = 0; fi < n_new_faces; ++fi) {
      if (out_coloring[fi] == -1) {
        out_coloring[fi] = max_colors - 1;
      }
    }
  }

  // Welsh-Powell will try to use as few colors as possible, but we would like to use as
  // many as possible to make the regions easier to distinguish.  So we try to replace
  // each color without a correspondence with a random one that doesn't violate
  // connectivity constraints.  If we started with a valid coloring, our coloring will
  // remain valid after each replacement.
  auto color_can_be_assigned = std::vector<bool>(max_colors);
  auto valid_colors = std::vector<int>();
  valid_colors.reserve(max_colors);
  for (size_t fi = 0; fi < n_new_faces; ++fi) {
    // Only randomize faces without a correspondence.
    if (new2ref[fi] != -1)
      continue;

    for (int color = 0; color < max_colors; ++color)
      color_can_be_assigned[color] = true;
    valid_colors.clear();

    for (const auto neighbor : new_connectivity[fi]) {
      if (neighbor != fi && neighbor != ref_graph.boundary_face_) {
        const auto neighbor_color = out_coloring[neighbor];
        color_can_be_assigned[neighbor_color] = false;
      }
    }

    for (int color = 0; color < max_colors; ++color) {
      if (color_can_be_assigned[color]) {
        valid_colors.push_back(color);
      }
    }

    if (!valid_colors.empty()) {
      const auto pick = fi % valid_colors.size();
      out_coloring[fi] = valid_colors[pick];
    }
  }

  return current_color != max_colors;
}

static int
label_overlap_amount(const Image<const int32_t>& ref_img, const int32_t ref_label,
                     const Image<const int32_t>& new_img, const int32_t new_label) {
  const auto* const ref_data = ref_img.data_;
  const auto* const new_data = new_img.data_;
  const auto n = ref_img.width_ * ref_img.height_;
  auto overlap = 0;
  for (Index i = 0; i < n; ++i) {
    if (ref_data[i] == ref_label && new_data[i] == new_label) {
      overlap++;
    }
  }
  return overlap;
}

bool color_by_reference_raster( //
  const int max_colors, //
  const Image<const int32_t>& ref_label_img, const span<const int> ref_coloring, //
  const Image<const int32_t>& new_label_img, const span<int> out_coloring) {

  // There's no face/color in the reference.
  if (ref_coloring.empty()) {
    return map_color_raster(new_label_img, max_colors, out_coloring);
  }

  const auto n_ref_faces = ref_coloring.size();
  const auto n_new_faces = out_coloring.size();
  const auto max_label_ref =
    *std::max_element(ref_label_img.data_, ref_label_img.data_ + (ref_label_img.width_ *
                                                                  ref_label_img.height_));
  force_assert(max_label_ref + 1 == (int)n_ref_faces);
  const auto max_label_new =
    *std::max_element(new_label_img.data_, new_label_img.data_ + (new_label_img.width_ *
                                                                  new_label_img.height_));
  force_assert(max_label_new + 1 == (int)n_new_faces);
  // TODO: Should really check for *any* overlap.
  force_assert(ref_coloring.data() != out_coloring.data());

  //
  // Pre-computation.
  //

  constexpr auto boundary_label = 0;
  const auto new_connectivity = compute_connectivity(new_label_img, (int)n_new_faces);
  // Compute areas.
  force_assert(ref_label_img.height_ == new_label_img.height_);
  force_assert(ref_label_img.width_ == new_label_img.width_);
  const auto w = ref_label_img.width_;
  const auto h = ref_label_img.height_;

  auto ref_label_areas = std::vector<int>(n_ref_faces, 0);
  const auto* data = ref_label_img.data_;
  for (Index y = 0; y < h; ++y) {
    for (Index x = 0; x < w; ++x) {
      const auto label = data[y * w + x];
      ref_label_areas[label]++;
    }
  }
  ref_label_areas[boundary_label] = 0;

  auto new_label_areas = std::vector<int>(n_new_faces, 0);
  data = new_label_img.data_;
  for (Index y = 0; y < h; ++y) {
    for (Index x = 0; x < w; ++x) {
      const auto label = data[y * w + x];
      new_label_areas[label]++;
    }
  }
  new_label_areas[boundary_label] = 0;

  auto new_labels_by_area = std::vector<std::pair<int, int>>();
  new_labels_by_area.reserve(n_new_faces - 1);
  for (size_t la = 0; la < n_new_faces; ++la) {
    if (la != boundary_label) {
      new_labels_by_area.emplace_back((int)la, new_label_areas[la]);
    }
  }
  std::sort(new_labels_by_area.begin(), new_labels_by_area.end(),
            [](const std::pair<int, Float>& a, const std::pair<int, Float>& b) {
              return a.second > b.second;
            });

  //
  // Find correspondences between the regions.
  //

  auto new2ref = std::vector<int>(n_new_faces, -1);
  auto ref2new = std::vector<int>(n_ref_faces, -1);
  new2ref[boundary_label] = (int)boundary_label;
  ref2new[boundary_label] = (int)boundary_label;

  // We do this by using the amount of overlap, starting with the biggest faces in
  // new_graph.  The matching is greedy.
  for (const auto& [new_fi, new_area] : new_labels_by_area) {
    // Enforce a minimum overlap of 20% to get a match.  This ensures that a large area
    // does not greedily take a negligibly overlapping region away from a smaller area
    // with good overlap.
    auto best_overlap = 0.2 * new_area;
    auto best_ref_fi = -1;
    // Find the most similar region in ref.
    for (size_t ref_fi = 0; ref_fi < n_ref_faces; ++ref_fi) {
      if (ref2new[ref_fi] != -1)
        continue; // Ref label already matched.

      const auto max_overlap = std::min(ref_label_areas[ref_fi], new_area);
      if (max_overlap <= best_overlap)
        continue; // The best we can do is not enough.

      const auto overlap_area = label_overlap_amount(ref_label_img, (int32_t)ref_fi,
                                                     new_label_img, (int32_t)new_fi);
      if (overlap_area > best_overlap) {
        best_overlap = overlap_area;
        best_ref_fi = (int)ref_fi;

        if (best_overlap >= 0.5 * new_area) {
          // Regions are disjoint, so if we get a majority, it must be the best one.
          break;
        }
      }
    }
    if (best_ref_fi != -1) {
      new2ref[new_fi] = best_ref_fi;
      ref2new[best_ref_fi] = new_fi;
    }
  }

  //
  // Assign colors based on correspondences.
  //

  std::fill(out_coloring.begin(), out_coloring.end(), -1);
  out_coloring[boundary_label] = max_colors;
  // First try to assign faces with correspondences.
  for (const auto& [new_fi, _] : new_labels_by_area) {
    assert(out_coloring[new_fi] == -1);
    const auto ref_fi = new2ref[new_fi];
    if (ref_fi == -1)
      continue;

    const auto proposed_color = ref_coloring[ref_fi];
    assert(proposed_color != max_colors);
    assert(proposed_color != -1);
    auto skip = false;
    for (const auto neigh : new_connectivity[new_fi]) {
      if (out_coloring[neigh] == proposed_color) {
        skip = true; // A neighbour has a color of proposed_color.
        break;
      }
    }
    if (!skip) {
      // No neighbours have a color of proposed_color; this is safe.
      out_coloring[new_fi] = proposed_color;
    }
  }

  {
    // Determine if we are in the simple case where we can assign each remaining face a
    // unique color.
    auto largest_color_used = -1;
    auto n_faces_left_uncolored = 0;
    for (size_t fi = 0; fi < n_new_faces; ++fi) {
      if (out_coloring[fi] == -1) {
        n_faces_left_uncolored++;
      } else if (fi != boundary_label) {
        largest_color_used = std::max(largest_color_used, out_coloring[fi]);
      }
    }
    if (largest_color_used + n_faces_left_uncolored < max_colors) {
      // It's safe to give each face its own color.
      for (size_t fi = 0; fi < n_new_faces; ++fi) {
        if (out_coloring[fi] == -1) {
          out_coloring[fi] = ++largest_color_used;
          assert(out_coloring[fi] < max_colors);
        }
      }
      return true;
    }
  }

  //
  // Try to assign any faces left unassigned using a Welsh-Powell-ish heuristic.
  //

  auto face_indices_by_dec_deg = std::vector<size_t>();
  face_indices_by_dec_deg.reserve(n_new_faces);
  for (size_t fi = 0; fi < n_new_faces; ++fi)
    face_indices_by_dec_deg.push_back(fi);
  std::sort(face_indices_by_dec_deg.begin(), face_indices_by_dec_deg.end(),
            [&](size_t a, size_t b) {
              return new_connectivity[a].size() > new_connectivity[b].size();
            });

  int current_color = 0;
  while (current_color < max_colors) {
    auto all_done = true;
    for (const auto fi : face_indices_by_dec_deg) {
      if (out_coloring[fi] == -1) {
        auto skip = false;
        for (const auto neigh : new_connectivity[fi]) {
          if (out_coloring[neigh] == current_color) {
            skip = true;
            break;
          }
        }
        if (!skip) {
          out_coloring[fi] = current_color;
        } else {
          all_done = false;
        }
      }
    }
    if (all_done)
      break;
    current_color++;
  }

  if (current_color == max_colors) {
    // We failed to find a coloring (though one may exist).
    for (size_t fi = 0; fi < n_new_faces; ++fi) {
      if (out_coloring[fi] == -1) {
        out_coloring[fi] = max_colors - 1;
      }
    }
  }

  for (size_t i = 0; i < n_new_faces; ++i) {
    if (i != boundary_label)
      force_assert(out_coloring[i] < max_colors);
  }

  // Welsh-Powell will try to use as few colors as possible, but we would like to use as
  // many as possible to make the regions easier to distinguish.  So we try to replace
  // each color without a correspondence with a random one that doesn't violate
  // connectivity constraints.  If we started with a valid coloring, our coloring will
  // remain valid after each replacement.
  auto color_can_be_assigned = std::vector<bool>(max_colors);
  auto valid_colors = std::vector<int>();
  valid_colors.reserve(max_colors);
  for (size_t fi = 0; fi < n_new_faces; ++fi) {
    // Only randomize faces without a correspondence.
    if (new2ref[fi] != -1)
      continue;

    for (int color = 0; color < max_colors; ++color)
      color_can_be_assigned[color] = true;
    valid_colors.clear();

    for (const auto neighbor : new_connectivity[fi]) {
      if (neighbor != fi && neighbor != boundary_label) {
        const auto neighbor_color = out_coloring[neighbor];
        color_can_be_assigned[neighbor_color] = false;
      }
    }

    for (int color = 0; color < max_colors; ++color) {
      if (color_can_be_assigned[color]) {
        valid_colors.push_back(color);
      }
    }

    if (!valid_colors.empty()) {
      const auto pick = fi % valid_colors.size();
      out_coloring[fi] = valid_colors[pick];
    }
  }

  return current_color != max_colors;
}

void compute_correspondence(const StrokeGraph& ref_graph, const size_t n_ref_faces,
                            const StrokeGraph& new_graph, const size_t n_new_faces,
                            std::vector<int>& new2ref, std::vector<int>& ref2new) {
  //
  // Pre-computation.
  //

  // Get face connectivities for the graph color problem.
  const auto new_connectivity = compute_connectivity(new_graph);

  // Create polygons to use for clipping.
  auto ref_polygons = std::vector<cl::Paths>(n_ref_faces);
  for (size_t fi = 0; fi < n_ref_faces; ++fi) {
    if (fi != ref_graph.boundary_face_)
      ref_polygons[fi] = face_polygon(ref_graph.face(fi));
  }
  auto new_polygons = std::vector<cl::Paths>(n_new_faces);
  for (size_t fi = 0; fi < n_new_faces; ++fi) {
    if (fi != new_graph.boundary_face_)
      new_polygons[fi] = face_polygon(new_graph.face(fi));
  }

  // Compute areas.
  auto ref_face_areas = std::vector<Float>(n_ref_faces);
  for (size_t fi = 0; fi < n_ref_faces; ++fi) {
    if (fi == ref_graph.boundary_face_) {
      ref_face_areas[fi] = 0;
    } else {
      ref_face_areas[fi] = clip_area_to_real_area(clip_area_scaled(ref_polygons[fi]));
    }
  }
  auto new_faces_by_area = std::vector<std::pair<int, Float>>();
  new_faces_by_area.reserve(n_new_faces - 1);
  for (size_t fi = 0; fi < n_new_faces; ++fi) {
    if (fi != new_graph.boundary_face_) {
      new_faces_by_area.emplace_back(
        (int)fi, clip_area_to_real_area(clip_area_scaled(new_polygons[fi])));
    }
  }
  std::sort(new_faces_by_area.begin(), new_faces_by_area.end(),
            [](const std::pair<int, Float>& a, const std::pair<int, Float>& b) {
              return a.second > b.second;
            });

  //
  // Find correspondences between the faces.
  //

  // Maps face indices in new_graph to face indices in old_graph.
  new2ref = std::vector<int>(n_new_faces, -1);
  ref2new = std::vector<int>(n_ref_faces, -1);
  new2ref[new_graph.boundary_face_] = (int)ref_graph.boundary_face_;
  ref2new[ref_graph.boundary_face_] = (int)new_graph.boundary_face_;

  // We do this by using the amount of overlap, starting with the biggest faces in
  // new_graph.  The matching is greedy.
  for (const auto& [new_fi, new_area] : new_faces_by_area) {
    const auto& new_polygon = new_polygons[new_fi];

    // Enforce a minimum overlap of 20% to get a match.  This ensures that a large area
    // does not greedily take a negligibly overlapping region away from a smaller area
    // with good overlap.
    auto best_overlap = 0.2 * new_area;
    auto best_ref_fi = -1;
    // Find the most similar face in ref_graph.
    for (size_t ref_fi = 0; ref_fi < n_ref_faces; ++ref_fi) {
      if (ref2new[ref_fi] != -1)
        continue; // Ref face already matched.

      const auto max_overlap = std::min(ref_face_areas[ref_fi], new_area);
      if (max_overlap <= best_overlap)
        continue; // The best we can do is not enough.

      const auto& ref_polygon = ref_polygons[ref_fi];
      const auto intersection = boolean_intersection(ref_polygon, new_polygon);
      const auto overlap_area = clip_area_to_real_area(clip_area_scaled(*intersection));
      if (overlap_area > best_overlap) {
        best_overlap = overlap_area;
        best_ref_fi = (int)ref_fi;

        if (best_overlap >= 0.5 * new_area)
          break; // Faces are disjoint, so if we get a majority, it must be the best one.
      }
    }
    if (best_ref_fi != -1) {
      new2ref[new_fi] = best_ref_fi;
      ref2new[best_ref_fi] = new_fi;
    }
  }
}

void compute_correspondence_raster(const Image<const std::int32_t>& ref_label_img,
                                   const size_t n_ref_faces,
                                   const Image<const std::int32_t>& new_label_img,
                                   const size_t n_new_faces, std::vector<int>& new2ref,
                                   std::vector<int>& ref2new) {
  const auto max_label_ref =
    *std::max_element(ref_label_img.data_, ref_label_img.data_ + (ref_label_img.width_ *
                                                                  ref_label_img.height_));
  force_assert(max_label_ref + 1 == (int)n_ref_faces);
  const auto max_label_new =
    *std::max_element(new_label_img.data_, new_label_img.data_ + (new_label_img.width_ *
                                                                  new_label_img.height_));
  force_assert(max_label_new + 1 == (int)n_new_faces);
  // TODO: Should really check for *any* overlap.

  //
  // Pre-computation.
  //

  constexpr auto boundary_label = 0;
  const auto new_connectivity = compute_connectivity(new_label_img, (int)n_new_faces);
  // Compute areas.
  force_assert(ref_label_img.height_ == new_label_img.height_);
  force_assert(ref_label_img.width_ == new_label_img.width_);
  const auto w = ref_label_img.width_;
  const auto h = ref_label_img.height_;

  auto ref_label_areas = std::vector<int>(n_ref_faces, 0);
  const auto* data = ref_label_img.data_;
  for (Index y = 0; y < h; ++y) {
    for (Index x = 0; x < w; ++x) {
      const auto label = data[y * w + x];
      ref_label_areas[label]++;
    }
  }
  ref_label_areas[boundary_label] = 0;

  auto new_label_areas = std::vector<int>(n_new_faces, 0);
  data = new_label_img.data_;
  for (Index y = 0; y < h; ++y) {
    for (Index x = 0; x < w; ++x) {
      const auto label = data[y * w + x];
      new_label_areas[label]++;
    }
  }
  new_label_areas[boundary_label] = 0;

  auto new_labels_by_area = std::vector<std::pair<int, int>>();
  new_labels_by_area.reserve(n_new_faces - 1);
  for (size_t la = 0; la < n_new_faces; ++la) {
    if (la != boundary_label) {
      new_labels_by_area.emplace_back((int)la, new_label_areas[la]);
    }
  }
  std::sort(new_labels_by_area.begin(), new_labels_by_area.end(),
            [](const std::pair<int, Float>& a, const std::pair<int, Float>& b) {
              return a.second > b.second;
            });

  //
  // Find correspondences between the regions.
  //

  new2ref = std::vector<int>(n_new_faces, -1);
  ref2new = std::vector<int>(n_ref_faces, -1);
  new2ref[boundary_label] = (int)boundary_label;
  ref2new[boundary_label] = (int)boundary_label;

  // We do this by using the amount of overlap, starting with the biggest faces in
  // new_graph.  The matching is greedy.
  for (const auto& [new_fi, new_area] : new_labels_by_area) {
    // Enforce a minimum overlap of 20% to get a match.  This ensures that a large area
    // does not greedily take a negligibly overlapping region away from a smaller area
    // with good overlap.
    auto best_overlap = 0.2 * new_area;
    auto best_ref_fi = -1;
    // Find the most similar region in ref.
    for (size_t ref_fi = 0; ref_fi < n_ref_faces; ++ref_fi) {
      if (ref2new[ref_fi] != -1)
        continue; // Ref label already matched.

      const auto max_overlap = std::min(ref_label_areas[ref_fi], new_area);
      if (max_overlap <= best_overlap)
        continue; // The best we can do is not enough.

      const auto overlap_area = label_overlap_amount(ref_label_img, (int32_t)ref_fi,
                                                     new_label_img, (int32_t)new_fi);
      if (overlap_area > best_overlap) {
        best_overlap = overlap_area;
        best_ref_fi = (int)ref_fi;

        if (best_overlap >= 0.5 * new_area) {
          // Regions are disjoint, so if we get a majority, it must be the best one.
          break;
        }
      }
    }
    if (best_ref_fi != -1) {
      new2ref[new_fi] = best_ref_fi;
      ref2new[best_ref_fi] = new_fi;
    }
  }
}

int label_area(const Image<const std::int32_t>& label_img, const size_t fi,
               const size_t n_faces) {
  assert(fi < n_faces);
  constexpr auto boundary_label = 0;
  const auto w = label_img.width_;
  const auto h = label_img.height_;

  const auto* data = label_img.data_;
  auto label_areas = std::vector<int>(n_faces, 0);
  data = label_img.data_;
  for (Index y = 0; y < h; ++y) {
    for (Index x = 0; x < w; ++x) {
      const auto label = data[y * w + x];
      label_areas[label]++;
    }
  }
  label_areas[boundary_label] = 0;

  return label_areas[fi];
}

} // namespace sketching
