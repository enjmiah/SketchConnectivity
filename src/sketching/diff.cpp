#include "diff.h"

#include "clipping.h"
#include "stroke_graph.h"

#include <polyclipping/clipper.hpp>

namespace sketching {

namespace cl = ClipperLib;

// TODO: Function is duplicated from graph_color.cpp, but I don't want to make the
//       clipping module depend on the stroke_graph module.  Clean this up somehow?
static ClipperLib::Paths face_polygon(const StrokeGraph::FaceView f) {
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

void changed_faces(const StrokeGraph& graph1, const StrokeGraph& graph2,
                   std::vector<size_t>& out_changed1, std::vector<size_t>& out_changed2) {
  out_changed1.clear();
  out_changed2.clear();
  const auto n_faces_1 = graph1.faces_.size();
  const auto n_faces_2 = graph2.faces_.size();

  // It's pointless to try and match slivers.
  constexpr auto sliver_area_thresh = 1e-6;

  // Create polygons to use for clipping.
  auto ref_polygons = std::vector<cl::Paths>(n_faces_1);
  for (size_t fi = 0; fi < n_faces_1; ++fi) {
    if (fi != graph1.boundary_face_)
      ref_polygons[fi] = face_polygon(graph1.face(fi));
  }
  auto new_polygons = std::vector<cl::Paths>(n_faces_2);
  for (size_t fi = 0; fi < n_faces_2; ++fi) {
    if (fi != graph2.boundary_face_)
      new_polygons[fi] = face_polygon(graph2.face(fi));
  }

  // Compute areas.
  auto face_areas_1 = std::vector<Float>(n_faces_1);
  for (size_t fi = 0; fi < n_faces_1; ++fi) {
    if (fi == graph1.boundary_face_) {
      face_areas_1[fi] = 0;
    } else {
      face_areas_1[fi] = clip_area_to_real_area(clip_area_scaled(ref_polygons[fi]));
    }
  }
  auto face_areas_2 = std::vector<Float>(n_faces_2);
  auto new_faces_by_area = std::vector<std::pair<int, Float>>();
  new_faces_by_area.reserve(n_faces_2 - 1);
  for (size_t fi = 0; fi < n_faces_2; ++fi) {
    if (fi == graph1.boundary_face_) {
      face_areas_2[fi] = 0;
    } else {
      const auto area = clip_area_to_real_area(clip_area_scaled(new_polygons[fi]));
      face_areas_2[fi] = area;
      new_faces_by_area.emplace_back((int)fi, area);
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
  auto new2ref = std::vector<int>(n_faces_2, -1);
  auto ref2new = std::vector<int>(n_faces_1, -1);
  new2ref[graph2.boundary_face_] = (int)graph1.boundary_face_;
  ref2new[graph1.boundary_face_] = (int)graph2.boundary_face_;

  // We do this by using the amount of overlap, starting with the biggest faces in
  // new_graph.  The matching is greedy.
  for (const auto& [fi2, new_area] : new_faces_by_area) {
    const auto& new_polygon = new_polygons[fi2];
    if (new_area < sliver_area_thresh)
      continue; // Pointless to try and match slivers.

    // Enforce a minimum overlap to get a match.
    auto best_overlap = 0.8;
    auto best_fi1 = -1;
    // Find the most similar face in ref_graph.
    for (size_t ref_fi = 0; ref_fi < n_faces_1; ++ref_fi) {
      if (ref2new[ref_fi] != -1)
        continue; // Ref face already matched.
      if (face_areas_1[ref_fi] < sliver_area_thresh)
        continue; // Pointless to try and match slivers.

      if (graph1.faces_[ref_fi].cycles_.size() != graph2.faces_[fi2].cycles_.size())
        continue; // Different genus; must be different.

      const auto max_overlap = std::min(face_areas_1[ref_fi], new_area) /
                               std::max(face_areas_1[ref_fi], new_area);
      if (max_overlap <= best_overlap)
        continue; // The best we can do is not enough.

      const auto& ref_polygon = ref_polygons[ref_fi];
      const auto intersection = boolean_intersection(ref_polygon, new_polygon);
      const auto overlap_area = clip_area_to_real_area(clip_area_scaled(*intersection));
      const auto overlap = overlap_area / std::max(face_areas_1[ref_fi], new_area);
      if (overlap > best_overlap) {
        best_overlap = overlap;
        best_fi1 = (int)ref_fi;
        break;
      }
    }
    if (best_fi1 != -1) {
      new2ref[fi2] = best_fi1;
      ref2new[best_fi1] = fi2;
    }
  }

  for (size_t fi = 0; fi < n_faces_1; ++fi) {
    if (ref2new[fi] == -1 && face_areas_1[fi] >= sliver_area_thresh) {
      out_changed1.push_back(fi);
    }
  }
  for (size_t fi = 0; fi < n_faces_2; ++fi) {
    if (new2ref[fi] == -1 && face_areas_2[fi] >= sliver_area_thresh) {
      out_changed2.push_back(fi);
    }
  }
}

} // namespace sketching
