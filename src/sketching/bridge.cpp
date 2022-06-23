#include "bridge.h"

#include "clipping.h"
#include "detail/suppress_warning.h"
#include "eigen_compat.h"
#include "force_assert.h"
#include "global_solve/incremental_decomposition.h"
#include "global_solve/incremental_obj.h"
#include "global_solve/incremental_param.h"
#include "global_solve/incremental_region_util.h"
#include "global_solve/incremental_util.h"
#include "intersect.h"
#include "render.h"
#include "stroke_graph_extra.h"

#include <cfloat>

namespace sketching {

#define ARRAY_SIZE(arr) (sizeof((arr)) / sizeof((arr)[0]))

// Note we don't have a round 0, so entry 0 is just filler.
static constexpr Float face_size_ratio_buf[] = //
  {0.0, 20.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0};
static constexpr Float prob_threshold_buf[] = //
  {0.0, 0.0, 0.45, 0.40, 0.35, 0.3, 0.25, 0.2, 0.15, 0.10, 0.05, 0.00};

static const span<const Float> face_size_ratio = {face_size_ratio_buf,
                                                  ARRAY_SIZE(face_size_ratio_buf)};
static const span<const Float> prob_threshold = {prob_threshold_buf,
                                                 ARRAY_SIZE(prob_threshold_buf)};

static bool pt_inside_polygon(const Vec2 p, const CoordMat& polygon) {
  const auto n = polygon.rows();
  auto currentP = from_eigen(polygon.row(0));
  auto u1 = currentP.x_ - p.x_;
  auto v1 = currentP.y_ - p.y_;
  auto k = 0;
  for (Index i = 0; i < n; ++i) {
    // Find the next point on the cycle.
    const auto next_i = (i + 1) % n;
    const auto nextP = from_eigen(polygon.row(next_i));

    const auto v2 = nextP.y_ - p.y_;

    if ((v1 < 0 && v2 < 0) || (v1 > 0 && v2 > 0)) {
      currentP = nextP;
      v1 = v2;
      u1 = currentP.x_ - p.x_;
      continue;
    }

    const auto u2 = nextP.x_ - p.x_;

    if (v2 > 0 && v1 <= 0) {
      const auto f = (u1 * v2) - (u2 * v1);
      if (f > 0.0) {
        k++;
      } else if (f == 0) {
        return true; // On boundary.
      }
    } else if (v1 > 0 && v2 <= 0) {
      const auto f = (u1 * v2) - (u2 * v1);
      if (f < 0.0) {
        k++;
      } else if (f == 0) {
        return true; // On boundary.
      }
    } else if (v2 == 0 && v1 < 0) {
      const auto f = (u1 * v2) - (u2 * v1);
      if (f == 0.0) {
        return true; // On boundary.
      }
    } else if (v1 == 0 && v2 < 0) {
      const auto f = u1 * v2 - u2 * v1;
      if (f == 0.0) {
        return true; // On boundary.
      }
    } else if (v1 == 0 && v2 == 0) {
      if ((u2 <= 0 && u1 >= 0) || (u1 <= 0 && u2 >= 0)) {
        return true; // On boundary.
      }
    }
    currentP = nextP;
    v1 = v2;
    u1 = u2;
  }

  return k % 2 != 0;
}

/** Uses the shoelace formula. */
static Float polygon_area(const CoordMat& polygon) {
  const auto n = polygon.rows();
  assert(!polygon.row(0).isApprox(polygon.row(n - 1), /*prec=*/0.0));
  auto acc = 0.0;
  for (Index i = 0; i < n; ++i) {
    const auto next = (i + 1) % n;
    acc += (polygon(i, 0) * polygon(next, 1) - polygon(next, 0) * polygon(i, 1));
  }
  return 0.5 * acc;
}

/** Must be non-intersecting. */
static Vec2 polygon_centroid(const CoordMat& polygon) {
  const auto n = polygon.rows();
  assert(!polygon.row(0).isApprox(polygon.row(n - 1), /*prec=*/0.0));
  auto acc = Vec2(0, 0);
  for (Index i = 0; i < n; ++i) {
    const auto next = (i + 1) % n;
    acc.x_ += (polygon(i, 0) + polygon(next, 0)) *
              (polygon(i, 0) * polygon(next, 1) - polygon(next, 0) * polygon(i, 1));
    acc.y_ += (polygon(i, 1) + polygon(next, 1)) *
              (polygon(i, 0) * polygon(next, 1) - polygon(next, 0) * polygon(i, 1));
  }
  return (1.0 / (6 * polygon_area(polygon))) * acc;
}

BridgeInfo find_bridge_locations(const StrokeGraph& graph) {
  const auto bvh = EnvelopeBVH(graph.orig_strokes_);
  const auto n_strokes = bvh.nodes.size();
  auto centerline_bbs = std::vector<BoundingBox>();
  centerline_bbs.reserve(n_strokes);
  for (size_t si = 0; si < n_strokes; ++si) {
    const auto& stroke = *bvh.nodes[si].geometry;
    centerline_bbs.emplace_back(bounds(stroke));
  }
  auto envelopes = std::vector<ClipperLib::Paths>();
  envelopes.reserve(bvh.nodes.size());

  const auto old_decimation_status = get_render_decimation();
  // Use high quality polygons to be accurate.
  set_render_decimation(false);
  {
    auto polygon_buffer = std::vector<CoordMat>();
    for (size_t si = 0; si < n_strokes; ++si) {
      const auto& stroke = *bvh.nodes[si].geometry;
      outline_to_polygons(stroke, polygon_buffer);
      const auto clip_path = to_clip_paths(polygon_buffer);
      auto clean_clip_path = boolean_union(*clip_path);
      envelopes.emplace_back(std::move(*clean_clip_path));
      polygon_buffer.clear();
    }
    assert(envelopes.size() == n_strokes);
  }
  set_render_decimation(old_decimation_status);

  auto info = BridgeInfo();

  for (const auto& clip_path : envelopes) {
    info.envelopes_visual.emplace_back(from_clip_paths(clip_path));
  }

  for (size_t si = 0; si < n_strokes; ++si) {
    const auto& stroke = *bvh.nodes[si].geometry;
    stroke.ensure_arclengths();
  }

  auto intersections = std::vector<Vec2>();
  for (size_t si = 0; si < n_strokes; ++si) {
    const auto& stroke1 = *bvh.nodes[si].geometry;
    for (size_t sj = si + 1; sj < n_strokes; ++sj) {
      const auto& stroke2 = *bvh.nodes[sj].geometry;
      if (bvh.nodes[si].bb.touches(bvh.nodes[sj].bb)) {
        const auto clip_paths = boolean_intersection(envelopes[si], envelopes[sj]);
        if (clip_paths->empty()) {
          continue;
        }

        for (const auto& region : *clip_paths) {
          assert(!region.empty());
          auto coords = from_clip_path(region);
          if (polygon_area(coords) <= 1e-6) {
            // These tend to cause a numerical mess.
            continue;
          }
          auto loop_as_stroke = Stroke(coords.rows(), /*has_time=*/false);
          for (Index i = 0; i < coords.rows(); ++i) {
            loop_as_stroke.x(i) = coords(i, 0);
            loop_as_stroke.y(i) = coords(i, 1);
            loop_as_stroke.width(i) = 0.0;
          }
          auto needs_bridge = true;

          // Find centerline intersections.
          assert(intersections.empty());
          intersect_different({*bvh.nodes[si].geometry, centerline_bbs[si]},
                              {*bvh.nodes[sj].geometry, centerline_bbs[sj]},
                              intersections);
          // If there is a centerline intersection inside the region, then this connection
          // has already been found.
          for (const auto arclens : intersections) {
            const auto p = stroke1.pos(arclens.x_);
            if (pt_inside_polygon(p, coords)) {
              needs_bridge = false;
              break;
            }
          }

          if (needs_bridge) {
            // Determine if endpoint overlaps with intersection region.
            // If so, we already found this connection.
            const auto endpoints = std::initializer_list<std::pair<size_t, Index>>{
              {si, 0},
              {si, stroke1.size() - 1},
              {sj, 0},
              {sj, stroke2.size() - 1},
            };
            for (const auto& [endpoint_si, p_i] : endpoints) {
              auto endpoint_already_connected = false;
              auto end = StrokeTime((int)endpoint_si, p_i == 0 ? 0.0 : 1.0);
              if (convert_orig2strokes(graph, end)) {
                force_assert(end.second == 0.0 || end.second == 1.0);
                const auto other_si = (endpoint_si == si ? sj : si);
                const auto v =
                  endpoint_to_vertex(graph, Endpoint(end.first, end.second == 0.0));
                const auto he = v.hedge();
                auto it = he;
                auto valence = 0;
                do {
                  auto other_end =
                    StrokeTime((int)it.stroke_idx(), it.forward() ? 0.0 : 1.0);
                  if (convert_strokes2orig(graph, other_end) &&
                      other_end.first == (int)other_si) {
                    endpoint_already_connected = true;
                    break;
                  }
                  it = it.twin().next();
                  valence++;
                  force_assert(valence < 1024 && "likely infinite loop found");
                } while (it != he);
              }

              if (endpoint_already_connected) {
                const auto& stroke = *bvh.nodes[endpoint_si].geometry;
                const auto p = stroke.xy(p_i);
                if (pt_inside_polygon(p, coords)) {
                  needs_bridge = false;
                  break;
                }
                Vec2 proj;
                Float s = NAN;
                const auto dist_to_polygon = closest_point(loop_as_stroke, p, proj, s);
                // Necessary because of precision loss when converting to Clipper integer
                // coordinates.
                constexpr auto fudge = 2.0 / clip_precision_factor;
                if (dist_to_polygon <= 0.5 * stroke.width(p_i) + fudge) {
                  needs_bridge = false;
                  break;
                }
              }
            }
          }

          if (needs_bridge) {
            const auto centroid = polygon_centroid(coords);
            Float arclen1 = NAN, arclen2 = NAN;
            Vec2 proj;
            closest_point(stroke1, centroid, proj, arclen1);
            closest_point(stroke2, centroid, proj, arclen2);
            auto& junc = info.bridges.emplace_back(JunctionType::X); // I dunno...
            junc.points.emplace_back((int)si, arclen1 / stroke1.length());
            junc.points.emplace_back((int)sj, arclen2 / stroke2.length());
          }

          (needs_bridge ? info.bridge_intersections_visual
                        : info.other_intersections_visual)
            .emplace_back(std::move(coords));

          intersections.clear();
        }
      }
    }
  }

  return info;
}

static StrokeGraph::VertexView add_vertex_precise_safe(StrokeGraph& graph,
                                                       const size_t si, Float arclen) {
  // Avoid creating length 0 edges.
  constexpr auto eps = 1e-6;
  if (arclen <= eps) {
    arclen = 0.0;
  } else if (arclen >= graph.strokes_[si].length() - eps) {
    arclen = graph.strokes_[si].length();
  }

  StrokeGraph::VertexView v;
  add_vertex_precise(graph, si, arclen, v);
  return v;
}

void augment_with_bridges(StrokeGraph& graph, const BridgeInfo& bridges) {

  // FIXME: Where is the arclen getting out of date?
  for (const auto& stroke : graph.strokes_) {
    if (stroke.size() > 0) {
      stroke.compute_arclengths();
    }
  }

  // FIXME: bridge_vertices should just update the faces properly; why doesn't it work?
  graph.faces_.clear();
  for (auto& edge_rec : graph.hedges_) {
    edge_rec.face_ = StrokeGraph::invalid;
  }

  for (const auto& bridge : bridges.bridges) {
    //
    // Project connection location to edges.
    //

    const auto orig_si1 = bridge.points[0].first;
    const auto orig_si2 = bridge.points[1].first;
    const auto orig_p = graph.orig_strokes_[orig_si1].pos_norm(bridge.points[0].second);
    const auto orig_q = graph.orig_strokes_[orig_si2].pos_norm(bridge.points[1].second);

    auto location1 = StrokeTime(-1, -1.0);
    auto best_dist = Float(INFINITY);
    for (const auto& [edge_idx, map] : graph.orig2strokes_[orig_si1]) {
      Vec2 proj;
      Float arclen = NAN;
      const auto dist = closest_point(graph.strokes_[edge_idx], orig_p, proj, arclen);
      if (dist < best_dist) {
        best_dist = dist;
        location1 = StrokeTime((int)edge_idx, arclen / graph.strokes_[edge_idx].length());
      }
    }

    auto location2 = StrokeTime(-1, -1.0);
    best_dist = Float(INFINITY);
    for (const auto& [edge_idx, map] : graph.orig2strokes_[orig_si2]) {
      Vec2 proj;
      Float arclen = NAN;
      const auto dist = closest_point(graph.strokes_[edge_idx], orig_q, proj, arclen);
      if (dist < best_dist) {
        best_dist = dist;
        location2 = StrokeTime((int)edge_idx, arclen / graph.strokes_[edge_idx].length());
      }
    }

    if (location1.first >= 0 && location2.first >= 0) {
      //
      // Make the connection.
      //

      const auto p = graph.strokes_[location1.first].pos_norm(location1.second);
      const auto q = graph.strokes_[location2.first].pos_norm(location2.second);

      // If this assumption is violated, we need to be more careful about the order in
      // which we add the vertices corresponding to location1 and location2, due to stroke
      // locations becoming invalid after cutting. We would need to first map location2 to
      // original indexing, add the vertex corresponding to location1, then map location2
      // back to stroke indexing, then add the vertex corresponding to location2.
      // This shouldn't happen right now because we do not find bridges between an input
      // stroke and itself.
      force_assert(location1.first != location2.first);

      // Collection of (hit location, vertex index).
      auto connection_points = std::vector<std::pair<Float, size_t>>();
      connection_points.emplace_back(
        0.0, add_vertex_precise_safe(graph, location1.first,
                                     location1.second *
                                       graph.strokes_[location1.first].length())
               .index_);
      connection_points.emplace_back(
        1.0, add_vertex_precise_safe(graph, location2.first,
                                     location2.second *
                                       graph.strokes_[location2.first].length())
               .index_);
      // Store n_strokes so we do not intersect against edges we added for this bridge.
      const auto n_strokes = graph.strokes_.size();
      for (size_t i = 0; i < n_strokes; ++i) {
        auto hits = std::vector<Vec2>();
        intersect_segment_stroke_exclusive(p, q, graph.bvh_.polyline_bvh_leaf(i), hits);
        if (hits.size() == 1) {
          // If we only have one hit, then avoid the mapping complexity and just add the
          // vertex.  This is actually necessary, because for a bridge edge, the mapping
          // back to original stroke indexing will fail.  However, since bridge edges are
          // straight, we are guaranteed to only hit the same bridge edge at most once.
          connection_points.emplace_back(
            hits[0].x_, add_vertex_precise_safe(graph, i, hits[0].y_).index_);
        } else {
          auto insertion_points = std::vector<StrokeTime>();
          for (const auto& hit : hits) {
            insertion_points.emplace_back((int)i, hit.y_ / graph.strokes_[i].length());
            const auto ok = convert_strokes2orig(graph, insertion_points.back());
            force_assert(ok && "couldn't map from strokes to orig");
          }
          for (size_t k = 0; k < insertion_points.size(); ++k) {
            auto st = insertion_points[k];
            const auto ok = convert_orig2strokes(graph, st);
            force_assert(ok && "couldn't map from strokes to orig");
            const auto arclen = st.second * graph.strokes_[st.first].length();
            connection_points.emplace_back(
              hits[k].x_, add_vertex_precise_safe(graph, st.first, arclen).index_);
          }
        }
      }
      std::sort(connection_points.begin(), connection_points.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });

      for (size_t i = 0; i + 1 < connection_points.size(); ++i) {
        const auto vi1 = connection_points[i].second;
        const auto vi2 = connection_points[i + 1].second;
        if (vi1 != vi2) {
          bridge_vertices(graph, vi1, vi2);
        }
      }
    }
  }

  construct_faces(graph);
}

static StrokeTime orig2strokes_position_preserving(const StrokeGraph& graph,
                                                   const StrokeTime& orig_loc) {
  auto loc = StrokeTime(-1, -1.0);
  auto best_dist = Float(INFINITY);
  const auto p = graph.orig_strokes_[orig_loc.first].pos_norm(orig_loc.second);
  for (const auto& [edge_idx, map] : graph.orig2strokes_[orig_loc.first]) {
    Vec2 proj;
    Float arclen = 0;
    graph.strokes_[edge_idx].ensure_arclengths();
    const auto dist = closest_point(graph.strokes_[edge_idx], p, proj, arclen);
    if (dist < best_dist) {
      best_dist = dist;
      loc = StrokeTime((int)edge_idx, arclen / graph.strokes_[edge_idx].length());
    }
  }
  assert(loc.first != -1);
  assert(loc.second != -1.0);
  return loc;
}

using VertexView = StrokeGraph::VertexView;

static bool is_connected_helper(const StrokeGraph& graph, //
                                const size_t orig_si1, const Float arclen1,
                                const size_t orig_si2, const Float arclen2) {
  const auto& orig_stroke1 = graph.orig_strokes_[orig_si1];
  const auto& orig_stroke2 = graph.orig_strokes_[orig_si2];
  auto orig_pos1 = StrokeTime((int)orig_si1, arclen1 / orig_stroke1.length());
  if (arclen1 <=
      std::min(0.5 * orig_stroke1.width_at(arclen1), 0.5 * orig_stroke1.length())) {
    orig_pos1.second = 0.0;
  } else if (arclen1 >= orig_stroke1.length() - 0.5 * orig_stroke1.width_at(arclen1)) {
    orig_pos1.second = 1.0;
  }
  const auto edge_pos1 = ((orig_pos1.second == 0.0 || orig_pos1.second == 1.0)
                            ? as_edge_position(graph, orig_pos1)
                            : orig2strokes_position_preserving(graph, orig_pos1));
  const auto& edge1 = graph.strokes_[edge_pos1.first];
  const auto edge_arclen1 = edge_pos1.second * edge1.length();

  VertexView vertices_to_search[2] = {VertexView(), VertexView()};
  if (edge_arclen1 <= 0.5 * edge1.width_at(edge_arclen1)) {
    vertices_to_search[0] = graph.hedge(2 * edge_pos1.first).origin();
  }
  if (edge_arclen1 >= edge1.length() - 0.5 * edge1.width_at(edge_arclen1)) {
    vertices_to_search[1] = graph.hedge(2 * edge_pos1.first).dest();
  }

  const auto pen_width = orig_stroke2.pen_width();
  for (const auto& v : vertices_to_search) {
    if (!v.is_valid())
      continue;

    const auto he = v.hedge();
    auto it = he;
    auto iters = 0;
    do {
      {
        auto other_loc = StrokeTime((int)it.stroke_idx(), it.forward() ? 0.0 : 1.0);
        if (convert_strokes2orig(graph, other_loc) && other_loc.first == (int)orig_si2 &&
            std::abs(other_loc.second * orig_stroke2.length() - arclen2) <
              3 * pen_width) {
          return true;
        }
      }

      const auto he2 = it.twin();
      auto it2 = he2;
      do {
        auto other_loc = StrokeTime((int)it2.stroke_idx(), it2.forward() ? 0.0 : 1.0);
        if (convert_strokes2orig(graph, other_loc) && other_loc.first == (int)orig_si2 &&
            std::abs(other_loc.second * orig_stroke2.length() - arclen2) <
              3 * pen_width) {
          return true;
        }

        it2 = it2.twin().next();
        iters++;
        force_assert(iters < 1024 && "likely infinite loop found");
      } while (it2 != he2);

      it = it.twin().next();
      iters++;
      force_assert(iters < 1024 && "likely infinite loop found");
    } while (it != he);
  }
  return false;
}

static bool is_connected(const StrokeGraph& graph, //
                         const size_t orig_si1, const Float arclen1,
                         const size_t orig_si2, const Float arclen2) {
  return is_connected_helper(graph, orig_si1, arclen1, orig_si2, arclen2) ||
         is_connected_helper(graph, orig_si2, arclen2, orig_si1, arclen1);
}

static bool will_create_region(const StrokeGraph& graph, //
                               const size_t orig_si1, const Float arclen1,
                               const size_t orig_si2, const Float arclen2,
                               const size_t current_fi) {
  const auto orig_pos1 =
    StrokeTime((int)orig_si1, arclen1 / graph.orig_strokes_[orig_si1].length());
  const auto orig_pos2 =
    StrokeTime((int)orig_si2, arclen2 / graph.orig_strokes_[orig_si2].length());
  const auto edge_pos1 = orig2strokes_position_preserving(graph, orig_pos1);
  const auto edge_pos2 = orig2strokes_position_preserving(graph, orig_pos2);

  auto he1 = graph.hedge(2 * edge_pos1.first);
  if (he1.face_idx() != current_fi) {
    he1 = he1.twin();
  }
  assert(he1.face_idx() == current_fi);
  auto he2 = graph.hedge(2 * edge_pos2.first);
  if (he2.face_idx() != current_fi) {
    he2 = he2.twin();
  }
  assert(he2.face_idx() == current_fi);
  if (he1.index_ == he2.index_) {
    return true;
  }

  auto it = he1;
  auto iters = 0;
  while (true) {
    it = it.next();
    if (it.index_ == he1.index_) {
      return false;
    } else if (it.index_ == he2.index_) {
      return true;
    }

    iters++;
    force_assert(iters < 1024 && "likely infinite loop found");
  }
}

StrokeGraph bridge_added(const StrokeGraph& graph, const StrokeTime& orig_pos1,
                         const StrokeTime& orig_pos2, const Float env_dist,
                         MSVC_WARNING_SUPPRESS(4459) const Float face_size_ratio,
                         bool* const region_constraint_ok) {
  *region_constraint_ok = true;

  // Create the edge then look at the face sizes.
  auto tmp_graph = graph.clone();
  const auto graph_loc1 = orig2strokes_position_preserving(graph, orig_pos1);
  const auto v1 = add_vertex_precise_safe(tmp_graph, graph_loc1.first,
                                          graph_loc1.second *
                                            graph.strokes_[graph_loc1.first].length());
  const auto tmp_graph_loc2 = orig2strokes_position_preserving(tmp_graph, orig_pos2);
  const auto v2 = add_vertex_precise_safe(
    tmp_graph, tmp_graph_loc2.first,
    tmp_graph_loc2.second * tmp_graph.strokes_[tmp_graph_loc2.first].length());

  tmp_graph.faces_.clear();
  for (auto& edge_rec : tmp_graph.hedges_) {
    edge_rec.face_ = StrokeGraph::invalid;
  }
  const auto bridge_si = bridge_vertices(tmp_graph, v1.index_, v2.index_);
  construct_faces(tmp_graph);

  const auto edge = tmp_graph.hedge(2 * bridge_si);
  const auto twin = edge.twin();
  // assert(edge.face_idx() != twin.face_idx());
  const auto face_size1 =
    (edge.face_idx() == tmp_graph.boundary_face_
       ? INFINITY
       : std::max(0.0,
                  face_maximum_inscribing_circle_radius(tmp_graph, edge.face_idx())));
  if (2 * face_size1 <= face_size_ratio * std::max(env_dist, 0.0)) {
    *region_constraint_ok = false; // Connection too large relative to the face formed.
    return tmp_graph;
  }
  const auto face_size2 =
    (twin.face_idx() == tmp_graph.boundary_face_
       ? INFINITY
       : std::max(0.0,
                  face_maximum_inscribing_circle_radius(tmp_graph, twin.face_idx())));
  if (2 * face_size2 <= face_size_ratio * std::max(env_dist, 0.0)) {
    *region_constraint_ok = false; // Connection too large relative to the face formed.
  }
  return tmp_graph;
}

struct BridgeCandidate {
  BridgeCandidate()
    : junc_(JunctionType::X) {}

  Junction junc_;
  Float env_dist_ = NAN;
};

static BridgeInfo
find_final_bridge_locations_round1(const StrokeGraph& graph,
                                   const span<const ClassifierPrediction> candidates) {
  // Find biggest connection on each face.
  auto largest_connection = std::vector<Float>(graph.faces_.size(), 0.0);
  auto largest_connection_indices = std::vector<size_t>(graph.faces_.size(), 0);
  for (size_t i = 0; i < candidates.size(); ++i) {
    const auto& conn = candidates[i];
    if (conn.connected) {
      const auto orig_a = conn.orig_a;
      assert(conn.orig_a.second == 0.0 || conn.orig_a.second == 1.0);
      const auto orig_b =
        reproject_cand2_on_original(graph, orig_a, as_edge_position(graph, conn.orig_b));
      const auto orig_si1 = orig_a.first;
      const auto& orig_stroke1 = graph.orig_strokes_[orig_si1];
      const auto orig_arclen1 = orig_a.second * orig_stroke1.length();
      const auto orig_si2 = orig_b.first;
      const auto& orig_stroke2 = graph.orig_strokes_[orig_si2];
      const auto orig_arclen2 = orig_b.second * orig_stroke2.length();
      const auto cen_dist =
        (orig_stroke1.pos(orig_arclen1) - orig_stroke2.pos(orig_arclen2)).norm();
      const auto env_dist = cen_dist - 0.5 * (orig_stroke1.width_at(orig_arclen1) +
                                              orig_stroke2.width_at(orig_arclen2));
      auto end1 = conn.orig_a;
      const auto ok = convert_orig2strokes(graph, end1);
      force_assert(ok && "couldn't map from orig to strokes");
      const auto fi1 = graph.hedge(2 * end1.first).face_idx();
      const auto fi2 = graph.hedge(2 * end1.first + 1).face_idx();
      if (env_dist > largest_connection[fi1]) {
        largest_connection[fi1] = env_dist;
        largest_connection_indices[fi1] = i;
      }
      if (env_dist > largest_connection[fi2]) {
        largest_connection[fi2] = env_dist;
        largest_connection_indices[fi2] = i;
      }
    }
  }

  auto info = BridgeInfo();
  for (size_t fi = 0; fi < graph.faces_.size(); ++fi) {
    if (largest_connection[fi] > 0.0) {
      info.largest_connections.emplace_back(
        pred2junc(candidates[largest_connection_indices[fi]]));
    }
  }

  auto bridge_candidates = std::vector<BridgeCandidate>();

  // Find all bridge candidates (without checking region term yet).
  for (size_t fi = 0; fi < graph.faces_.size(); ++fi) {
    auto face_stroke_indices = std::vector<size_t>();
    for (const auto hi : graph.faces_[fi].cycles_) {
      const auto he = graph.hedge(hi);
      auto it = he;
      auto iters = 0;
      do {
        if (!(it.flags() & StrokeGraph::HedgeRecord::Bridge)) {
          assert(graph.strokes2orig_[it.stroke_idx()].size() == 1);
          face_stroke_indices.push_back(graph.strokes2orig_[it.stroke_idx()][0]);
        }
        iters++;
        force_assert(iters < 1024 && "likely infinite loop found");
        it = it.next();
      } while (it != he);
    }

    std::sort(face_stroke_indices.begin(), face_stroke_indices.end());
    face_stroke_indices.erase(
      std::unique(face_stroke_indices.begin(), face_stroke_indices.end()),
      face_stroke_indices.end());
    const auto n = face_stroke_indices.size();
    for (size_t i = 0; i < n; ++i) {
      const auto si = face_stroke_indices[i];
      for (size_t j = i; j < n; ++j) {
        const auto sj = face_stroke_indices[j];
        const auto& stroke_i = graph.orig_strokes_[si];
        const auto& stroke_j = graph.orig_strokes_[sj];
        if (stroke_i.size() > 1 && stroke_j.size() > 1) {
          // Find closest points on the two strokes.
          Float arclen1 = NAN, arclen2 = NAN;
          if (si == sj) {
            const auto dist = closest_points_on_self(stroke_i, arclen1, arclen2);
            if (dist < 1e-6 || !std::isfinite(dist)) {
              continue; // Likely already connected or spurious.
            }
          } else {
            const auto dist = closest_points(
              graph.orig_bvh_->nodes[si], graph.orig_bvh_->nodes[sj], arclen1, arclen2);
            if (dist < 1e-6) {
              continue; // Likely already connected.
            }
          }
          assert(std::isfinite(arclen1));
          assert(std::isfinite(arclen2));

          const auto orig_loc1 = StrokeTime((int)si, arclen1 / stroke_i.length());
          const auto graph_loc1 = orig2strokes_position_preserving(graph, orig_loc1);
          const auto graph_si = graph_loc1.first;
          if (graph.hedge(2 * graph_si).face_idx() != fi &&
              graph.hedge(2 * graph_si + 1).face_idx() != fi) {
            continue; // Not relevant to this face.
          }
          const auto orig_loc2 = StrokeTime((int)sj, arclen2 / stroke_j.length());
          const auto graph_loc2 = orig2strokes_position_preserving(graph, orig_loc2);
          const auto graph_sj = graph_loc2.first;
          if (graph.hedge(2 * graph_sj).face_idx() != fi &&
              graph.hedge(2 * graph_sj + 1).face_idx() != fi) {
            continue; // Not relevant to this face.
          }

          const auto p = stroke_i.pos(arclen1);
          const auto q = stroke_j.pos(arclen2);
          if (!line_of_sight(p, q, *graph.orig_bvh_, 1e-7, 1.0 - 1e-7)) {
            continue; // Connection is blocked.
          }
          if (!line_of_sight(graph.strokes_[graph_si].pos_norm(graph_loc1.second),
                             graph.strokes_[graph_sj].pos_norm(graph_loc2.second),
                             graph.strokes_, graph.bvh_.centerline_bbs())) {
            continue; // Connection is blocked in graph.
          }

          if (fi == graph.boundary_face_) {
            if (graph.hedge(2 * graph_si).face_idx() != graph.boundary_face_) {
              assert(graph.hedge(2 * graph_si + 1).face_idx() == graph.boundary_face_);
              const auto other_face = graph.hedge(2 * graph_si).face();
              if (point_in_face(other_face, 0.5 * (p + q))) {
                continue; // Connection not in the boundary region.
              }
            } else if (graph.hedge(2 * graph_si + 1).face_idx() != graph.boundary_face_) {
              assert(graph.hedge(2 * graph_si).face_idx() == graph.boundary_face_);
              const auto other_face = graph.hedge(2 * graph_si + 1).face();
              if (point_in_face(other_face, 0.5 * (p + q))) {
                continue; // Connection not in the boundary region.
              }
            }
          } else if (!point_in_face(graph.face(fi), 0.5 * (p + q))) {
            continue; // Connection not inside the face.
          }

          const auto cen_dist = (p - q).norm();
          const auto env_dist =
            cen_dist - 0.5 * (stroke_i.width_at(arclen1) + stroke_j.width_at(arclen2));
          if (env_dist >= largest_connection[fi]) {
            continue; // Too far.
          }

          if (is_connected(graph, si, arclen1, sj, arclen2)) {
            continue; // Already connected.
          }

          if (!will_create_region(graph, si, arclen1, sj, arclen2, /*current_fi=*/fi)) {
            continue;
          }

          auto& cand = bridge_candidates.emplace_back();
          cand.env_dist_ = env_dist;
          auto& junc = cand.junc_;
          junc.points.emplace_back(orig_loc1);
          junc.points.emplace_back(orig_loc2);
        }
      }
    }
  }

  for (size_t i = 0; i < bridge_candidates.size(); ++i) {
    const auto& cand = bridge_candidates[i];
    const auto& junc = cand.junc_;

    auto region_constraint_ok = true;
    const auto tmp_graph =
      bridge_added(graph, junc.points[0], junc.points[1],
                   /*env_dist=*/cand.env_dist_,
                   /*face_size_ratio=*/face_size_ratio[1], &region_constraint_ok);
    if (!region_constraint_ok) {
      continue; // Connection too large relative to the face formed.
    }

    auto& mut_junc = bridge_candidates[i].junc_;
    info.bridges.emplace_back(std::move(mut_junc));
    assert(mut_junc.points.empty());
  }

  // Remove already-accepted bridges.
  bridge_candidates.erase(
    std::remove_if(bridge_candidates.begin(), bridge_candidates.end(),
                   [](const BridgeCandidate& c) { return c.junc_.points.empty(); }),
    bridge_candidates.end());

  return info;
}

static BridgeInfo
find_final_bridge_locations_later(const StrokeGraph& graph,
                                  span<const ClassifierPrediction> candidates,
                                  const int round) {
  auto info = BridgeInfo();

  for (size_t i = 0; i < candidates.size(); ++i) {
    const auto& cand = candidates[i];
    if (!cand.connected) {
      if (cand.prob < prob_threshold[round]) {
        continue;
      }

      auto junc = pred2junc(cand);
      const auto si = junc.points[0].first;
      const auto sj = junc.points[1].first;
      const auto& stroke_i = graph.orig_strokes_[si];
      const auto& stroke_j = graph.orig_strokes_[sj];
      auto arclen1 = junc.points[0].second * stroke_i.length();
      auto arclen2 = junc.points[1].second * stroke_j.length();

      // TODO: Check against longest connection made in the region.

      if (is_connected(graph, si, arclen1, sj, arclen2)) {
        continue; // Already connected.
      }

      const auto p = stroke_i.pos(arclen1);
      const auto q = stroke_j.pos(arclen2);
      const auto midpoint = 0.5 * (p + q);

      auto fi = graph.boundary_face_;
      for (size_t possible_fi = 0; possible_fi < graph.faces_.size(); ++possible_fi) {
        if (possible_fi != graph.boundary_face_) {
          if (point_in_face(graph.face(possible_fi), midpoint)) {
            fi = possible_fi;
            break;
          }
        }
      }

      const auto orig_loc1 = StrokeTime((int)si, arclen1 / stroke_i.length());
      const auto graph_loc1 = orig2strokes_position_preserving(graph, orig_loc1);
      const auto graph_si = graph_loc1.first;
      if (graph.hedge(2 * graph_si).face_idx() != fi &&
          graph.hedge(2 * graph_si + 1).face_idx() != fi) {
        continue; // Not relevant to this face.
      }
      const auto orig_loc2 = StrokeTime((int)sj, arclen2 / stroke_j.length());
      const auto graph_loc2 = orig2strokes_position_preserving(graph, orig_loc2);
      const auto graph_sj = graph_loc2.first;
      if (graph.hedge(2 * graph_sj).face_idx() != fi &&
          graph.hedge(2 * graph_sj + 1).face_idx() != fi) {
        continue; // Not relevant to this face.
      }

      if (!line_of_sight(p, q, *graph.orig_bvh_)) {
        continue; // Connection is blocked.
      }
      if (!line_of_sight(graph.strokes_[graph_si].pos_norm(graph_loc1.second),
                         graph.strokes_[graph_sj].pos_norm(graph_loc2.second),
                         graph.strokes_, graph.bvh_.centerline_bbs())) {
        continue; // Connection is blocked in graph.
      }

      if (!will_create_region(graph, si, arclen1, sj, arclen2, /*current_fi=*/fi)) {
        continue;
      }

      const auto cen_dist = (p - q).norm();
      const auto env_dist = std::max(
        0.0, cen_dist - 0.5 * (stroke_i.width_at(arclen1) + stroke_j.width_at(arclen2)));

      auto region_constraint_ok = true;
      const auto tmp_graph =
        bridge_added(graph, junc.points[0], junc.points[1],
                     /*env_dist=*/env_dist,
                     /*face_size_ratio=*/face_size_ratio[round], &region_constraint_ok);
      if (!region_constraint_ok) {
        continue; // Connection too large relative to the face formed.
      }

      auto& out_junc = info.bridges.emplace_back(std::move(junc));
      out_junc.type = JunctionType::X;
    }
  }

  return info;
}

BridgeInfo find_final_bridge_locations(const StrokeGraph& graph,
                                       const span<const ClassifierPrediction> candidates,
                                       const int round) {
  force_assert(round > 0 && "round is 1-indexed");
  force_assert(round < (int)face_size_ratio.size() && "invalid round");

  if (round == 1) {
    return find_final_bridge_locations_round1(graph, candidates);
  } else {
    return find_final_bridge_locations_later(graph, candidates, round);
  }
}

std::vector<Junction>
augment_with_final_bridges(StrokeGraph& graph, const BridgeInfo& info, const int round) {
  force_assert(round > 0 && "round is 1-indexed");
  force_assert(round < (int)face_size_ratio.size() && "invalid round");

  std::vector<Junction> connected_junctions;

  const auto n_bridges = info.bridges.size();
  auto connection_distances = std::vector<Float>(n_bridges, INFINITY);
  for (size_t i = 0; i < n_bridges; ++i) {
    const auto& junc = info.bridges[i];
    const auto [orig_si1, narclen1] = junc.points[0];
    const auto [orig_si2, narclen2] = junc.points[1];
    const auto& stroke1 = graph.orig_strokes_[orig_si1];
    const auto& stroke2 = graph.orig_strokes_[orig_si2];
    const auto arclen1 = narclen1 * stroke1.length();
    const auto arclen2 = narclen2 * stroke2.length();
    const auto cen_dist = (stroke1.pos(arclen1) - stroke2.pos(arclen2)).norm();
    const auto env_dist = std::max(
      0.0, cen_dist - 0.5 * (stroke1.width_at(arclen1) + stroke2.width_at(arclen2)));
    connection_distances[i] = env_dist;
  }

  auto connections_by_distance = std::vector<int>(n_bridges, -1);
  for (size_t i = 0; i < n_bridges; ++i) {
    connections_by_distance[i] = (int)i;
  }
  std::sort(connections_by_distance.begin(), connections_by_distance.end(),
            [&](const auto a, const auto b) {
              return connection_distances[a] < connection_distances[b];
            });

  const auto ratio = face_size_ratio[round];
  for (const auto i : connections_by_distance) {
    const auto& junc = info.bridges[i];
    const auto graph_loc1 = orig2strokes_position_preserving(graph, junc.points[0]);
    const auto graph_loc2 = orig2strokes_position_preserving(graph, junc.points[1]);

    // Line of sight can be lost due to a previous connection.
    if (line_of_sight(graph.strokes_[graph_loc1.first].pos_norm(graph_loc1.second),
                      graph.strokes_[graph_loc2.first].pos_norm(graph_loc2.second),
                      graph.strokes_, graph.bvh_.centerline_bbs())) {
      const auto env_dist = connection_distances[i];
      auto region_constraint_ok = true;
      auto tmp_graph = bridge_added(graph, junc.points[0], junc.points[1], env_dist,
                                    /*face_size_ratio=*/ratio, &region_constraint_ok);
      if (region_constraint_ok) {
        connected_junctions.emplace_back(junc);
        // Temporarily use this type to indicate bridges
        connected_junctions.back().type = JunctionType::X;
        graph = std::move(tmp_graph);
      }
    }
  }

  return connected_junctions;
}

void augment_with_final_bridges(StrokeGraph& graph, const BridgeInfo& info) {
  const auto n_bridges = info.bridges.size();
  auto connection_distances = std::vector<Float>(n_bridges, INFINITY);
  for (size_t i = 0; i < n_bridges; ++i) {
    const auto& junc = info.bridges[i];
    const auto [orig_si1, narclen1] = junc.points[0];
    const auto [orig_si2, narclen2] = junc.points[1];
    const auto& stroke1 = graph.orig_strokes_[orig_si1];
    const auto& stroke2 = graph.orig_strokes_[orig_si2];
    const auto arclen1 = narclen1 * stroke1.length();
    const auto arclen2 = narclen2 * stroke2.length();
    const auto cen_dist = (stroke1.pos(arclen1) - stroke2.pos(arclen2)).norm();
    const auto env_dist = std::max(
      0.0, cen_dist - 0.5 * (stroke1.width_at(arclen1) + stroke2.width_at(arclen2)));
    connection_distances[i] = env_dist;
  }

  auto connections_by_distance = std::vector<int>(n_bridges, -1);
  for (size_t i = 0; i < n_bridges; ++i) {
    connections_by_distance[i] = (int)i;
  }
  std::sort(connections_by_distance.begin(), connections_by_distance.end(),
            [&](const auto a, const auto b) {
              return connection_distances[a] < connection_distances[b];
            });

  const auto ratio = std::numeric_limits<Float>::infinity();
  for (const auto i : connections_by_distance) {
    const auto& junc = info.bridges[i];
    const auto graph_loc1 = orig2strokes_position_preserving(graph, junc.points[0]);
    const auto graph_loc2 = orig2strokes_position_preserving(graph, junc.points[1]);

    // Line of sight can be lost due to a previous connection.
    if (line_of_sight(graph.strokes_[graph_loc1.first].pos_norm(graph_loc1.second),
                      graph.strokes_[graph_loc2.first].pos_norm(graph_loc2.second),
                      graph.strokes_, graph.bvh_.centerline_bbs())) {
      const auto env_dist = connection_distances[i];
      auto region_constraint_ok = true;
      auto tmp_graph = bridge_added(graph, junc.points[0], junc.points[1], env_dist,
                                    /*face_size_ratio=*/ratio, &region_constraint_ok);
      if (region_constraint_ok) {
        graph = std::move(tmp_graph);
      }
    }
  }
}

void augment_with_overlapping_final_bridges(StrokeGraph& graph, const BridgeInfo& info) {
  const auto n_bridges = info.bridges.size();
  auto connection_distances = std::vector<Float>(n_bridges, INFINITY);
  for (size_t i = 0; i < n_bridges; ++i) {
    const auto& junc = info.bridges[i];
    const auto [orig_si1, narclen1] = junc.points[0];
    const auto [orig_si2, narclen2] = junc.points[1];
    const auto& stroke1 = graph.orig_strokes_[orig_si1];
    const auto& stroke2 = graph.orig_strokes_[orig_si2];
    const auto arclen1 = narclen1 * stroke1.length();
    const auto arclen2 = narclen2 * stroke2.length();
    const auto cen_dist = (stroke1.pos(arclen1) - stroke2.pos(arclen2)).norm();
    const auto env_dist = std::max(
      0.0, cen_dist - 0.5 * (stroke1.width_at(arclen1) + stroke2.width_at(arclen2)));
    connection_distances[i] = env_dist;
  }

  auto connections_by_distance = std::vector<int>(n_bridges, -1);
  for (size_t i = 0; i < n_bridges; ++i) {
    connections_by_distance[i] = (int)i;
  }
  std::sort(connections_by_distance.begin(), connections_by_distance.end(),
            [&](const auto a, const auto b) {
              return connection_distances[a] < connection_distances[b];
            });

  for (const auto i : connections_by_distance) {
    const auto env_dist = connection_distances[i];
    if (env_dist > 0.0) {
      continue;
    }

    const auto& junc = info.bridges[i];
    const auto graph_loc1 = orig2strokes_position_preserving(graph, junc.points[0]);
    const auto graph_loc2 = orig2strokes_position_preserving(graph, junc.points[1]);

    // Line of sight can be lost due to a previous connection.
    if (line_of_sight(graph.strokes_[graph_loc1.first].pos_norm(graph_loc1.second),
                      graph.strokes_[graph_loc2.first].pos_norm(graph_loc2.second),
                      graph.strokes_, graph.bvh_.centerline_bbs())) {
      auto region_constraint_ok = true;
      auto tmp_graph = bridge_added(graph, junc.points[0], junc.points[1], env_dist,
                                    /*face_size_ratio=*/DBL_MAX, &region_constraint_ok);
      if (region_constraint_ok) {
        graph = std::move(tmp_graph);
      }
    }
  }
}

void multi_bridge(const StrokeGraph& plane_graph,
                  const std::vector<Junction>& in_candidates,
                  StrokeGraph::SnappingType snapping_type, FeatureType feature_type,
                  StrokeSnapInfo& predictions, StrokeGraph& stroke_graph,
                  Float accept_ratio, Float lowest_p, Float largest_non_region_gap,
                  Float accept_ratio_factor) {
  bool to_add = true;

  std::vector<Junction> candidates = in_candidates;
  std::vector<Junction> all_connected_candidates;
  std::vector<Junction> connected_candidates;
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (!candidates[i].repr.empty()) {
      if (to_add)
        all_connected_candidates.emplace_back(candidates[i]);
      else
        connected_candidates.emplace_back(candidates[i]);
    }
  }

  // Decompose based on containing region
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (!candidates[i].repr.empty())
      continue;
    candidates[i].fi = 0;
    for (size_t j = 1; j < stroke_graph.faces_.size(); ++j) {
      auto junc = candidates[i];
      auto ok = original_stroke_to_stroke_indexing(stroke_graph, junc);
      force_assert(ok && "couldn't map from orig to strokes");
      auto p1 =
        stroke_graph.strokes_[junc.points[0].first].pos_norm(junc.points[0].second);
      auto p2 =
        stroke_graph.strokes_[junc.points[1].first].pos_norm(junc.points[1].second);
      if (point_in_face(stroke_graph.face(j), 0.5 * (p1 + p2))) {
        candidates[i].fi = (int)j;
        break;
      }
    }
  }

  // 1. Pick disconnected pairs
  StrokeGraph result_graph = stroke_graph.clone();
  std::set<size_t> connected_indices;
  bool changed = false;
  do {
    changed = false;
    std::map<Float, std::pair<size_t, size_t>> sorted_pairs;
    for (size_t i = 0; i + 1 < candidates.size(); ++i) {
      if (!candidates[i].repr.empty() || connected_indices.count(i) ||
          candidates[i].probability < lowest_p /*||
          candidates[i].probability >= junction_cutoff*/
          || has_bridge_vertex(stroke_graph, candidates[i]))
        continue;
      for (size_t j = i + 1; j < candidates.size(); ++j) {
        if (!candidates[j].repr.empty() || connected_indices.count(j) ||
            candidates[j].probability < lowest_p /*||
            candidates[j].probability >= junction_cutoff*/
            || has_bridge_vertex(stroke_graph, candidates[j]))
          continue;

        if (candidates[i].fi != candidates[j].fi)
          continue;
        Float prob = candidates[i].probability + candidates[j].probability;
        prob = -prob;
        while (sorted_pairs.count(prob))
          prob += 1e-5;
        sorted_pairs[prob] = std::make_pair(i, j);
      }
    }

    struct MultiConnection {
      std::pair<size_t, size_t> pair;
      StrokeGraph graph;
      std::vector<Junction> candidates;
    };

    std::map<Float, MultiConnection> sorted_connectable_pairs;
    for (const auto& [prob, p] : sorted_pairs) {
      size_t i = p.first;
      size_t j = p.second;
      // 2. Extend both and see if they intersect
      std::vector<Junction> connection_juncs{candidates[i], candidates[j]};
      std::vector<std::vector<Junction>> expanded_connection_juncs;
      std::vector<Junction> component_check = all_connected_candidates;
      component_check.insert(component_check.end(), connected_candidates.begin(),
                             connected_candidates.end());
      for (const auto& junc : connection_juncs) {
        std::vector<Junction> expanded_juncs;
        expand_adjacent_junction(stroke_graph, junc, expanded_juncs);
        if (expanded_juncs.empty()) {
          expanded_juncs.emplace_back(junc);
        }
        component_check.insert(component_check.end(), expanded_juncs.begin(),
                               expanded_juncs.end());
        expanded_connection_juncs.emplace_back(expanded_juncs);
      }

      decompose_candidates(plane_graph, component_check);
      std::unordered_set<int> to_connect_comp;
      for (const auto& junc : component_check) {
        for (const auto& junc2 : connection_juncs) {
          if ((junc.points[0] == junc2.points[0] && junc.points[1] == junc2.points[1]) ||
              (junc.points[0] == junc2.points[1] && junc.points[0] == junc2.points[1])) {
            to_connect_comp.emplace(junc.component_idx);
          }
        }
      }

      if (to_connect_comp.size() == 1)
        continue;

      // 3. Connect
      std::vector<Junction> varying_candidates;
      StrokeGraph varying_graph;
      if (!to_add) {
        varying_candidates = connected_candidates;
        for (const auto& juncs : expanded_connection_juncs) {
          for (const auto& junc : juncs) {
            varying_candidates.emplace_back(junc);
          }
        }
        varying_graph = plane_graph.clone();

      } else {
        varying_candidates = connected_candidates;
        for (const auto& juncs : expanded_connection_juncs) {
          for (const auto& junc : juncs) {
            varying_candidates.emplace_back(junc);
          }
        }
        varying_graph = stroke_graph.clone();
      }

      std::vector<bool> connected_flags;
      connected_flags.resize(varying_candidates.size(), true);
      std::vector<std::pair<size_t, size_t>> adj_faces;
      std::vector<Float> junc_distances;
      std::vector<StrokeGraph::VertexID> junc_vertex_ids;
      auto modified_graph =
        modify_graph(varying_graph, varying_candidates, connected_flags, adj_faces,
                     junc_distances, junc_vertex_ids);

      // 4. Check if they form a new region
      if (modified_graph && modified_graph->faces_.size() > result_graph.faces_.size()) {
        varying_graph = std::move(*modified_graph);

        // 5. Check if both of them are on boundary
        std::set<std::string> new_juncs;
        std::map<size_t, std::vector<Junction>> f2juncs;
        std::map<size_t, std::unordered_set<size_t>> junc2fis;
        for (size_t fi = 1; fi < varying_graph.faces_.size(); ++fi) {
          std::vector<Junction> in_face_junctions;
          find_junctions_in_face(varying_graph, varying_candidates, fi, in_face_junctions,
                                 false);
          bool seen_new = false;
          for (const auto& junc : in_face_junctions) {
            for (size_t jj = 0; jj < expanded_connection_juncs.size(); ++jj) {
              const auto& ext_juncs = expanded_connection_juncs[jj];
              for (const auto& junc2 : ext_juncs) {
                if ((junc.points[0] == junc2.points[0] &&
                     junc.points[1] == junc2.points[1]) ||
                    (junc.points[0] == junc2.points[1] &&
                     junc.points[0] == junc2.points[1])) {
                  new_juncs.emplace(junc.repr);
                  seen_new = true;
                  junc2fis[jj].emplace(fi);
                }
              }
            }
          }

          if (seen_new) {
            f2juncs[fi] = in_face_junctions;
          }
        }

        // At least one is not on boundary
        if (new_juncs.size() < 2)
          continue;

        // Check if they share a face
        bool share_one = false;
        for (const auto& [fi, juncs] : f2juncs) {
          bool in_all = true;
          for (const auto& [junc, fis] : junc2fis) {
            if (!fis.count(fi)) {
              in_all = false;
              break;
            }
          }
          if (in_all)
            share_one = true;
        }
        if (!share_one)
          continue;

        bool too_large = false;
        Float min_ratio = std::numeric_limits<Float>::infinity();
        bool face_checked = false;
        for (const auto& [fi, juncs] : f2juncs) {
          Float largest_gap = -1;
          for (const auto& junc : juncs) {
            assert(junc.orig_dist >= 0);
            if (!new_juncs.count(junc.repr))
              largest_gap = std::max(junc.orig_dist, largest_gap);
          }

          Float existing_largest_gap = -1;
          for (const auto& junc : juncs) {
            assert(junc.orig_dist >= 0);
            if (!new_juncs.count(junc.repr))
              existing_largest_gap = std::max(junc.orig_dist, existing_largest_gap);
          }

          largest_gap = -1;
          for (const auto& junc : juncs) {
            assert(junc.orig_dist >= 0);
            if (new_juncs.count(junc.repr))
              largest_gap = std::max(junc.orig_dist, largest_gap);
          }
          assert(largest_gap >= 0);
          largest_gap = std::max(1e-10, largest_gap);

          // Check the ratio
          std::unordered_map<std::string, Float> region_size_cache;
          Float radius = get_face_size_const(varying_graph, region_size_cache, fi);
          Float ratio = 2 * radius / largest_gap;
          min_ratio = std::min(min_ratio, ratio);
          face_checked = true;

          // Check if any existing connection now violates the hard constraint
          if (!euclidean_region_condition_check(radius, existing_largest_gap))
            too_large = true;

          // Use hard threshold here
          if (accept_ratio_factor < 0) {
            assert(accept_ratio >= 0);
            if (ratio < accept_ratio) {
              too_large = true;
            }
          }
          // Use soft threshold here
          else {
            Float lowest_prob = 1;
            for (const auto& junc : juncs) {
              if (new_juncs.count(junc.repr))
                lowest_prob = std::min(lowest_prob, junc.probability);
            }
            Float shifted_prob =
              (lowest_prob * 100 + accept_ratio_factor * ratio) / 100.0;
            if (shifted_prob < 0.5 || ratio < 2.5)
              too_large = true;
          }

          if (too_large)
            break;
        }

        if (too_large)
          continue;

        assert(face_checked);

        min_ratio = -min_ratio;
        while (sorted_connectable_pairs.count(min_ratio))
          min_ratio += 1e-5;

        std::vector<Junction> varying_connected_candidates;
        sorted_connectable_pairs[min_ratio] = MultiConnection{
          std::make_pair(i, j), std::move(varying_graph), std::move(varying_candidates)};

        // We now connect based on summed probability
        break;
      }
    }

    if (!sorted_connectable_pairs.empty()) {
      changed = true;
      result_graph = std::move(sorted_connectable_pairs.begin()->second.graph);
      connected_candidates =
        std::move(sorted_connectable_pairs.begin()->second.candidates);
      connected_indices.emplace(sorted_connectable_pairs.begin()->second.pair.first);
      connected_indices.emplace(sorted_connectable_pairs.begin()->second.pair.second);
    }
  } while (changed);

  stroke_graph = std::move(result_graph);
  for (size_t j = 0; j < connected_candidates.size(); ++j) {
    std::vector<Junction> junctions{connected_candidates[j]};
    std::vector<std::pair<ClassifierPrediction, VizProbabilities>> pred;
    junc2pred(stroke_graph, junctions, pred, std::vector<Junction>());
    predictions.predictions.emplace_back(pred.front().first);
    predictions.predictions.back().connected = !connected_candidates[j].repr.empty();
  }
}

std::vector<Junction> unbridge_interior(const StrokeGraph& stroke_graph,
                                        const std::vector<Junction>& candidates) {
  std::vector<Junction> candidates_to_disconnect;

  // Build the junction, region association
  std::set<size_t> int_juncs;
  std::map<size_t, std::vector<Junction>> f2juncs;
  std::map<size_t, std::unordered_set<size_t>> junc2int_fis;
  for (size_t fi = 0; fi < stroke_graph.faces_.size(); ++fi) {
    std::vector<Junction> face_boundary_junctions;
    find_junctions_in_face(stroke_graph, candidates, fi, face_boundary_junctions, false);
    std::vector<Junction> in_face_junctions;
    find_junctions_in_face(stroke_graph, candidates, fi, in_face_junctions, true);
    for (const auto& junc : in_face_junctions) {
      bool is_interior = true;
      for (size_t jj = 0; jj < face_boundary_junctions.size(); ++jj) {
        const auto& junc2 = face_boundary_junctions[jj];
        if ((junc.points[0] == junc2.points[0] && junc.points[1] == junc2.points[1]) ||
            (junc.points[0] == junc2.points[1] && junc.points[0] == junc2.points[1])) {
          is_interior = false;
          break;
        }
      }

      for (size_t jj = 0; jj < candidates.size(); ++jj) {
        const auto& junc2 = candidates[jj];
        if ((junc.points[0] == junc2.points[0] && junc.points[1] == junc2.points[1]) ||
            (junc.points[0] == junc2.points[1] && junc.points[0] == junc2.points[1])) {
          if (is_interior) {
            int_juncs.emplace(jj);
            junc2int_fis[jj].emplace(fi);
          }
        }
      }
    }
    f2juncs[fi] = face_boundary_junctions;
  }

  // Reconsider interior connected candidates
  for (const auto ji : int_juncs) {
    assert(junc2int_fis[ji].size() == 1);

    // Get the largest gap on the corresponding boundary
    const auto& junc = candidates[ji];
    size_t fi = *junc2int_fis[ji].begin();
    Float largest_gap = -1;
    for (const auto& junc2 : f2juncs[fi]) {
      auto tmp_junc = junc2;
      if (tmp_junc.orig_dist < 0) {
        tmp_junc.orig_dist = junction_distance_init(stroke_graph, tmp_junc);
      }
      largest_gap = std::max(largest_gap, tmp_junc.orig_dist);
    }

    auto tmp_junc = junc;
    if (tmp_junc.orig_dist < 0) {
      tmp_junc.orig_dist = junction_distance_init(stroke_graph, tmp_junc);
    }
    if (largest_gap >= 0 && largest_gap < tmp_junc.orig_dist)
      candidates_to_disconnect.emplace_back(junc);
  }

  return candidates_to_disconnect;
}

} // namespace sketching
