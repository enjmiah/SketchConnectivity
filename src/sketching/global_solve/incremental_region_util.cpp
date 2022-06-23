#include "incremental_region_util.h"

#include "../clipping.h"
#include "../detail/suppress_warning.h"
#include "../endpoint.h"
#include "../features/junction_features_impl.h"
#include "../force_assert.h"
#include "../intersect.h"
#include "../junction.h"
#include "../render.h"
#include "../resample.h"
#include "../stroke_graph.h"
#include "../stroke_graph_extra.h"
#include "incremental_decomposition.h"
#include "incremental_obj.h"
#include "incremental_param.h"
#include "incremental_util.h"

#include <mapbox/polylabel.hpp>

#include <deque>

#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(disable : 4100) // unreferenced formal parameter
#pragma warning(disable : 4267) // type conversion
#endif

namespace sketching {
extern HardConstraintType hard_constraint_type = HardConstraintType::JunctionDist;
extern Float hard_region_junc_ratio = 1; // Initial one for the junction distance
// extern Float hard_region_junc_ratio = 1.0 / 4; // Initial one for the stroke width
// extern Float hard_region_junc_ratio = 0; // Disable the region hard constraint

namespace {
double clip_path_inv_coord(const ClipperLib::cInt& p_coord) {
  return double(p_coord) / clip_precision_factor;
}

/**
 * Find the distance from point p to the closest stroke envelope on face fi.
 */
static Float distance_to_face_envelopes(Vec2 p, const StrokeGraph& graph, Index fi) {
  auto best_dist = std::numeric_limits<Float>::infinity();
  for (const auto cycle : graph.face(fi).cycles()) {
    const auto he = graph.hedge(cycle);
    auto it = he;
    do {
      const auto si = it.stroke_idx();
      if (graph.bvh_.envelope_bbs()[si].distanceTo(p) < best_dist) {
        best_dist = std::min(best_dist, signed_dist_stroke(p, it.stroke()));
      }
      it = it.next();
    } while (it != he);
  }
  return best_dist;
}

inline size_t end_hash(size_t sid, Float time) {
  bool is_tail = time > 0.5;
  return sid * 10 + is_tail;
}

void build_high_valence_star(
  const StrokeGraph::VertexView v,
  const std::unordered_map<size_t, size_t>& snapped_connections,
  std::unordered_map<size_t, size_t>& star_indexing,
  std::unordered_map<size_t, size_t>& star_count) {
  size_t stroke_star_count = 0;
  auto he = v.hedge();
  auto hit = he;
  do {
    auto end = hedge_to_endpoint(hit);
    size_t e_hash = end_hash(end.stroke_idx(), end.is_tail());
    star_indexing[e_hash] = stroke_star_count++;
    size_t cur = e_hash;
    hit = hit.twin().next();

    end = hedge_to_endpoint(hit);
    size_t next = end_hash(end.stroke_idx(), end.is_tail());

    // Register the snapped connections
    if (snapped_connections.count(cur) && snapped_connections.at(cur) == next) {
      star_count[cur]++;
      star_count[next]++;
    }
  } while (hit != he);
}

bool is_t_positioned(const Junction& candidate) {
  return candidate.type == JunctionType::T ||
         ((std::abs(candidate.points[0].second) > 1e-10) &&
          (std::abs(1 - candidate.points[0].second) > 1e-10)) ||
         ((std::abs(candidate.points[1].second) > 1e-10) &&
          (std::abs(1 - candidate.points[1].second) > 1e-10));
}

bool lookup_junc_hash(const StrokeGraph::VertexView v, const Junction& candidate,
                      size_t& s1_hash, size_t& s2_hash, int& s1_hash2, int& s2_hash2) {
  Junction junc = candidate;
  bool ok = original_stroke_to_stroke_indexing(*v.graph_, junc);
  force_assert(ok && "couldn't map from orig to strokes");

  s1_hash = end_hash(junc.points[0].first, junc.points[0].second);
  s2_hash = end_hash(junc.points[1].first, junc.points[1].second);
  s1_hash2 = -1;
  s2_hash2 = -1;
  bool cont_both_sides = false;
  // It's possible that we pick the wrong side out of the T continous side
  if (is_t_positioned(candidate)) {
    auto h = endpoint_to_hedge(
      *v.graph_, Endpoint{(size_t)junc.points[0].first, junc.points[0].second < 0.5});
    auto c_he = h.continuity_edge();
    if (c_he.is_valid() && c_he.is_active()) {
      auto end1 = hedge_to_endpoint(c_he);
      s1_hash2 = end_hash(end1.stroke_idx(), end1.is_tail());
    }

    if (candidate.is_weak && candidate.points[1].second != 0.0 &&
        candidate.points[1].second != 1.0) {
      h = endpoint_to_hedge(
        *v.graph_, Endpoint{(size_t)junc.points[1].first, junc.points[1].second < 0.5});
      c_he = h.continuity_edge();
      if (c_he.is_valid() && c_he.is_active()) {
        auto end1 = hedge_to_endpoint(c_he);
        s2_hash2 = end_hash(end1.stroke_idx(), end1.is_tail());
        cont_both_sides = true;
      }
    }
  }

  return cont_both_sides;
}

bool adj_test(const std::unordered_map<size_t, size_t>& star_indexing, const int s1_hash,
              const int s2_hash) {
  if (s1_hash < 0 || s2_hash < 0 ||
      !(star_indexing.count(s1_hash) && star_indexing.count(s2_hash)))
    return true;
  // Can only connect adjacent strokes
  if (((star_indexing.at(s1_hash) + 1) % star_indexing.size() !=
         star_indexing.at(s2_hash) &&
       (star_indexing.at(s2_hash) + 1) % star_indexing.size() !=
         star_indexing.at(s1_hash))) {

    return false;
  }
  return true;
}

// Note is_assignment_possible is different from is_assignment_valid.
// is_assignment_possible checks if this assignment is possible (aka can we extend this
// state in the region solve) with respect to the snapped stroke topology.
bool is_assignment_possible(
  const Junction& candidate, const StrokeGraph::VertexView v,
  const std::unordered_map<size_t, size_t>& snapped_connections) {
  // Collect the star at v
  std::unordered_map<size_t, size_t> star_indexing;
  std::unordered_map<size_t, size_t> star_count;
  build_high_valence_star(v, snapped_connections, star_indexing, star_count);

  // Register the connection
  Junction junc = candidate;
  bool ok = original_stroke_to_stroke_indexing(*v.graph_, junc);
  force_assert(ok && "couldn't map from orig to strokes");
  assert(
    (junc.type == JunctionType::R &&
     (std::abs(junc.points[0].second) < 1e-10 ||
      std::abs(1.0 - junc.points[0].second) < 1e-10) &&
     (std::abs(junc.points[1].second) < 1e-10 ||
      std::abs(1.0 - junc.points[1].second) < 1e-10)) ||
    (junc.type == JunctionType::T && (std::abs(junc.points[1].second) < 1e-10 ||
                                      std::abs(1.0 - junc.points[1].second) < 1e-10)));

  size_t s1_hash;
  size_t s2_hash;
  int s1_hash2;
  int s2_hash2;
  bool cont_both_sides =
    lookup_junc_hash(v, candidate, s1_hash, s2_hash, s1_hash2, s2_hash2);

  assert(
    (!is_t_positioned(candidate) && star_indexing.count(s1_hash) &&
     star_indexing.count(s2_hash)) ||
    (is_t_positioned(candidate) &&
     ((star_indexing.count(s1_hash) && star_indexing.count(s2_hash)) ||
      (s1_hash2 >= 0 && star_indexing.count(s1_hash2) && star_indexing.count(s2_hash)) ||
      (cont_both_sides && star_indexing.count(s2_hash2)))));

  if (!adj_test(star_indexing, s1_hash, s2_hash)) {
    if (is_t_positioned(candidate) && adj_test(star_indexing, s1_hash2, s2_hash))
      s1_hash = s1_hash2;
    else if (cont_both_sides && adj_test(star_indexing, s1_hash, s2_hash2))
      s2_hash = s2_hash2;
    else if (cont_both_sides && adj_test(star_indexing, s1_hash2, s2_hash2)) {
      s1_hash = s1_hash2;
      s2_hash = s2_hash2;
    } else
      return false;
  }

  return true;
}

void build_snapped_connections(const StrokeGraph::VertexView v,
                               std::unordered_map<size_t, size_t>& snapped_connections) {
  auto he = v.hedge();
  auto hit = he;
  do {
    auto end = hedge_to_endpoint(hit);
    size_t cur = end_hash(end.stroke_idx(), end.is_tail());
    hit = hit.twin().next();
    end = hedge_to_endpoint(hit);
    size_t next = end_hash(end.stroke_idx(), end.is_tail());

    snapped_connections[cur] = next;
  } while (hit != he);
}

size_t end_vid(const StrokeGraph& stroke_graph, const StrokeTime& end) {
  StrokeTime s_end = end;
  auto ok = convert_orig2strokes(stroke_graph, s_end);
  force_assert(ok && "couldn't convert from orig to strokes");

  if (!(s_end.second == 0.0 || s_end.second == 1.0)) {
    SPDLOG_WARN("Mapped position {} on stroke {} not at vertex", s_end.second,
                s_end.first);
  }
  s_end.second = (s_end.second < 0.5) ? 0.0 : 1.0;
  const auto v =
    endpoint_to_vertex(stroke_graph, Endpoint(s_end.first, s_end.second == 0.0));
  assert(v.is_valid() && v.is_active());
  return v.index_;
}

} // namespace

void group_connected_junctions(const StrokeGraph& stroke_graph,
                               span<const Junction> candidates,
                               const std::vector<bool>& junction_connected,
                               std::vector<size_t>& connected_indices,
                               std::vector<size_t>& component_labels) {
  connected_indices.clear();

  std::vector<Junction> connected_candidates;
  std::vector<size_t> init_connected_indices;
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (!junction_connected[i])
      continue;
    init_connected_indices.emplace_back(i);
    connected_candidates.emplace_back(candidates[i]);
  }

  std::vector<int> junc_components;
  color_endpoint_graph(stroke_graph, connected_candidates, junc_components, 0.0);

  std::map<size_t, std::vector<size_t>> comp2cand;
  for (size_t i = 0; i < junc_components.size(); ++i) {
    assert(junc_components[i] >= 0);
    if (!comp2cand.count(junc_components[i]))
      comp2cand[junc_components[i]] = std::vector<size_t>();
    comp2cand[junc_components[i]].emplace_back(init_connected_indices[i]);
  }

  for (const auto& [c, junc_indices] : comp2cand) {
    // Connect T-junctions first. Otherwise, some assignments may fail since we can't snap
    // vertex with valence > 1 to an interior position.
    for (const auto idx : junc_indices) {
      if (candidates[idx].type == JunctionType::R)
        continue;
      connected_indices.emplace_back(idx);
      component_labels.emplace_back(c);
    }
    for (const auto idx : junc_indices) {
      if (candidates[idx].type == JunctionType::T)
        continue;
      connected_indices.emplace_back(idx);
      component_labels.emplace_back(c);
    }
  }
}

void propose_candidates_incremental(span<const Stroke> strokes,
                                    const StrokeGraph& final_graph,
                                    const std::vector<Junction>& final_predictions,
                                    const GraphState& in_state, size_t cur_stroke_step,
                                    std::vector<Junction>& new_candidates,
                                    bool train_time) {
  const StrokeGraph& stroke_graph = in_state.graph_;
  IncrementalCache cache = in_state.cache_;

  // 1. Generate all new candidate junctions
  assert(!stroke_graph.orig_strokes_.empty());
  std::vector<std::vector<Junction>> all_new_candidate_sets;
  all_snap_candidates(stroke_graph, stroke_graph.orig_strokes_.size() - 1, &cache,
                      cache.allowed_connections_, prediction_feature_type,
                      all_new_candidate_sets, snap_candidate_count, train_time);
  size_t allowed_idx = all_new_candidate_sets.size();
  all_snap_candidates(stroke_graph, stroke_graph.orig_strokes_.size() - 1, &cache,
                      cache.enforced_connections_, prediction_feature_type,
                      all_new_candidate_sets, snap_candidate_count, train_time);

  // Only saves valid candidates
  for (size_t i = 0; i < all_new_candidate_sets.size(); ++i) {
    const auto& all_new_candidates = all_new_candidate_sets[i];
    // TODO: change to use a single boolean. The current version is for debugging.
    std::vector<bool> junc_valid;
    for (const auto& junc : all_new_candidates) {
      junc_valid.emplace_back(is_candidate_valid(
        stroke_graph, strokes, final_graph, cur_stroke_step, junc, final_predictions));
    }

    // Bind the candidates from the same expansion together
    if (std::all_of(junc_valid.begin(), junc_valid.end(), [](bool v) { return v; })) {
      for (const auto& junc : all_new_candidates) {
        // It's possible to get duplicate variables due to the expansion
        if (std::find(new_candidates.begin(), new_candidates.end(), junc) ==
            new_candidates.end()) {
          new_candidates.emplace_back(junc);
          // Enforce this connection (if the graph allows)
          if (i >= allowed_idx)
            new_candidates.back().probability = 1;
        }
      }
    }
  }

  // 2. Order the junctions based on the drawing order. Since the new junctions must
  // appear later than the existing one, we only need to order the new ones.
  if (!new_candidates.empty()) {
    junction_drawing_order_sort(new_candidates);
  }
}

void propose_candidates_final(span<const Stroke> strokes, const StrokeGraph& final_graph,
                              const std::vector<Junction>& final_predictions,
                              const GraphState& in_state, size_t cur_stroke_step,
                              std::vector<Junction>& new_candidates) {
  const StrokeGraph& stroke_graph = in_state.graph_;
  IncrementalCache cache = in_state.cache_;

  // 1. Get the unique set of final-view candidates visible at the current step
  std::vector<Junction> new_candidates1;
  new_candidates1.reserve(final_predictions.size());
  std::set<std::tuple<StrokeTime, StrokeTime, JunctionType::Type>> seen_junctions;
  for (const auto& junc : final_predictions) {
    if (std::max(junc.points[0].first, junc.points[1].first) > cur_stroke_step)
      continue;

    if ((!include_ts && junc.type == JunctionType::T) ||
        is_corner_candidate(stroke_graph, junc) || has_bridge_vertex(stroke_graph, junc))
      continue;

    // TODO: limit to only the candidates related to the newest stroke
    Junction junc_sort = junc;
    if (junc_sort.type == JunctionType::R)
      junc_sort.sort_entries();
    auto junc_key =
      std::make_tuple(junc_sort.points[0], junc_sort.points[1], junc_sort.type);
    if (seen_junctions.count(junc_key))
      continue;
    seen_junctions.emplace(junc_key);
    new_candidates1.emplace_back(junc_sort);
  }

  // 2. Update junctions: Predict and filter, Reproject T-junctions
  std::vector<Junction> new_candidates1_updated;
  update_disconnected_junction_predictions(stroke_graph, new_candidates1,
                                           prediction_feature_type);
  // We pre-filter out prob = 0 junctions so we don't add unnecessary junctions in the
  // later graph completion
  for (auto& junc : new_candidates1) {
    // This would only happen if both ends get dissolved
    if (junc.type == JunctionType::R || reproject_t_junc(in_state.graph_, junc)) {
      // TODO: Use 0 instead latter?
      if (junc.probability < 1e-4)
        continue;
      new_candidates1_updated.emplace_back(junc);
    }
  }

  std::vector<std::pair<Junction, bool>> new_candidates2;
  {
    std::vector<Junction> seen_junc;
    new_candidates2.reserve(new_candidates1_updated.size());
    for (auto& junc : new_candidates1_updated) {
      size_t dangling_orig_sid = junc.points[1].first;
      if (std::find(seen_junc.begin(), seen_junc.end(), junc) != seen_junc.end() ||
          is_corner_candidate(stroke_graph, junc) ||
          has_bridge_vertex(stroke_graph, junc))
        continue;
      new_candidates2.emplace_back(junc, false);
      seen_junc.emplace_back(junc);
    }
    // print_junctions(seen_junc);
  }

  // 3. Group candidates belonging to the same high-valence vertex
  std::vector<std::vector<Junction>> all_new_candidate_sets;
  std::vector<Junction> bind_junc;
  std::vector<Junction> add_bind_junc;
  for (auto& [junc, seen] : new_candidates2) {
    if (seen)
      continue;
    bind_junc.clear();
    add_bind_junc.clear();

    size_t dangling_orig_sid = junc.points[1].first;

    expand_junction(in_state.graph_, junc, bind_junc);

    // Update seen flags and also verify there's no overlapping between bound groups
    for (auto& junc_bind : bind_junc) {
      if (junc_bind.type == JunctionType::R)
        junc_bind.sort_entries();
      for (auto& [junc2, seen2] : new_candidates2) {
        if (!(junc2 == junc_bind))
          continue;
        force_assert(!seen2);
        seen2 = true;
        break;
      }
    }

    for (auto& junc2 : bind_junc) {
      auto junc3 = junc2;
      if (junc3.type == JunctionType::T) {
        auto ok = reproject_t_junc(final_graph, junc3);
        // force_assert(ok && "couldn't reproject");
        if (!ok || is_corner_candidate(final_graph, junc3) ||
            has_bridge_vertex(final_graph, junc3)) {
          continue;
        }
      }
      junc2.orig_dist = junction_distance_init(final_graph, junc3);
      add_bind_junc.emplace_back(junc2);
    }
    all_new_candidate_sets.emplace_back(add_bind_junc);
  }

  // 4. Check if they are valid
  for (size_t i = 0; i < all_new_candidate_sets.size(); ++i) {
    const auto& all_new_candidates = all_new_candidate_sets[i];
    // TODO: change to use a single boolean. The current version is for debugging.
    std::vector<bool> junc_valid;
    for (const auto& junc : all_new_candidates) {
      junc_valid.emplace_back(is_candidate_valid(
        stroke_graph, strokes, final_graph, cur_stroke_step, junc, final_predictions));
    }

    // Bind the candidates from the same expansion together
    if (std::all_of(junc_valid.begin(), junc_valid.end(), [](bool v) { return v; })) {
      for (const auto& junc : all_new_candidates) {
        // It's possible to get duplicate variables due to the expansion
        if (std::find(new_candidates.begin(), new_candidates.end(), junc) ==
            new_candidates.end()) {
          new_candidates.emplace_back(junc);
        }
      }
    }
  }

  print_junctions(new_candidates);

  // 5. Order the junctions based on the drawing order. Since the new junctions must
  // appear later than the existing one, we only need to order the new ones.
  if (!new_candidates.empty()) {
    junction_drawing_order_sort(new_candidates);
  }
}

void complete_graph_candidates(const StrokeGraph& stroke_graph,
                               span<const Stroke> strokes, const StrokeGraph& final_graph,
                               size_t cur_stroke_step,
                               const std::vector<Junction>& to_complete_candidates,
                               std::vector<Junction>& new_candidates) {
  complete_candidate_graph(stroke_graph, to_complete_candidates, new_candidates);
  for (size_t i = 0; i < new_candidates.size(); ++i) {
    auto& junc = new_candidates[i];
    if (!is_weak_candidate_valid(stroke_graph, strokes, final_graph, cur_stroke_step,
                                 junc)) {
      junc.must_disconnect = true;
    }
  }
}

void update_candidate_record(const StrokeGraph& stroke_graph,
                             const std::vector<Junction>& in_candidates,
                             const std::vector<bool>& in_connectivity,
                             const std::vector<Junction>& new_candidates,
                             std::vector<Junction>& varying_candidates,
                             std::vector<Junction>& out_candidates,
                             std::vector<bool>& out_connectivity) {
  // 1. Assemble a varying set: previously disconnected candidates + new candidates
  varying_candidates.reserve(in_candidates.size() + new_candidates.size());
  for (size_t i = 0; i < in_candidates.size(); ++i) {
    if (in_connectivity[i])
      continue;
    varying_candidates.emplace_back(in_candidates[i]);
  }
  size_t new_varying_pos = varying_candidates.size();

  // Predict given the current graph and store the probabilities
  varying_candidates.insert(varying_candidates.end(), new_candidates.begin(),
                            new_candidates.end());
  update_disconnected_junction_predictions(stroke_graph, varying_candidates,
                                           prediction_feature_type);

  // 2. Make the new candidate record for the next round
  out_candidates.insert(out_candidates.end(), in_candidates.begin(), in_candidates.end());
  out_connectivity.insert(out_connectivity.end(), in_connectivity.begin(),
                          in_connectivity.end());

  // Attach the ones not in the record already
  for (size_t i = new_varying_pos; i < varying_candidates.size(); ++i) {
    out_candidates.emplace_back(varying_candidates[i]);
    out_connectivity.emplace_back(false);
  }
}

static void lookup_connection_position(const StrokeGraph& stroke_graph,
                                       const Junction& junc, StrokeGraph::VertexView& v,
                                       StrokeTime& stroke_t,
                                       StrokeTime& stroke_attach_t) {
  StrokeTime s_t = (std::abs(junc.points[0].second) < 1e-10) ||
                       (std::abs(1 - junc.points[0].second) < 1e-10)
                     ? junc.points[0]
                     : junc.points[1];
  StrokeTime attach_s_t = (s_t == junc.points[0]) ? junc.points[1] : junc.points[0];

  // For R junctions, we snap the newer vertex to the older vertex
  if (junc.type == JunctionType::R) {
    s_t = (junc.points[0].first < junc.points[1].first) ? junc.points[1] : junc.points[0];
    attach_s_t = (s_t == junc.points[0]) ? junc.points[1] : junc.points[0];
  }

  // Find the endpoint vertex
  v = orig2endpoint(stroke_graph, s_t);

  // When it's an end-end junction, make sure v is dangling
  if (!v.is_valid() || !v.is_active()) {
    v = StrokeGraph::VertexView();
    s_t = (s_t == junc.points[0]) ? junc.points[1] : junc.points[0];
    attach_s_t = (s_t == junc.points[0]) ? junc.points[1] : junc.points[0];

    v = orig2endpoint(stroke_graph, s_t);
  }

  // Convert to stroke arclength
  stroke_t = s_t;
  stroke_attach_t = attach_s_t;
  const auto ok = convert_orig2strokes(stroke_graph, stroke_attach_t);
  force_assert(ok && "couldn't convert from orig to strokes");
  const auto s_ok = convert_orig2strokes(stroke_graph, stroke_t);
  force_assert(s_ok && "couldn't convert from orig to strokes");
  stroke_graph.strokes_[stroke_attach_t.first].ensure_arclengths();
  stroke_graph.strokes_[stroke_t.first].ensure_arclengths();
}

void expand_trial_candidate(const StrokeGraph& graph, span<const Junction> candidates,
                            const size_t idx, std::set<size_t>& binding) {
  std::vector<Junction> bind_junc;
  expand_junction(graph, candidates[idx], bind_junc);

  StrokeGraph tmp_graph = graph.clone();

  std::set<size_t> dedup_binding;
  dedup_binding.insert(idx);
  for (const auto& junc : bind_junc) {
    // Find
    auto junc_itr =
      std::find_if(candidates.begin(), candidates.end(), [&junc](const Junction& junc_) {
        return (junc_.type == junc.type) &&
               (junc_.points[0].first == junc.points[0].first) &&
               (std::abs(junc_.points[0].second - junc.points[0].second) < 1e-10) &&
               (junc_.points[1].first == junc.points[1].first) &&
               (std::abs(junc_.points[1].second - junc.points[1].second) < 1e-10);
      });
    if (junc_itr != candidates.end()) {
      auto i = junc_itr - candidates.begin();
      dedup_binding.emplace(i);
    }
  }
  for (const auto i : dedup_binding) {
    const auto& junc = candidates[i];

    // Make sure this assignment is possible
    StrokeGraph::VertexView v;
    StrokeTime stroke_t, stroke_attach_t;
    lookup_connection_position(tmp_graph, junc, v, stroke_t, stroke_attach_t);

    StrokeGraph::VertexView new_vertex;
    add_vertex(tmp_graph, stroke_attach_t.first,
               stroke_attach_t.second *
                 tmp_graph.strokes_[stroke_attach_t.first].length(),
               new_vertex);

    // Fails to snap. This is an invalid assignment
    if (!new_vertex.is_valid() ||
        (v != new_vertex && !is_snap_valid(tmp_graph, v.index_, new_vertex.index_))) {
      binding.clear();
      return;
    }
    std::unordered_map<size_t, size_t> local_snapped_connections;
    build_snapped_connections(v, local_snapped_connections);
    build_snapped_connections(new_vertex, local_snapped_connections);

    if (v != new_vertex && !tmp_graph.snap_vertices(v.index_, new_vertex.index_,
                                                    tmp_graph.snapping_method_type_)) {
      binding.clear();
      return;
    }

    // Check if this state is possible (if we want to keep extending this state).
    // A state with missing connections on a cycle is possible (since it may get
    // extended later in the solve) while a state with an end node with valence > 2 is
    // impossible.
    if (!is_assignment_possible(junc, new_vertex, local_snapped_connections))
      continue;

    binding.insert(i);
  }
}

bool implies_interior_interior(const StrokeGraph& stroke_graph,
                               const std::vector<Junction>& connected_candidates) {
  // 1. Find connected components
  std::vector<Junction> varying_candidates = connected_candidates;
  decompose_candidates(stroke_graph, varying_candidates);

  // 2. See if any interior-interior connection is needed.
  std::unordered_map<int, std::vector<Junction>> junc_component;
  for (const auto& junc : varying_candidates) {
    if (!junc_component.count(junc.component_idx))
      junc_component[junc.component_idx] = std::vector<Junction>();
    junc_component[junc.component_idx].emplace_back(junc);
  }

  for (const auto& [c, juncs] : junc_component) {
    std::unordered_set<size_t> high_valence_vid;
    std::vector<StrokeTime> ends;
    for (const auto& junc : juncs) {
      if (junc.type == JunctionType::R) {
        auto vid0 = end_vid(stroke_graph, junc.points[0]);
        auto vid1 = end_vid(stroke_graph, junc.points[1]);
        if (stroke_graph.vertex(vid0).valence() > 1)
          high_valence_vid.emplace(vid0);
        if (stroke_graph.vertex(vid1).valence() > 1)
          high_valence_vid.emplace(vid1);
        if (high_valence_vid.size() > 1)
          return true;
      } else {
        const auto itr =
          std::find_if(ends.begin(), ends.end(), [junc](const StrokeTime& end) {
            return (end.first == junc.points[0].first) &&
                   (std::abs(end.second - junc.points[0].second) <
                    std::numeric_limits<Float>::epsilon());
          });
        if (itr == ends.end())
          ends.emplace_back(junc.points[0]);
      }
    }

    if (ends.size() > 1)
      return true;
    if (!high_valence_vid.empty() && !ends.empty())
      return true;
  }

  return false;
}

bool exists_weak_only_connection(const StrokeGraph& stroke_graph,
                                 const std::vector<Junction>& connected_candidates) {
  // 1. Find connected components
  std::vector<Junction> varying_candidates = connected_candidates;
  decompose_candidates(stroke_graph, varying_candidates);

  // 2. See if any interior-interior connection is needed.
  std::unordered_map<int, std::vector<Junction>> junc_component;
  for (const auto& junc : varying_candidates) {
    if (!junc_component.count(junc.component_idx))
      junc_component[junc.component_idx] = std::vector<Junction>();
    junc_component[junc.component_idx].emplace_back(junc);
  }

  for (const auto& [c, juncs] : junc_component) {
    bool seen_non_weak = false;
    for (const auto& junc : juncs) {
      if (!junc.is_weak) {
        seen_non_weak = true;
      }
    }

    if (!seen_non_weak)
      return true;
  }

  return false;
}

bool is_junction_on_cycle(const StrokeGraph::VertexView v, const Junction& candidate) {
  // Collect the star at v
  std::unordered_map<size_t, size_t> star_indexing;
  std::unordered_map<size_t, size_t> star_count;
  build_high_valence_star(v, std::unordered_map<size_t, size_t>(), star_indexing,
                          star_count);

  Junction junc = candidate;
  bool ok = original_stroke_to_stroke_indexing(*v.graph_, junc);
  force_assert(ok && "couldn't map from orig to strokes");

  // TODOs: Actually make sure the stroke position is at its end. Note the current problem
  // is introduced by dropping short end substroke in add_stroke_incremental_topological.
  // It updates the mapping but the old connected junctions still have the original
  // positions.
  size_t s1_hash;
  size_t s2_hash;
  int s1_hash2;
  int s2_hash2;
  bool cont_both_sides =
    lookup_junc_hash(v, candidate, s1_hash, s2_hash, s1_hash2, s2_hash2);

  assert((!is_t_positioned(candidate) && star_indexing.count(s1_hash) &&
          star_indexing.count(s2_hash)) ||
         (is_t_positioned(candidate) &&
            ((star_indexing.count(s1_hash) && star_indexing.count(s2_hash)) ||
             (star_indexing.count(s1_hash2) && star_indexing.count(s2_hash))) ||
          (cont_both_sides && star_indexing.count(s2_hash2))));

  if (!adj_test(star_indexing, s1_hash, s2_hash) &&
      !(is_t_positioned(candidate) && adj_test(star_indexing, s1_hash2, s2_hash)) &&
      !(cont_both_sides && (adj_test(star_indexing, s1_hash2, s2_hash2) ||
                            adj_test(star_indexing, s1_hash, s2_hash2))))
    return false;

  return true;
}

bool is_assignment_valid(span<const Junction> candidates,
                         const std::vector<bool>& junction_connected,
                         const StrokeGraph::VertexView v) {
  // Collect the star at v
  std::unordered_map<size_t, size_t> star_indexing;
  std::unordered_map<size_t, size_t> star_count;
  build_high_valence_star(v, std::unordered_map<size_t, size_t>(), star_indexing,
                          star_count);

  // Register the connections
  int seed_vi = -1;
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (!junction_connected[i])
      continue;

    Junction junc = candidates[i];

    // Check if this junction is at this vertex
    bool is_at_vertex = false;
    for (const auto& vid : v.vertex_ids()) {
      if (vid.repr() == junc.repr) {
        is_at_vertex = true;
        break;
      }
    }
    if (!is_at_vertex)
      continue;

    seed_vi = i;

    bool ok = original_stroke_to_stroke_indexing(*v.graph_, junc);
    force_assert(ok && "couldn't map from orig to strokes");
    assert(
      (junc.type == JunctionType::R &&
       (std::abs(junc.points[0].second) < 1e-10 ||
        std::abs(1.0 - junc.points[0].second) < 1e-10) &&
       (std::abs(junc.points[1].second) < 1e-10 ||
        std::abs(1.0 - junc.points[1].second) < 1e-10)) ||
      (junc.type == JunctionType::T && (std::abs(junc.points[1].second) < 1e-10 ||
                                        std::abs(1.0 - junc.points[1].second) < 1e-10)));

    size_t s1_hash;
    size_t s2_hash;
    int s1_hash2;
    int s2_hash2;
    bool cont_both_sides =
      lookup_junc_hash(v, candidates[i], s1_hash, s2_hash, s1_hash2, s2_hash2);

    if (!adj_test(star_indexing, s1_hash, s2_hash)) {
      if (is_t_positioned(candidates[i]) && adj_test(star_indexing, s1_hash2, s2_hash))
        s1_hash = s1_hash2;
      else if (cont_both_sides && adj_test(star_indexing, s1_hash, s2_hash2))
        s2_hash = s2_hash2;
      else if (cont_both_sides && adj_test(star_indexing, s1_hash2, s2_hash2)) {
        s1_hash = s1_hash2;
        s2_hash = s2_hash2;
      } else
        return false;
    }

    // Can only connect adjacent strokes
    if (!adj_test(star_indexing, s1_hash, s2_hash))
      return false;

    star_count[s1_hash]++;
    star_count[s2_hash]++;

    // Check local connectivity topology. We can't have more than two-way connections.
    if (star_count[s1_hash] > 2 || star_count[s2_hash] > 2)
      return false;
  }

  // Check the overall topology
  // We can a chain or multiple chains. As long as, we don't have the candidate between
  // multiple components / the two ends of a chain.
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (junction_connected[i])
      continue;
    Junction junc = candidates[i];
    bool ok = original_stroke_to_stroke_indexing(*v.graph_, junc);
    force_assert(ok && "couldn't map from orig to strokes");

    // This can be added an existing connected junction.
    if (!((std::abs(junc.points[0].second) < 1e-10 ||
           std::abs(1.0 - junc.points[0].second) < 1e-10) &&
          (std::abs(junc.points[1].second) < 1e-10 ||
           std::abs(1.0 - junc.points[1].second) < 1e-10)))
      continue;

    size_t s1_hash;
    size_t s2_hash;
    int s1_hash2;
    int s2_hash2;
    bool cont_both_sides =
      lookup_junc_hash(v, candidates[i], s1_hash, s2_hash, s1_hash2, s2_hash2);

    auto is_cycle_connectable = [&star_indexing, &star_count](int s1_hash,
                                                              int s2_hash) -> bool {
      if (s1_hash < 0 || s2_hash < 0)
        return false;
      // This is either not fully in this connected junction or not on the cycle.
      if (!(star_indexing.count(s1_hash) && star_indexing.count(s2_hash)) ||
          ((star_indexing[s1_hash] + 1) % star_indexing.size() !=
             star_indexing[s2_hash] &&
           (star_indexing[s2_hash] + 1) % star_indexing.size() != star_indexing[s1_hash]))
        return false;

      // Can't see a potential linking pair of the cycle that is not connected.
      if (star_count[s1_hash] <= 1 && star_count[s2_hash] <= 1)
        return true;

      return false;
    };

    if (is_cycle_connectable(s1_hash, s2_hash) ||
        (is_t_positioned(candidates[i]) && is_cycle_connectable(s1_hash2, s2_hash)) ||
        (cont_both_sides && (is_cycle_connectable(s1_hash2, s2_hash2) ||
                             is_cycle_connectable(s1_hash, s2_hash2))))
      return false;
  }

  // Extra catch. Since we may use a snapping in the place of an actual connection that
  // should be connected but not, do this binding check explicitly. Check if all expanded
  // candidates are connected.
  // The above check should be sufficient to determine if we have any redundant
  // connections.
  assert(seed_vi >= 0);

  std::set<size_t> binding;
  expand_trial_candidate(*v.graph_, candidates, seed_vi, binding);

  // This is either unable to be snapped or is topologically invalid
  if (!binding.count(seed_vi))
    return false;

  for (const auto idx : binding) {
    size_t s1_hash;
    size_t s2_hash;
    int s1_hash2;
    int s2_hash2;
    bool cont_both_sides =
      lookup_junc_hash(v, candidates[idx], s1_hash, s2_hash, s1_hash2, s2_hash2);

    if (!adj_test(star_indexing, s1_hash, s2_hash)) {
      if (candidates[idx].type == JunctionType::T &&
          adj_test(star_indexing, s1_hash2, s2_hash))
        s1_hash = s1_hash2;
      else if (cont_both_sides && adj_test(star_indexing, s1_hash, s2_hash2))
        s2_hash = s2_hash2;
      else if (cont_both_sides && adj_test(star_indexing, s1_hash2, s2_hash2)) {
        s1_hash = s1_hash2;
        s2_hash = s2_hash2;
      } else
        continue;
    }

    // Can only connect adjacent strokes
    if (!adj_test(star_indexing, s1_hash, s2_hash))
      continue;

    // We are missing a connection
    if (!junction_connected.at(idx))
      return false;
  }

  return true;
}

std::string face_id(const StrokeGraph::FaceView f) {
  // There may be multiple cycles, sort their individual IDs lexicographically
  std::vector<std::string> face_ids;
  face_ids.reserve(f.cycles().size());
  constexpr bool include_interior = true;
  for (const auto hi : f.cycles()) {
    size_t init_min = StrokeGraph::invalid;
    size_t junc_min = StrokeGraph::invalid;

    // Sort the vertex IDs along the cycle by assigning the first one to be either: 1) the
    // init vertex with min ID number; or 2) the junc vertex with min ID number (if init
    // vertices don't exist).
    std::vector<std::vector<StrokeGraph::VertexID>> combined_cycle_ids;
    const auto he = f.graph_->hedge(hi);
    auto it = he;
    auto iterations = 0;
    do {
      // Ignore the strokes within the region
      MSVC_WARNING_SUPPRESS(4127);
      if (!include_interior && !combined_cycle_ids.empty() &&
          combined_cycle_ids.back().front() == it.dest().vertex_ids().front()) {
        combined_cycle_ids.pop_back();
      } else {
        combined_cycle_ids.emplace_back();
        for (auto const& vid : it.origin().vertex_ids()) {
          combined_cycle_ids.back().emplace_back(vid);
        }

        // Add the dissolved vertices
        if (it.forward())
          for (const auto& [t, vid] : f.graph_->strokes2vid_[it.stroke_idx()]) {
            combined_cycle_ids.back().emplace_back(vid);
          }
        else
          for (auto itr = f.graph_->strokes2vid_[it.stroke_idx()].rbegin();
               itr != f.graph_->strokes2vid_[it.stroke_idx()].rend(); ++itr) {
            combined_cycle_ids.back().emplace_back(itr->second);
          }
      }

      it = it.next();
      iterations++;
      assert(iterations < 1024 && "likely infinite loop found");
    } while (it != he);

    std::vector<StrokeGraph::VertexID> cycle_ids;
    for (const auto& ids : combined_cycle_ids) {
      for (const auto& id : ids) {
        cycle_ids.emplace_back(id);
      }
    }
    for (const auto& ids : cycle_ids) {
      if (ids.connection_type_ == StrokeGraph::VertexID::Type::Initialization) {
        init_min = std::min(init_min, ids.connection_index_);
      } else if (ids.connection_type_ == StrokeGraph::VertexID::Type::Junction) {
        junc_min = std::min(junc_min, ids.connection_index_);
      }
    }

    MSVC_WARNING_SUPPRESS(4127);
    if (!include_interior && init_min == StrokeGraph::invalid &&
        junc_min == StrokeGraph::invalid)
      continue;

    if (init_min == StrokeGraph::invalid && junc_min == StrokeGraph::invalid)
      return "";

    StrokeGraph::VertexID first_vertex_id =
      (init_min != StrokeGraph::invalid)
        ? StrokeGraph::VertexID({StrokeGraph::VertexID::Type::Initialization, init_min})
        : StrokeGraph::VertexID({StrokeGraph::VertexID::Type::Junction, junc_min});

    // Move to the start position
    size_t start_cyc_i = 0;
    for (; start_cyc_i != cycle_ids.size(); ++start_cyc_i) {
      if (cycle_ids[start_cyc_i] == first_vertex_id)
        break;
    }

    size_t cyc_i = start_cyc_i;
    std::string sorted_cycle_id_token = "";
    do {
      sorted_cycle_id_token += cycle_ids[cyc_i].repr() + ",";
      cyc_i = (cyc_i + 1) % cycle_ids.size();
    } while (cyc_i != start_cyc_i);

    face_ids.emplace_back(sorted_cycle_id_token);
  }

  std::sort(face_ids.begin(), face_ids.end());
  std::string f_id = "";
  for (auto const& s : face_ids) {
    f_id += s + ";";
  }

  return f_id;
}

Float face_max_stroke_width(const StrokeGraph::FaceView f) {
  Float max_width = 0;
  for (const auto hi : f.cycles()) {
    std::vector<size_t> s_indices;

    size_t iterations = 0;
    const auto he = f.graph_->hedge(hi);
    auto it = he;
    do {
      if (!(it.flags() & StrokeGraph::HedgeRecord::Bridge) &&
          it.stroke_idx() < f.graph_->strokes2orig_.size())
        max_width =
          std::max(f.graph_->orig_strokes_[it.orig_stroke_idx()].pen_width(), max_width);
      it = it.next();
      iterations++;
      assert(iterations < 1024 && "likely infinite loop found");
    } while (it != he);
  }

  return max_width;
}

Float junction_distance_init(const StrokeGraph& stroke_graph, const Junction& junc) {
  StrokeTime orig_stroke_t = junc.points[1], orig_stroke_attach_t = junc.points[0];
  stroke_graph.orig_strokes_[orig_stroke_t.first].ensure_arclengths();
  stroke_graph.orig_strokes_[orig_stroke_attach_t.first].ensure_arclengths();
  features::AbsEnvelopeDistance abs_envelope;
  auto s1_idx = stroke_graph.orig_strokes_[orig_stroke_t.first].fractional_index(
    orig_stroke_t.second * stroke_graph.orig_strokes_[orig_stroke_t.first].length());
  auto s2_idx = stroke_graph.orig_strokes_[orig_stroke_attach_t.first].fractional_index(
    orig_stroke_attach_t.second *
    stroke_graph.orig_strokes_[orig_stroke_attach_t.first].length());
  Float env_dist = abs_envelope(
    stroke_graph.orig_strokes_[orig_stroke_t.first],
    orig_stroke_t.second * stroke_graph.orig_strokes_[orig_stroke_t.first].length(),
    s1_idx.first, stroke_graph.orig_strokes_[orig_stroke_attach_t.first],
    orig_stroke_attach_t.second *
      stroke_graph.orig_strokes_[orig_stroke_attach_t.first].length(),
    s2_idx.first);
  env_dist = std::max(env_dist, 0.0);

  return env_dist;
}

Float junction_distance(const StrokeGraph& stroke_graph, const Junction& junc) {
  force_assert(junc.orig_dist >= 0);
  return junc.orig_dist;
}

std::unique_ptr<StrokeGraph>
modify_graph(const StrokeGraph& stroke_graph, span<Junction> candidates,
             const std::vector<bool>& junction_connected,
             std::vector<std::pair<size_t, size_t>>& adj_faces,
             std::vector<Float>& junc_distances,
             std::vector<StrokeGraph::VertexID>& junc_vertices,
             const StrokeGraph::SnappingType snapping_type) {
  adj_faces.reserve(candidates.size());
  junc_distances.reserve(candidates.size());
  junc_vertices.reserve(candidates.size());
  std::map<size_t, StrokeGraph::VertexID> sort_junc_vertices;

  for (size_t i = 0; i < candidates.size(); ++i) {
    if (!junction_connected[i])
      continue;
    Junction& junc = candidates[i];
    junc.repr.clear();
  }

  // Decide the connection order so that junctions belonging to the same connected
  // component are connected consecutively
  std::vector<size_t> connected_indices;
  std::vector<size_t> component_labels;
  group_connected_junctions(stroke_graph, candidates, junction_connected,
                            connected_indices, component_labels);

  std::map<size_t, std::vector<size_t>> group_indices;
  for (size_t i = 0; i < connected_indices.size(); ++i) {
    if (!group_indices.count(component_labels[i]))
      group_indices[component_labels[i]] = std::vector<size_t>();
    group_indices[component_labels[i]].emplace_back(connected_indices[i]);
  }

  auto out_graph = std::unique_ptr<StrokeGraph>();
  for (const auto& [c, indices] : group_indices) {
    Vec2 snap_pos = Vec2::Empty();
    bool uninit = true;
    bool seen_interior = false;
    std::map<size_t, StrokeTime> snap_ends;

    // TODO: Better strategy to determine the snapping position
    // First see if there's any interior position
    for (const auto i : indices) {
      const Junction& junc = candidates[i];
      int end_d0 = end_degree(stroke_graph, junc.points[0]);
      if (junc.type == JunctionType::R && end_d0 == 1) {
        continue;
      }
      snap_ends[junc.points[0].first] = junc.points[0];
      seen_interior = true;
    }
    if (snap_ends.empty()) {
      for (const auto i : indices) {
        const Junction& junc = candidates[i];
        snap_ends[junc.points[0].first] = junc.points[0];
        // Snap the tail to the head
        if (!snap_ends.count(junc.points[1].first) ||
            junc.points[1].second < snap_ends[junc.points[1].first].second)
          snap_ends[junc.points[1].first] = junc.points[1];
      }
    }

    if (snap_ends.size() == 1) {
      auto end = snap_ends.begin()->second;
      if (convert_orig2strokes(stroke_graph, end)) {
        Vec2 p = stroke_graph.strokes_[end.first].pos_norm(end.second);
        snap_pos = p;
        uninit = false;
      }
    } else {
      // For high-valence vertices, try finding an end visible to all other ends
      bool found = false;
      bool blocked = false;
      for (const auto& [sid, s_end] : snap_ends) {
        StrokeTime end = s_end;
        if (convert_orig2strokes(stroke_graph, end)) {
          Vec2 test_p = stroke_graph.strokes_[end.first].pos_norm(end.second);
          for (const auto i : indices) {
            Junction s_junc = candidates[i];
            if (!original_stroke_to_stroke_indexing(stroke_graph, s_junc)) {
              return nullptr;
            }
            Vec2 p = stroke_graph.strokes_[s_junc.points[0].first].pos_norm(
              s_junc.points[0].second);
            Vec2 q = stroke_graph.strokes_[s_junc.points[1].first].pos_norm(
              s_junc.points[1].second);
            bool p_visible =
              (p == test_p || line_of_sight(p, test_p, stroke_graph.strokes_,
                                            stroke_graph.bvh_.centerline_bbs()));
            bool q_visible =
              (q == test_p || line_of_sight(q, test_p, stroke_graph.strokes_,
                                            stroke_graph.bvh_.centerline_bbs()));
            if (!(p_visible && q_visible)) {
              blocked = true;
              break;
            }
          }

          if (!blocked) {
            snap_pos = test_p;
            uninit = false;
            found = true;
            break;
          }
        }
        if (found)
          break;
      }
    }

    // Actually connect
    for (auto i : indices) {
      Junction& junc = candidates[i];

      Junction s_junc = junc;
      if (!original_stroke_to_stroke_indexing(stroke_graph, s_junc)) {
        return nullptr;
      }
      Vec2 p =
        stroke_graph.strokes_[s_junc.points[0].first].pos_norm(s_junc.points[0].second);
      Vec2 q =
        stroke_graph.strokes_[s_junc.points[1].first].pos_norm(s_junc.points[1].second);
      // This would form sliver or is currently blocked by connections made earlier
      if (!is_snap_valid(stroke_graph, junc) ||
          (uninit && !line_of_sight(p, q, stroke_graph.strokes_,
                                    stroke_graph.bvh_.centerline_bbs())))
        return nullptr;

      if (!uninit) {
        if (p != snap_pos && !line_of_sight(p, snap_pos, stroke_graph.strokes_,
                                            stroke_graph.bvh_.centerline_bbs())) {
          return nullptr;
        }
        if (q != snap_pos && !line_of_sight(q, snap_pos, stroke_graph.strokes_,
                                            stroke_graph.bvh_.centerline_bbs())) {
          return nullptr;
        }
      }

      if (!out_graph) {
        // Can't delay it for much longer.  We've got to copy the graph.  :(
        out_graph = std::make_unique<StrokeGraph>();
        *out_graph = stroke_graph.clone();

        // Avoid rebuilding faces unless it's the last connection to save time.
        out_graph->faces_.clear();
        for (size_t j = 0; j < out_graph->hedges_.size(); ++j) {
          out_graph->hedges_[j].face_ = StrokeGraph::invalid;
        }
      }

      // TODO: We no longer need to redo this lookup every time since we are no longer
      //       changing stroke indexing in this loop.
      StrokeGraph::VertexView v;
      StrokeTime stroke_t, stroke_attach_t;
      lookup_connection_position(*out_graph, junc, v, stroke_t, stroke_attach_t);

      // Find the adjacent stroke
      assert(v.is_valid() && v.is_active());
      size_t adj_orig_s = v.hedge().orig_stroke_idx();

      // Either look up or insert the vertices. And before connecting, save the local
      // snapped connectivity.
      StrokeGraph::VertexView new_vertex;
      add_vertex_precise(*out_graph, stroke_attach_t.first,
                         stroke_attach_t.second *
                           out_graph->strokes_[stroke_attach_t.first].length(),
                         new_vertex);
      if (new_vertex.is_valid() && v != new_vertex) {
        // Modify the type of the vertex based on the snap position
        // TODO: For high-valence junctions this may be wrong, but I'm not sure what
        // downstream code expects here.  Revisit if we actually need this information.
        out_graph->vertices_[v.index_].junc_type_ =
          (new_vertex.is_dangling() ? JunctionType::R : JunctionType::T);
      }

      // Fails to snap. This is an invalid assignment
      if (!new_vertex.is_valid() ||
          (v != new_vertex && !is_snap_valid(*out_graph, v.index_, new_vertex.index_))) {
        return nullptr;
      }
      std::unordered_map<size_t, size_t> local_snapped_connections;
      build_snapped_connections(v, local_snapped_connections);
      build_snapped_connections(new_vertex, local_snapped_connections);

      const Vec2* snap_ptr = &snap_pos;
      if (uninit || seen_interior) {
        snap_ptr = nullptr;
      }

      if (v != new_vertex && !out_graph->snap_vertices(v.index_, new_vertex.index_,
                                                       snapping_type, -1, snap_ptr))
        return nullptr;

      // Check if this state is possible (if we want to keep extending this state).
      // A state with missing connections on a cycle is possible (since it may get
      // extended later in the solve) while a state with an end node with valence > 2 is
      // impossible.
      if (!is_assignment_possible(junc, new_vertex, local_snapped_connections))
        return nullptr;

      // Find the next valid junction index
      int new_id = -1;
      for (size_t j = 0; j < out_graph->vertices_.size(); ++j) {
        for (auto const& vid : out_graph->vertices_[j].ids_) {
          if (vid.connection_type_ == StrokeGraph::VertexID::Junction)
            new_id = std::max(new_id, (int)vid.connection_index_);
        }
      }
      const auto& vid = out_graph->vertices_[new_vertex.index_].ids_.emplace_back(
        StrokeGraph::VertexID{StrokeGraph::VertexID::Junction, (size_t)++new_id});
      junc.repr = vid.repr();

      sort_junc_vertices[i] = new_vertex.vertex_ids().back();
    }
  }

  for (const auto& [i, vid] : sort_junc_vertices) {
    junc_vertices.emplace_back(vid);
  }

  if (out_graph) {
    construct_faces(*out_graph);
  } else {
    out_graph = std::make_unique<StrokeGraph>();
    *out_graph = stroke_graph.clone();
  }

  // Find the two adjacent faces of the new connections
  const StrokeGraph& cur_graph = (out_graph) ? *out_graph : stroke_graph;
  junc_distances.clear();
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (!junction_connected[i])
      continue;
    Float env_dist = junction_distance(stroke_graph, candidates[i]);
    junc_distances.emplace_back(env_dist);
    std::unordered_set<size_t> adj_fi;
    for (size_t fi = 0; fi < cur_graph.faces_.size(); ++fi) {
      bool found = false;
      for (const auto hi : cur_graph.faces_[fi].cycles_) {
        const auto he = cur_graph.hedge(hi);
        auto it = he;
        do {
          for (const auto& [t, vid] : cur_graph.strokes2vid_[it.stroke_idx()]) {
            auto vid_str = vid.repr();
            if (vid_str == candidates[i].repr) {
              adj_fi.emplace(fi);
              found = true;
              break;
            }
          }
          for (const auto& vid : it.origin().vertex_ids()) {
            auto vid_str = vid.repr();
            if (vid_str == candidates[i].repr &&
                is_high_valence_junction_in_region_cycle(
                  cur_graph, candidates, fi, it.origin(), candidates[i].repr)) {
              adj_fi.emplace(fi);
              found = true;
              break;
            }
          }
          if (found)
            break;
          it = it.next();
        } while (it != he);
        if (found)
          break;
      }
    }
    std::vector<size_t> v_final_adj_faces;
    v_final_adj_faces.insert(v_final_adj_faces.end(), adj_fi.begin(), adj_fi.end());
    if (v_final_adj_faces.size() == 2 || v_final_adj_faces.size() == 1)
      adj_faces.emplace_back(v_final_adj_faces.front(), v_final_adj_faces.back());
    else
      return nullptr;
  }

  return out_graph;
}

std::unique_ptr<StrokeGraph>
modify_graph(const StrokeGraph& stroke_graph, span<Junction> candidates,
             const std::vector<bool>& junction_connected,
             std::vector<std::pair<size_t, size_t>>& adj_faces,
             std::vector<Float>& junc_distances,
             std::vector<StrokeGraph::VertexID>& junc_vertices) {
  return modify_graph(stroke_graph, candidates, junction_connected, adj_faces,
                      junc_distances, junc_vertices, stroke_graph.snapping_method_type_);
}

bool euclidean_region_condition_check(Float region_size, Float junc_dist) {
  SPDLOG_DEBUG("{} > {} * {} = {}: {}", region_size, junc_dist, hard_region_junc_ratio,
               junc_dist * hard_region_junc_ratio,
               region_size > junc_dist * hard_region_junc_ratio);
  bool check = (region_size > junc_dist * hard_region_junc_ratio);
  return check;
}

bool euclidean_region_condition_check(
  const StrokeGraph& graph, std::unordered_map<std::string, Float>& region_size_cache,
  size_t fi, const Float check_junc_dist) {
  const auto f_size = get_face_size(graph, region_size_cache, fi, check_junc_dist);
  return euclidean_region_condition_check(f_size, check_junc_dist);
}
bool stroke_width_region_condition_check(
  const StrokeGraph& graph, std::unordered_map<std::string, Float>& region_size_cache,
  size_t fi) {
  Float f_stroke_width = face_stroke_width_min(graph, fi);
  const auto f_size = get_face_size(graph, region_size_cache, fi, f_stroke_width);
  return euclidean_region_condition_check(f_size, f_stroke_width);
}

bool hard_region_condition_check(
  const StrokeGraph& graph, std::unordered_map<std::string, Float>& region_size_cache,
  size_t fi, const Float check_junc_dist) {
  if (hard_constraint_type == HardConstraintType::JunctionDist) {
    return euclidean_region_condition_check(graph, region_size_cache, fi,
                                            check_junc_dist);
  } else {
    return stroke_width_region_condition_check(graph, region_size_cache, fi);
  }
}

bool hard_region_condition_check_const(
  const StrokeGraph& graph,
  const std::unordered_map<std::string, Float>& region_size_cache, size_t fi,
  const Float check_junc_dist) {
  const auto f_size = get_face_size_const(graph, region_size_cache, fi);
  return euclidean_region_condition_check(f_size, check_junc_dist);
}

Float face_maximum_inscribing_circle_radius(const StrokeGraph& stroke_graph,
                                            size_t face_idx, Eigen::Vector2d& center) {
  if (face_idx == stroke_graph.boundary_face_) {
    return std::numeric_limits<Float>::infinity();
  }

  // For now collect strokes
  auto f = stroke_graph.face(face_idx);
  std::vector<std::vector<Stroke>> poly_strokes;
  auto iterations = 0;
  for (const auto hi : f.cycles()) {
    poly_strokes.emplace_back();

    std::vector<size_t> s_indices;

    const auto he = f.graph_->hedge(hi);
    auto it = he;
    do {
      auto s_it = it;
      if (!it.forward())
        s_it = it.twin();

      int orig_sid = -1;
      if (!(s_it.flags() & StrokeGraph::HedgeRecord::Bridge))
        orig_sid = s_it.orig_stroke_idx();
      size_t stroke_idx = s_it.stroke_idx();
      SPDLOG_DEBUG("In cycle: {}, {}, v{} -> v{}", stroke_idx, orig_sid,
                   it.origin().index_, it.dest().index_);

      // Special case of zero area loop. Note this should be removed by small face
      // collapsing.
      if (s_it.origin() == s_it.dest()) {
        std::vector<Vec2> poly = f.graph_->face_positions(f.index_);
        CoordMat poly_coords;
        poly_coords.resize(poly.size(), 2);
        for (size_t i = 0; i < poly.size(); ++i) {
          poly_coords.row(i) << poly[i].x(), poly[i].y();
        }

        auto face_polygon = to_clip_paths(span<const CoordMat>{&poly_coords, 1});
        Float check_area1 = clip_area_scaled(*face_polygon);

        Float check_area2 = 1;
        // This can't be the outside face
        if (it.twin().face_idx() != stroke_graph.boundary_face_) {
          poly = f.graph_->face_positions(it.twin().face_idx());
          poly_coords.resize(poly.size(), 2);
          for (size_t i = 0; i < poly.size(); ++i) {
            poly_coords.row(i) << poly[i].x(), poly[i].y();
          }
          face_polygon = to_clip_paths(span<const CoordMat>{&poly_coords, 1});
          check_area2 = clip_area_scaled(*face_polygon);
        }

        if (check_area1 < std::numeric_limits<Float>::epsilon() ||
            check_area2 < std::numeric_limits<Float>::epsilon()) {
          goto next_iteration;
        }
      }
      // End of the special case

      // Ignore the strokes within the region
      // if (!s_indices.empty() && s_indices.back() == stroke_idx) {
      if (false) {
        poly_strokes.back().pop_back();
        s_indices.pop_back();
      } else {
        // Write stroke index for debugging visualization on python side.
        poly_strokes.back().emplace_back(stroke_graph.strokes_[stroke_idx].clone());
        if (!it.forward())
          poly_strokes.back().back().reverse();
        s_indices.emplace_back(stroke_idx);
      }

next_iteration:
      it = it.next();
      iterations++;
      assert(iterations < 1024 && "likely infinite loop found");
    } while (it != he);

    if (poly_strokes.back().empty()) {
      poly_strokes.pop_back();
    }
  }

  if (poly_strokes.empty())
    return 0;

  // Tune to have precision = 1 based on Dance_lien-ze, Tsao_014
  double max_pen_width = -1;
  for (size_t i = 0; i < poly_strokes.size(); ++i) {
    for (const auto& s : poly_strokes[i]) {
      max_pen_width = std::max(max_pen_width, s.pen_width());
    }
  }
  double precision = max_pen_width / 8;
  if (max_pen_width < 1e-10) {
    // precision == 0 causes an infinite loop.
    // This can happen because of bridge edges.
    return 0.0;
  }

  // Convert to polygons
  mapbox::geometry::polygon<double> polygon;
  for (size_t i = 0; i < poly_strokes.size(); ++i) {
    mapbox::geometry::linear_ring<double> ring;
    for (const auto& s : poly_strokes[i]) {
      // Simplify the polygon to save region computation time
      const auto dec_s = decimated(s, precision);
      for (int j = 0; j < dec_s.size(); ++j)
        ring.push_back({dec_s.x(j), dec_s.y(j)});
    }
    polygon.push_back(std::move(ring));
  }

  if (poly_strokes.empty())
    return 0;

  double radius = mapbox::polylabel_distance(polygon, center.x(), center.y(), precision);

  // Now consider the stroke envelope but from a fixed center point
  radius =
    distance_to_face_envelopes(Vec2(center.x(), center.y()), stroke_graph, face_idx);

  return radius;
}

Float face_maximum_inscribing_circle_radius_clipping(const StrokeGraph& stroke_graph,
                                                     size_t face_idx,
                                                     Eigen::Vector2d& center) {
  if (face_idx == stroke_graph.boundary_face_) {
    return std::numeric_limits<Float>::infinity();
  }

  // For now collect strokes
  auto f = stroke_graph.face(face_idx);
  std::vector<std::vector<Stroke>> poly_strokes;
  auto iterations = 0;
  for (const auto hi : f.cycles()) {
    poly_strokes.emplace_back();

    std::vector<size_t> s_indices;

    const auto he = f.graph_->hedge(hi);
    auto it = he;
    do {
      auto s_it = it;
      if (!it.forward())
        s_it = it.twin();

      int orig_sid = -1;
      if (!(s_it.flags() & StrokeGraph::HedgeRecord::Bridge))
        orig_sid = s_it.orig_stroke_idx();
      size_t stroke_idx = s_it.stroke_idx();
      SPDLOG_DEBUG("In cycle: {}, {}, v{} -> v{}", stroke_idx, orig_sid,
                   it.origin().index_, it.dest().index_);

      // Special case of zero area loop. Note this should be removed by small face
      // collapsing.
      if (s_it.origin() == s_it.dest()) {
        std::vector<Vec2> poly = f.graph_->face_positions(f.index_);
        CoordMat poly_coords;
        poly_coords.resize(poly.size(), 2);
        for (size_t i = 0; i < poly.size(); ++i) {
          poly_coords.row(i) << poly[i].x(), poly[i].y();
        }

        auto face_polygon = to_clip_paths(span<const CoordMat>{&poly_coords, 1});
        Float check_area1 = clip_area_scaled(*face_polygon);

        Float check_area2 = 1;
        // This can't be the outside face
        if (it.twin().face_idx() != stroke_graph.boundary_face_) {
          poly = f.graph_->face_positions(it.twin().face_idx());
          poly_coords.resize(poly.size(), 2);
          for (size_t i = 0; i < poly.size(); ++i) {
            poly_coords.row(i) << poly[i].x(), poly[i].y();
          }
          face_polygon = to_clip_paths(span<const CoordMat>{&poly_coords, 1});
          check_area2 = clip_area_scaled(*face_polygon);
        }

        if (check_area1 < std::numeric_limits<Float>::epsilon() ||
            check_area2 < std::numeric_limits<Float>::epsilon()) {
          goto next_iteration;
        }
      }
      // End of the special case

      // Ignore the strokes within the region
      // if (!s_indices.empty() && s_indices.back() == stroke_idx) {
      if (false) {
        poly_strokes.back().pop_back();
        s_indices.pop_back();
      } else {
        // Write stroke index for debugging visualization on python side.
        poly_strokes.back().emplace_back(stroke_graph.strokes_[stroke_idx].clone());
        if (!it.forward())
          poly_strokes.back().back().reverse();
        s_indices.emplace_back(stroke_idx);
      }

next_iteration:
      it = it.next();
      iterations++;
      assert(iterations < 1024 && "likely infinite loop found");
    } while (it != he);

    if (poly_strokes.back().empty()) {
      poly_strokes.pop_back();
    }
  }

  if (poly_strokes.empty())
    return 0;

  // Tune to have precision = 1 based on Dance_lien-ze, Tsao_014
  double max_pen_width = -1;
  for (size_t i = 0; i < poly_strokes.size(); ++i) {
    for (const auto& s : poly_strokes[i]) {
      max_pen_width = std::max(max_pen_width, s.pen_width());
    }
  }
  double precision = max_pen_width / 8;

  // Convert to polygons
  std::vector<CoordMat> poly_coords;
  std::vector<std::pair<Float, Float>> dec_points;
  for (size_t i = 0; i < poly_strokes.size(); ++i) {
    poly_coords.push_back(CoordMat());
    dec_points.clear();
    for (const auto& s : poly_strokes[i]) {
      // Simplify the polygon to save region computation time
      const auto dec_s = decimated(s, precision);
      for (int j = 0; j < dec_s.size(); ++j) {
        dec_points.emplace_back(dec_s.x(j), dec_s.y(j));
      }
    }
    poly_coords.back().resize(dec_points.size(), 2);
    for (int j = 0; j < dec_points.size(); ++j) {
      poly_coords.back().row(j) << dec_points[j].first, dec_points[j].second;
    }
    if (poly_coords.back().size() == 0)
      poly_coords.pop_back();
  }

  if (poly_strokes.empty())
    return 0;

  // Remove the stroke envelopes
  auto face_polygon = to_clip_paths(poly_coords);

  std::vector<CoordMat> coords;
  for (size_t i = 0; i < poly_strokes.size(); ++i) {
    for (const auto& s : poly_strokes[i]) {
      if (s.size() > 1) {
        outline_to_polygons(s, coords, precision * 2);
      }
    }
  }
  const auto clipped = to_clip_paths(coords);
  auto diff = boolean_difference(*face_polygon, *clipped);
  mapbox::geometry::polygon<double> polygon;
  for (const auto& path : *diff) {
    mapbox::geometry::linear_ring<double> ring;

    Stroke s(path.size(), false);
    for (int j = 0; j < path.size(); ++j) {
      s.x_[j] = clip_path_inv_coord(path[j].X);
      s.y_[j] = clip_path_inv_coord(path[j].Y);
    }

    // Simplify the polygon to save region computation time
    const auto dec_s = decimated(s, precision);
    for (int j = 0; j < dec_s.size(); ++j) {
      ring.push_back({dec_s.x(j), dec_s.y(j)});
    }
    if (!ring.empty()) {
      if (polygon.empty())
        polygon.push_back(std::move(ring));
      else if (polygon.back().size() < ring.size()) {
        polygon.clear();
        polygon.push_back(std::move(ring));
      }
    }
  }
  if (polygon.empty())
    return 0;

  double radius = mapbox::polylabel_distance(polygon, center.x(), center.y(), precision);

  return radius;
}

Float face_perimeter(const StrokeGraph& stroke_graph, size_t face_idx,
                     bool include_interior_strokes) {
  /*if (face_idx == stroke_graph.boundary_face_) {
    return std::numeric_limits<Float>::infinity();
  }*/

  // For now collect strokes
  auto f = stroke_graph.face(face_idx);
  std::vector<std::vector<Stroke>> poly_strokes;
  std::vector<std::vector<size_t>> s_indices;
  auto iterations = 0;
  for (const auto hi : f.cycles()) {
    poly_strokes.emplace_back();
    s_indices.emplace_back();

    const auto he = f.graph_->hedge(hi);
    auto it = he;
    do {
      auto s_it = it;
      if (!it.forward())
        s_it = it.twin();

      int orig_sid = -1;
      if (!(s_it.flags() & StrokeGraph::HedgeRecord::Bridge))
        orig_sid = s_it.orig_stroke_idx();
      size_t stroke_idx = s_it.stroke_idx();
      SPDLOG_DEBUG("In cycle: {}, {}, v{} -> v{}", stroke_idx, orig_sid,
                   it.origin().index_, it.dest().index_);

      // Special case of zero area loop. Note this should be removed by small face
      // collapsing.
      if (s_it.origin() == s_it.dest()) {
        std::vector<Vec2> poly = f.graph_->face_positions(f.index_);
        CoordMat poly_coords;
        poly_coords.resize(poly.size(), 2);
        for (size_t i = 0; i < poly.size(); ++i) {
          poly_coords.row(i) << poly[i].x(), poly[i].y();
        }

        auto face_polygon = to_clip_paths(span<const CoordMat>{&poly_coords, 1});
        Float check_area1 = clip_area_scaled(*face_polygon);

        Float check_area2 = 1;
        // This can't be the outside face
        if (it.twin().face_idx() != stroke_graph.boundary_face_) {
          poly = f.graph_->face_positions(it.twin().face_idx());
          poly_coords.resize(poly.size(), 2);
          for (size_t i = 0; i < poly.size(); ++i) {
            poly_coords.row(i) << poly[i].x(), poly[i].y();
          }
          face_polygon = to_clip_paths(span<const CoordMat>{&poly_coords, 1});
          check_area2 = clip_area_scaled(*face_polygon);
        }

        if (check_area1 < std::numeric_limits<Float>::epsilon() ||
            check_area2 < std::numeric_limits<Float>::epsilon()) {
          goto next_iteration;
        }
      }
      // End of the special case

      {
        // Write stroke index for debugging visualization on python side.
        poly_strokes.back().emplace_back(stroke_graph.strokes_[stroke_idx].clone());
        if (!it.forward())
          poly_strokes.back().back().reverse();
        s_indices.back().emplace_back(stroke_idx);
      }

next_iteration:
      it = it.next();
      iterations++;
      assert(iterations < 1024 && "likely infinite loop found");
    } while (it != he);

    if (poly_strokes.back().empty()) {
      poly_strokes.pop_back();
      s_indices.pop_back();
    }
  }

  // Ignore the strokes within the region
  if (!include_interior_strokes) {
    for (size_t i = 0; i < poly_strokes.size(); ++i) {
      // Find the turning point
      bool to_update = false;
      do {
        to_update = false;
        size_t j = 0;
        size_t k = (poly_strokes[i].size() - 1) % poly_strokes[i].size();
        int start_sid = -1;
        for (j = 0; j < poly_strokes[i].size(); ++j) {
          k = (j + poly_strokes[i].size() - 1) % poly_strokes[i].size();
          if (s_indices[i][j] == s_indices[i][k]) {
            start_sid = s_indices[i][j];
            to_update = true;
            break;
          }
        }
        size_t prev_j, prev_k;
        while (to_update && s_indices[i][j] == s_indices[i][k] && j != k) {
          prev_j = j;
          prev_k = k;
          j = (j + 1) % poly_strokes[i].size();
          k = (k + poly_strokes[i].size() - 1) % poly_strokes[i].size();
          // j and k can swap when the vector count is even
          if (prev_j == k && prev_k == j) {
            break;
          }
        }

        if (to_update && s_indices[i][j] == s_indices[i][k]) {
          if (j == k) {
            size_t tmp_sid = s_indices[i][j];
            Stroke tmp_s = std::move(poly_strokes[i][j]);

            poly_strokes[i].clear();
            s_indices[i].clear();

            if (tmp_sid != start_sid) {
              poly_strokes[i].emplace_back(std::move(tmp_s));
              s_indices[i].emplace_back(tmp_sid);
            }
          } else {
            poly_strokes[i].clear();
            s_indices[i].clear();
          }

          to_update = false;
        } else if (to_update) {
          std::vector<Stroke> tmp_poly_s;
          std::vector<size_t> tmp_s_idx;
          while (j != k) {
            tmp_poly_s.emplace_back(std::move(poly_strokes[i][j]));
            tmp_s_idx.emplace_back(s_indices[i][j]);
            j = (j + 1) % poly_strokes[i].size();
          }
          tmp_poly_s.emplace_back(std::move(poly_strokes[i][k]));
          tmp_s_idx.emplace_back(s_indices[i][k]);

          poly_strokes[i] = std::move(tmp_poly_s);
          s_indices[i] = std::move(tmp_s_idx);
        }
      } while (to_update);
    }

    // Remove empty ones
    {
      std::vector<std::vector<Stroke>> tmp_poly_strokes;
      std::vector<std::vector<size_t>> tmp_s_indices;
      for (size_t i = 0; i < poly_strokes.size(); ++i) {
        if (poly_strokes[i].empty())
          continue;
        tmp_poly_strokes.emplace_back(std::move(poly_strokes[i]));
        tmp_s_indices.emplace_back(s_indices[i]);
      }
      poly_strokes = std::move(tmp_poly_strokes);
      s_indices = std::move(tmp_s_indices);
    }
  }

  if (poly_strokes.empty())
    return 0;

  // Tune to have precision = 1 based on Dance_lien-ze, Tsao_014
  double max_pen_width = -1;
  for (size_t i = 0; i < poly_strokes.size(); ++i) {
    for (const auto& s : poly_strokes[i]) {
      max_pen_width = std::max(max_pen_width, s.pen_width());
    }
  }
  double precision = max_pen_width / 8;

  // Convert to polygons
  std::vector<CoordMat> poly_coords;
  std::vector<std::pair<Float, Float>> dec_points;
  for (size_t i = 0; i < poly_strokes.size(); ++i) {
    poly_coords.push_back(CoordMat());
    dec_points.clear();
    for (const auto& s : poly_strokes[i]) {
      // Simplify the polygon to save region computation time
      const auto dec_s = decimated(s, precision);
      for (int j = 0; j < dec_s.size(); ++j) {
        dec_points.emplace_back(dec_s.x(j), dec_s.y(j));
      }
    }
    poly_coords.back().resize(dec_points.size(), 2);
    for (int j = 0; j < dec_points.size(); ++j) {
      poly_coords.back().row(j) << dec_points[j].first, dec_points[j].second;
    }
    if (poly_coords.back().size() == 0)
      poly_coords.pop_back();
  }

  if (poly_strokes.empty())
    return 0;

  // Remove the stroke envelopes
  auto face_polygon = to_clip_paths(poly_coords);

  std::vector<CoordMat> coords;
  for (size_t i = 0; i < poly_strokes.size(); ++i) {
    for (const auto& s : poly_strokes[i]) {
      if (s.size() > 1) {
        outline_to_polygons(s, coords, precision * 2);
      }
    }
  }
  const auto clipped = to_clip_paths(coords);
  auto diff = boolean_difference(*face_polygon, *clipped);

  Float perimeter = 0;
  for (const auto& path : *diff) {
    Stroke s(path.size(), false);
    for (int j = 0; j < path.size(); ++j) {
      s.x_[j] = clip_path_inv_coord(path[j].X);
      s.y_[j] = clip_path_inv_coord(path[j].Y);
    }

    // Simplify the polygon to save region computation time
    const auto dec_s = decimated(s, precision);
    for (int j = 1; j < dec_s.size(); ++j) {
      perimeter +=
        (Vec2(dec_s.x(j), dec_s.y(j)) - Vec2(dec_s.x(j - 1), dec_s.y(j - 1))).norm();
    }
  }

  return perimeter;
}

Float face_area(const StrokeGraph& stroke_graph, size_t face_idx) {
  if (face_idx == stroke_graph.boundary_face_) {
    return std::numeric_limits<Float>::infinity();
  }

  // For now collect strokes
  auto f = stroke_graph.face(face_idx);
  std::vector<std::vector<Stroke>> poly_strokes;
  auto iterations = 0;
  for (const auto hi : f.cycles()) {
    poly_strokes.emplace_back();

    std::vector<size_t> s_indices;

    const auto he = f.graph_->hedge(hi);
    auto it = he;
    do {
      auto s_it = it;
      if (!it.forward())
        s_it = it.twin();

      int orig_sid = -1;
      if (!(s_it.flags() & StrokeGraph::HedgeRecord::Bridge))
        orig_sid = s_it.orig_stroke_idx();
      size_t stroke_idx = s_it.stroke_idx();
      SPDLOG_DEBUG("In cycle: {}, {}, v{} -> v{}", stroke_idx, orig_sid,
                   it.origin().index_, it.dest().index_);

      // Special case of zero area loop. Note this should be removed by small face
      // collapsing.
      if (s_it.origin() == s_it.dest()) {
        std::vector<Vec2> poly = f.graph_->face_positions(f.index_);
        CoordMat poly_coords;
        poly_coords.resize(poly.size(), 2);
        for (size_t i = 0; i < poly.size(); ++i) {
          poly_coords.row(i) << poly[i].x(), poly[i].y();
        }

        auto face_polygon = to_clip_paths(span<const CoordMat>{&poly_coords, 1});
        Float check_area1 = clip_area_scaled(*face_polygon);

        Float check_area2 = 1;
        // This can't be the outside face
        if (it.twin().face_idx() != stroke_graph.boundary_face_) {
          poly = f.graph_->face_positions(it.twin().face_idx());
          poly_coords.resize(poly.size(), 2);
          for (size_t i = 0; i < poly.size(); ++i) {
            poly_coords.row(i) << poly[i].x(), poly[i].y();
          }
          face_polygon = to_clip_paths(span<const CoordMat>{&poly_coords, 1});
          check_area2 = clip_area_scaled(*face_polygon);
        }

        if (check_area1 < std::numeric_limits<Float>::epsilon() ||
            check_area2 < std::numeric_limits<Float>::epsilon()) {
          goto next_iteration;
        }
      }
      // End of the special case

      {
        // Write stroke index for debugging visualization on python side.
        poly_strokes.back().emplace_back(stroke_graph.strokes_[stroke_idx].clone());
        if (!it.forward())
          poly_strokes.back().back().reverse();
        s_indices.emplace_back(stroke_idx);
      }

next_iteration:
      it = it.next();
      iterations++;
      assert(iterations < 1024 && "likely infinite loop found");
    } while (it != he);

    if (poly_strokes.back().empty()) {
      poly_strokes.pop_back();
    }
  }

  if (poly_strokes.empty())
    return 0;

  // Tune to have precision = 1 based on Dance_lien-ze, Tsao_014
  double max_pen_width = -1;
  for (size_t i = 0; i < poly_strokes.size(); ++i) {
    for (const auto& s : poly_strokes[i]) {
      max_pen_width = std::max(max_pen_width, s.pen_width());
    }
  }
  double precision = max_pen_width / 8;

  // Convert to polygons
  std::vector<CoordMat> poly_coords;
  std::vector<std::pair<Float, Float>> dec_points;
  for (size_t i = 0; i < poly_strokes.size(); ++i) {
    poly_coords.push_back(CoordMat());
    dec_points.clear();
    for (const auto& s : poly_strokes[i]) {
      // Simplify the polygon to save region computation time
      const auto dec_s = decimated(s, precision);
      for (int j = 0; j < dec_s.size(); ++j) {
        dec_points.emplace_back(dec_s.x(j), dec_s.y(j));
      }
    }
    poly_coords.back().resize(dec_points.size(), 2);
    for (int j = 0; j < dec_points.size(); ++j) {
      poly_coords.back().row(j) << dec_points[j].first, dec_points[j].second;
    }
    if (poly_coords.back().size() == 0)
      poly_coords.pop_back();
  }

  if (poly_strokes.empty())
    return 0;

  // Remove the stroke envelopes
  auto face_polygon = to_clip_paths(poly_coords);

  std::vector<CoordMat> coords;
  for (size_t i = 0; i < poly_strokes.size(); ++i) {
    for (const auto& s : poly_strokes[i]) {
      if (s.size() > 1) {
        outline_to_polygons(s, coords, precision * 2);
      }
    }
  }
  const auto clipped = to_clip_paths(coords);
  auto diff = boolean_difference(*face_polygon, *clipped);
  Float area = clip_area_to_real_area(clip_area_scaled(*diff));
  return area;
}

Float face_stroke_width_min(const StrokeGraph& stroke_graph, size_t face_idx) {
  if (face_idx == stroke_graph.boundary_face_) {
    return 0;
  }

  // For now collect strokes
  auto f = stroke_graph.face(face_idx);
  std::vector<std::vector<Stroke>> poly_strokes;
  auto iterations = 0;
  for (const auto hi : f.cycles()) {
    poly_strokes.emplace_back();

    std::vector<size_t> s_indices;

    const auto he = f.graph_->hedge(hi);
    auto it = he;
    do {
      auto s_it = it;
      if (!it.forward())
        s_it = it.twin();

      int orig_sid = -1;
      if (!(s_it.flags() & StrokeGraph::HedgeRecord::Bridge))
        orig_sid = s_it.orig_stroke_idx();
      size_t stroke_idx = s_it.stroke_idx();
      SPDLOG_DEBUG("In cycle: {}, {}, v{} -> v{}", stroke_idx, orig_sid,
                   it.origin().index_, it.dest().index_);

      // Special case of zero area loop. Note this should be removed by small face
      // collapsing.
      if (s_it.origin() == s_it.dest()) {
        std::vector<Vec2> poly = f.graph_->face_positions(f.index_);
        CoordMat poly_coords;
        poly_coords.resize(poly.size(), 2);
        for (size_t i = 0; i < poly.size(); ++i) {
          poly_coords.row(i) << poly[i].x(), poly[i].y();
        }

        auto face_polygon = to_clip_paths(span<const CoordMat>{&poly_coords, 1});
        Float check_area1 = clip_area_scaled(*face_polygon);

        Float check_area2 = 1;
        // This can't be the outside face
        if (it.twin().face_idx() != stroke_graph.boundary_face_) {
          poly = f.graph_->face_positions(it.twin().face_idx());
          poly_coords.resize(poly.size(), 2);
          for (size_t i = 0; i < poly.size(); ++i) {
            poly_coords.row(i) << poly[i].x(), poly[i].y();
          }
          face_polygon = to_clip_paths(span<const CoordMat>{&poly_coords, 1});
          check_area2 = clip_area_scaled(*face_polygon);
        }

        if (check_area1 < std::numeric_limits<Float>::epsilon() ||
            check_area2 < std::numeric_limits<Float>::epsilon()) {
          goto next_iteration;
        }
      }
      // End of the special case

      // Ignore the strokes within the region
      // if (!s_indices.empty() && s_indices.back() == stroke_idx) {
      if (false) {
        poly_strokes.back().pop_back();
        s_indices.pop_back();
      } else {
        // Write stroke index for debugging visualization on python side.
        poly_strokes.back().emplace_back(stroke_graph.strokes_[stroke_idx].clone());
        if (!it.forward())
          poly_strokes.back().back().reverse();
        s_indices.emplace_back(stroke_idx);
      }

next_iteration:
      it = it.next();
      iterations++;
      assert(iterations < 1024 && "likely infinite loop found");
    } while (it != he);

    if (poly_strokes.back().empty()) {
      poly_strokes.pop_back();
    }
  }

  if (poly_strokes.empty())
    return 0;

  Float min_width = std::numeric_limits<Float>::infinity();
  for (const auto& strokes : poly_strokes) {
    for (const auto& s : strokes)
      if (s.pen_width() > 0)
        min_width = std::min(s.pen_width(), min_width);
  }
  assert(min_width < std::numeric_limits<Float>::infinity());

  return min_width;
}

void junction_distance_sort(const StrokeGraph& stroke_graph,
                            std::vector<Junction>& candidates) {
  std::sort(candidates.begin(), candidates.end(),
            [&stroke_graph](const Junction& junc1, const Junction& junc2) -> bool {
              Vec2 pos11 = stroke_graph.orig_strokes_[junc1.points[0].first].pos_norm(
                junc1.points[0].second);
              Vec2 pos12 = stroke_graph.orig_strokes_[junc1.points[1].first].pos_norm(
                junc1.points[1].second);
              Float dist1 = (pos11 - pos12).norm();

              Vec2 pos21 = stroke_graph.orig_strokes_[junc2.points[0].first].pos_norm(
                junc2.points[0].second);
              Vec2 pos22 = stroke_graph.orig_strokes_[junc2.points[1].first].pos_norm(
                junc2.points[1].second);
              Float dist2 = (pos21 - pos22).norm();
              return dist1 < dist2;
            });
}

void junction_drawing_order_sort(std::vector<Junction>& candidates) {
  std::sort(candidates.begin(), candidates.end(),
            [](const Junction& junc1, const Junction& junc2) -> bool {
              int order1 = std::max(junc1.points[0].first, junc1.points[1].first);
              int order2 = std::max(junc2.points[0].first, junc2.points[1].first);

              if (order1 == order2) {
                int order21 = std::min(junc1.points[0].first, junc1.points[1].first);
                int order22 = std::min(junc2.points[0].first, junc2.points[1].first);

                return order21 < order22;
              }

              return order1 < order2;
            });
}

void junction_probability_sort(std::vector<Junction>& candidates) {
  std::sort(candidates.begin(), candidates.end(),
            [](const Junction& junc1, const Junction& junc2) -> bool {
              if (junc1.probability != junc2.probability)
                return junc1.probability > junc2.probability;

              int order1 = std::max(junc1.points[0].first, junc1.points[1].first);
              int order2 = std::max(junc2.points[0].first, junc2.points[1].first);

              if (order1 == order2) {
                int order21 = std::min(junc1.points[0].first, junc1.points[1].first);
                int order22 = std::min(junc2.points[0].first, junc2.points[1].first);

                return order21 < order22;
              }

              return order1 < order2;
            });
}

bool debug_connectivity_state(const StrokeGraph& stroke_graph,
                              span<Junction> connected_junctions,
                              std::vector<Junction>& invalid_junctions) {
  std::vector<bool> junction_connected;
  junction_connected.resize(connected_junctions.size(), true);

  std::vector<std::pair<size_t, size_t>> adj_faces;
  std::vector<Float> junc_distances;
  std::vector<StrokeGraph::VertexID> junc_vertex_ids;
  auto varying_stroke_graph =
    modify_graph(stroke_graph, connected_junctions, junction_connected, adj_faces,
                 junc_distances, junc_vertex_ids);

  if (!varying_stroke_graph) {
    return false;
  }

  for (size_t i = 0; i < adj_faces.size(); ++i) {
    Float f1_size =
      face_maximum_inscribing_circle_radius(*varying_stroke_graph, adj_faces[i].first);
    Float f2_size =
      face_maximum_inscribing_circle_radius(*varying_stroke_graph, adj_faces[i].second);
    bool cur_pass_check = true;
    cur_pass_check &= euclidean_region_condition_check(f1_size, junc_distances[i]);
    cur_pass_check &= euclidean_region_condition_check(f2_size, junc_distances[i]);

    if (i + 1 == adj_faces.size())
      SPDLOG_INFO("{} vs {}: {}; {}: {}", junc_distances[i], adj_faces[i].first, f1_size,
                  adj_faces[i].second, f2_size);

    if (!cur_pass_check) {
      invalid_junctions.emplace_back(connected_junctions[i]);
    }
  }

  return true;
}
} // namespace sketching
