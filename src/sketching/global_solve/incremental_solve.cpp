#include "incremental_solve.h"

#include "../bounding_box.h"
#include "../closest.h"
#include "../force_assert.h"
#include "../graph_color.h"
#include "../incremental.h"
#include "../intersect.h"
#include "../io/pdf.h"
#include "../resample.h"
#include "../stroke_graph.h"
#include "../stroke_graph_extra.h"
#include "incremental_decomposition.h"
#include "incremental_obj.h"
#include "incremental_param.h"
#include "incremental_region.h"
#include "incremental_region_util.h"

#include <fstream>
#include <functional>
#include <queue>

namespace sketching {
Float region_term_ratio = 0.0;
FeatureType prediction_feature_type = FeatureType::OrigStroke;
bool to_bridge = false;

namespace {
void build_final_predictions(span<const Stroke> strokes, StrokeGraph& graph,
                             std::vector<Junction>& final_predictions) {
  IncrementalCache cache;
  for (size_t i = 0; i < strokes.size(); ++i) {
    add_stroke_incremental_topological(graph, &cache, strokes[i], false);
  }
  graph.orig_bvh_ = std::make_unique<PolylineBVH>(graph.orig_strokes_);

  // Propose candidate from dangling vertices
  std::vector<SnapInfo> stroke_pos_candidates;
  for (size_t vi = 0; vi < graph.vertices_.size(); ++vi) {
    if (!graph.vertex(vi).is_active())
      continue;
    snap_candidates(graph, vi, &cache, prediction_feature_type, stroke_pos_candidates,
                    snap_candidate_count, true, false);
  }

  final_predictions.reserve(stroke_pos_candidates.size());
  for (const auto& snap : stroke_pos_candidates) {
    Junction junc({StrokeTime((int)snap.stroke_pos2.first, snap.stroke_pos2.second),
                   StrokeTime((int)snap.stroke_pos1.first, snap.stroke_pos1.second)},
                  snap.prediction_type, false, 0.5, true);
    const auto ok = stroke_to_original_stroke_indexing(graph, junc);
    force_assert(ok && "couldn't map from strokes to orig");
    if (std::find(final_predictions.begin(), final_predictions.end(), junc) ==
          final_predictions.end() &&
        is_candidate_valid(graph, strokes, graph, graph.orig_strokes_.size() - 1, junc,
                           std::vector<Junction>())) {
      junc.orig_dist = junction_distance_init(graph, junc);
      final_predictions.emplace_back(junc);
    }
  }
  graph.orig_bvh_ = std::make_unique<PolylineBVH>(graph.orig_strokes_);

  // Make predictions
  update_disconnected_junction_predictions(graph, final_predictions,
                                           prediction_feature_type);
}

void expand_removal_set(GraphState& next_state, std::set<size_t>& removed) {
  // Bind/expand removed junctions
  std::set<size_t> new_removed;
  for (auto itr = removed.begin(); itr != removed.end(); ++itr) {
    if (next_state.candidates_[*itr].is_weak)
      continue;
    std::vector<Junction> bind_junc;
    expand_junction(next_state.graph_, next_state.candidates_.at(*itr), bind_junc);

    for (const auto& junc : bind_junc) {
      // Find
      auto junc_itr = std::find_if(
        next_state.candidates_.begin(), next_state.candidates_.end(),
        [&junc](const Junction& junc_) {
          return (junc_.type == junc.type) &&
                 (junc_.points[0].first == junc.points[0].first) &&
                 (std::abs(junc_.points[0].second - junc.points[0].second) < 1e-10) &&
                 (junc_.points[1].first == junc.points[1].first) &&
                 (std::abs(junc_.points[1].second - junc.points[1].second) < 1e-10);
        });
      // Remove if disconnected
      if (junc_itr != next_state.candidates_.end()) {
        auto i = junc_itr - next_state.candidates_.begin();
        if (!next_state.connectivity_[i])
          new_removed.insert(i);
      }
    }
  }
  removed.insert(new_removed.begin(), new_removed.end());
}

void update_junctions(GraphState& next_state) {
  // 1. Record the existing T junctions so we don't have duplicates when changing types.
  // TODO: We are not accounting for the situation where an end point can form multiple
  // different T junctions with a stroke

  // Note we don't remove or update connected junctions even when they are out of date
  std::set<size_t> removed;
  std::set<std::pair<size_t, size_t>> seen_t_juncs;
  for (size_t i = 0; i < next_state.candidates_.size(); ++i) {
    // Remove the candidates used only in region solve
    if (next_state.candidates_[i].is_weak && !next_state.connectivity_[i]) {
      removed.emplace(i);
    }
    if (next_state.candidates_[i].type != JunctionType::Type::T ||
        !next_state.connectivity_[i])
      continue;
    auto p = std::make_pair(std::min(next_state.candidates_[i].points[0].first,
                                     next_state.candidates_[i].points[1].first),
                            std::max(next_state.candidates_[i].points[0].first,
                                     next_state.candidates_[i].points[1].first));
    seen_t_juncs.emplace(p);
  }
  for (size_t i = 0; i < next_state.candidates_.size(); ++i) {
    if (next_state.candidates_[i].type != JunctionType::Type::T ||
        next_state.connectivity_[i])
      continue;
    auto p = std::make_pair(std::min(next_state.candidates_[i].points[0].first,
                                     next_state.candidates_[i].points[1].first),
                            std::max(next_state.candidates_[i].points[0].first,
                                     next_state.candidates_[i].points[1].first));
    if (seen_t_juncs.count(p)) {
      removed.emplace(i);
      continue;
    }
    seen_t_juncs.emplace(p);
  }

  auto is_valid_tjunc = [](const Junction& junc) -> bool {
    return std::abs(junc.points[1].second) < std::numeric_limits<Float>::epsilon() ||
           std::abs(1 - junc.points[1].second) < std::numeric_limits<Float>::epsilon();
  };

  // 2. Change types. Remove if we see any duplicate
  for (size_t i = 0; i < next_state.candidates_.size(); ++i) {
    Junction junc = next_state.candidates_[i];
    // Remove candidates that can no longer be mapped due to stroke deletion
    if (next_state.candidates_[i].points[0].second < 0 ||
        next_state.candidates_[i].points[1].second < 0 ||
        !original_stroke_to_stroke_indexing(next_state.graph_, junc)) {
      removed.emplace(i);
      continue;
    }
    if (next_state.connectivity_[i])
      continue;
    if (next_state.candidates_[i].type != JunctionType::Type::R) {
      Junction junc2 = junc;
      if (!is_valid_tjunc(junc2)) {
        continue;
      }
      if (!stroke_to_original_stroke_indexing(next_state.graph_, junc2)) {
        removed.emplace(i);
        continue;
      }
      if (junc2.points[0] == junc2.points[1]) {
        continue;
      }
      next_state.candidates_[i] = junc2;
    }
    if (!is_valid_tjunc(junc)) {
      // Convert to T junc
      junc.type = JunctionType::Type::T;
      if (!stroke_to_original_stroke_indexing(next_state.graph_, junc)) {
        removed.emplace(i);
        continue;
      }
      // Check if this T junc exists
      auto p2 = std::make_pair(std::min(junc.points[0].first, junc.points[1].first),
                               std::max(junc.points[0].first, junc.points[1].first));
      if (seen_t_juncs.count(p2)) {
        removed.emplace(i);
        continue;
      }
      seen_t_juncs.emplace(p2);
      next_state.candidates_[i] = junc;
    }
  }

  // 3. Update T junction geometries
  for (size_t i = 0; i < next_state.candidates_.size(); ++i) {
    if (next_state.candidates_[i].type != JunctionType::Type::T ||
        next_state.connectivity_[i] || removed.count(i)) {
      continue;
    }

    if (next_state.candidates_[i].points[1].second != 0.0 &&
        next_state.candidates_[i].points[1].second != 1.0) {
      removed.emplace(i);
      continue;
    }

    // This would only happen if both ends get dissolved
    if (!reproject_t_junc(next_state.graph_, next_state.candidates_[i])) {
      removed.emplace(i);
      continue;
    }
  }

  // Delete the junctions whose two ends are at the same vertex
  for (size_t i = 0; i < next_state.candidates_.size(); ++i) {
    if (removed.count(i) || next_state.connectivity_[i])
      continue;
    auto cand = Junction{next_state.candidates_[i]}; // Copy.

    auto ok = original_stroke_to_stroke_indexing(next_state.graph_, cand);
    force_assert(ok && "couldn't map from orig to strokes");

    if (!((std::abs(cand.points[0].second) < 1e-10 ||
           std::abs(1.0 - cand.points[0].second) < 1e-10) &&
          (std::abs(cand.points[1].second) < 1e-10 ||
           std::abs(1.0 - cand.points[1].second) < 1e-10)))
      continue;

    auto v1 =
      endpoint_to_vertex(next_state.graph_, Endpoint{(size_t)cand.points[0].first,
                                                     cand.points[0].second < 1e-10});
    auto v2 =
      endpoint_to_vertex(next_state.graph_, Endpoint{(size_t)cand.points[1].first,
                                                     cand.points[1].second < 1e-10});
    if (v1 == v2)
      removed.insert(i);
  }

  // Delete the junctions that can cause graph violation
  for (size_t i = 0; i < next_state.candidates_.size(); ++i) {
    if (removed.count(i) || next_state.connectivity_[i])
      continue;
    auto candidates = next_state.candidates_;
    std::unordered_map<std::string, Float> junc_distance_map =
      next_state.junc_distance_map_;
    std::unordered_map<std::string, Float> region_size_cache;
    std::set<size_t> binding;
    binding.emplace(i);

    // @optimize Create a version of try_connect_graph which doesn't clone the graph.
    auto modified_graph = try_connect_graph(next_state.graph_, region_size_cache,
                                            junc_distance_map, candidates, binding);
    if (!modified_graph)
      removed.insert(i); // Connection not possible.
  }

  // Bind/expand removed junctions
  expand_removal_set(next_state, removed);

  // Remove junctions
  for (auto itr = removed.rbegin(); itr != removed.rend(); ++itr) {
    size_t i = *itr;
    // Only checks if it's within range. It's possible that we remove a connected junction
    // if one of its corresponding original strokes is removed entirely.
    assert(i < next_state.connectivity_.size() &&
           next_state.connectivity_.size() == next_state.candidates_.size());
    next_state.candidates_.erase(next_state.candidates_.begin() + i);
    next_state.connectivity_.erase(next_state.connectivity_.begin() + i);
  }
}

void post_add_update(GraphState& next_state) {
  // Dissolve valence-2 vertices
  // TODO: This is not necessary for the non-incremental version. Do we want to have this
  // for the incremental?
  /*for (size_t vi = 0; vi < next_state.graph_.vertices_.size(); ++vi) {
    const auto v = next_state.graph_.vertex(vi);
    if (should_dissolve_vertex(v)) {
      dissolve_vertex(next_state.graph_, v.index_);
    }
  }*/

  // Update junctions
  // Junctions may change types as a result of 2-valence vertex dissolving
  // Junctions may change geometries as a result of connection
  update_junctions(next_state);

  // Update all junction predictions.
  {
    SPDLOG_DEBUG("Before update_disconnected_junction_predictions");

    std::vector<Junction> varying_candidates;
    varying_candidates.reserve(next_state.candidates_.size());
    for (size_t i = 0; i < next_state.candidates_.size(); ++i) {
      // For now, only update the disconnected junctions
      if (next_state.connectivity_[i])
        continue;
      varying_candidates.emplace_back(next_state.candidates_[i]);
    }
    update_disconnected_junction_predictions(next_state.graph_, varying_candidates,
                                             prediction_feature_type);
    size_t v_i = 0;
    std::set<size_t> removed;
    for (size_t i = 0; i < next_state.candidates_.size(); ++i) {
      // For now, only update the disconnected junctions
      if (next_state.connectivity_[i])
        continue;
      next_state.candidates_[i].probability = varying_candidates[v_i++].probability;
      if (next_state.candidates_[i].probability < std::numeric_limits<Float>::epsilon())
        removed.emplace(i);
    }

    // Bind/expand removed junctions
    expand_removal_set(next_state, removed);

    // Remove the junctions with prob = 0
    for (auto itr = removed.rbegin(); itr != removed.rend(); ++itr) {
      assert(!*(next_state.connectivity_.begin() + *itr));
      next_state.candidates_.erase(next_state.candidates_.begin() + *itr);
      next_state.connectivity_.erase(next_state.connectivity_.begin() + *itr);
    }

    SPDLOG_DEBUG("After update_disconnected_junction_predictions");
  }

  // Remove no longer existing variables from the distance record
  std::vector<std::string> removed_cand;
  for (const auto& [repr, dist] : next_state.junc_distance_map_) {
    bool found = false;
    for (const auto& j : next_state.candidates_) {
      if (j.repr == repr) {
        found = true;
        break;
      }
    }

    if (!found)
      removed_cand.emplace_back(repr);
  }
  for (const auto& repr : removed_cand)
    next_state.junc_distance_map_.erase(repr);

  // Recompute stroke length for safety
  for (auto& s : next_state.graph_.strokes_) {
    s.compute_arclengths();
  }
  for (auto& s : next_state.graph_.orig_strokes_) {
    s.compute_arclengths();
  }
}

} // namespace

int incremental_solve(span<const Stroke> strokes, StrokeGraph::SnappingType snapping_type,
                      std::vector<StrokeGraph>& stroke_graphs,
                      std::vector<StrokeSnapInfo>& predictions,
                      std::unordered_map<size_t, int>& prev_state, size_t max_num_strokes,
                      size_t max_per_stroke_states, const std::string& viz_dir,
                      const SolveType solve_type) {
  std::vector<std::vector<Junction>> candidates;
  std::vector<std::vector<bool>> connectivities;
  if (strokes.empty())
    return -1;

  max_num_states_region = max_per_stroke_states;

  // State management
  auto compare_states = [](const GraphState& s1, const GraphState& s2) {
    return s1.curr_stroke_step_ > s2.curr_stroke_step_ ||
           (s1.curr_stroke_step_ == s2.curr_stroke_step_ && s1.obj_ < s2.obj_);
  };
  typedef std::priority_queue<GraphState, std::vector<GraphState>,
                              decltype(compare_states)>
    StateQueue;
  StateQueue connection_states(compare_states);
  GraphState init_state{0,
                        StrokeGraph(snapping_type),
                        IncrementalCache(),
                        std::vector<Junction>(),
                        std::vector<bool>(),
                        std::unordered_map<std::string, Float>(),
                        std::unordered_map<std::string, Float>()};
  init_state.prev_state_idx_ = -1;
  init_state.state_idx_ = 0;
  if (solve_type == CCSolve)
    init_state.curr_stroke_step_ = strokes.size() - 1;
  else {
    // Build the candidate constraints based on the final graph
    set_future_constraints(&init_state.cache_, strokes, snap_candidate_count, false);
  }

  connection_states.push(init_state);

  std::unordered_map<size_t, StrokeGraph> intermediate_graphs;
  intermediate_graphs[init_state.state_idx_] = init_state.graph_.clone();

  std::vector<GraphState> final_states;

  std::map<std::pair<std::pair<size_t, Float>, std::pair<size_t, Float>>, size_t>
    global_junction_indexing;
  struct JunctionAssignment {
    size_t junc_idx_;
    bool connected_;
  };
  auto assignment_hash =
    [](const std::pair<size_t, std::vector<JunctionAssignment>>& record) -> size_t {
    std::string h = std::to_string(record.first) + ";";
    for (const auto& j : record.second) {
      h += std::to_string(j.junc_idx_) + ":" + std::to_string((int)j.connected_) + ";";
    }
    return std::hash<std::string>{}(h);
  };
  auto assignment_eql =
    [&assignment_hash](
      const std::pair<size_t, std::vector<JunctionAssignment>>& record1,
      const std::pair<size_t, std::vector<JunctionAssignment>>& record2) -> bool {
    return assignment_hash(record1) == assignment_hash(record2);
  };
  std::unordered_set<std::pair<size_t, std::vector<JunctionAssignment>>,
                     decltype(assignment_hash), decltype(assignment_eql)>
    junction_assignment_record(1024, assignment_hash, assignment_eql);
  auto get_global_junction_index =
    [&global_junction_indexing](const std::pair<size_t, Float>& end1,
                                const std::pair<size_t, Float>& end2) {
      auto p = std::make_pair(end1, end2);
      if (!global_junction_indexing.count(p))
        global_junction_indexing[p] = global_junction_indexing.size();
      return global_junction_indexing[p];
    };
  auto seen_junction_assignment_record =
    [&junction_assignment_record, &get_global_junction_index](
      size_t curr_stroke_step, const std::vector<Junction>& candidates,
      const std::vector<bool>& connectivity) -> bool {
    assert(candidates.size() == connectivity.size());
    std::vector<JunctionAssignment> record;
    record.reserve(candidates.size());
    for (size_t i = 0; i < candidates.size(); ++i) {
      JunctionAssignment assign{
        get_global_junction_index(candidates[i].points[0], candidates[i].points[1]),
        connectivity[i]};
      record.emplace_back(assign);
    }
    auto p = std::make_pair(curr_stroke_step, record);
    if (junction_assignment_record.count(p))
      return true;
    junction_assignment_record.insert(p);
    return false;
  };

  // Compute final predictions for the junction trimming
  std::vector<Junction> final_predictions;
  StrokeGraph final_graph(snapping_type);
  if (solve_type != CCSolve)
    build_final_predictions(strokes, final_graph, final_predictions);
  if (!viz_dir.empty() && (solve_type == RegionSolve)) {
    std::vector<std::pair<ClassifierPrediction, VizProbabilities>> preds;
    junc2pred(final_graph, final_predictions, preds);
    PlotParams params;
    params.media_box = visual_bounds(strokes);
    params.compress = false; // to be fast
    viz_pdf(strokes, final_graph, viz_dir + "/final_predictions.pdf", preds,
            std::vector<std::string>());
  }

  // Explore by getting a state and trying to add a new stroke to it
  size_t state_count = 1;
  int seen_stroke_step = -1;
  size_t per_stroke_count = 0;
  std::vector<StateQueue> connection_state_levels;
  size_t total_state_count = 1;
  do {
    connection_state_levels.clear();
    connection_state_levels = std::vector<StateQueue>{
      StateQueue(compare_states), StateQueue(compare_states), StateQueue(compare_states)};
    while (!connection_states.empty()) {
      GraphState state = connection_states.top();
      connection_states.pop();
      prev_state[state.state_idx_] = state.prev_state_idx_;

      // Optional: limit the number of states per stroke
      per_stroke_count++;
      if (state.curr_stroke_step_ != seen_stroke_step) {
        per_stroke_count = 0;
        seen_stroke_step = (int)state.curr_stroke_step_;
      }
      if (state.curr_stroke_step_ + unlimited_last_steps < strokes.size() &&
          per_stroke_count >= max_per_stroke_states)
        continue;

      SPDLOG_INFO("State {} (#{}) at {} / {}; Obj: {}; #{}", state.state_idx_,
                  total_state_count++, state.curr_stroke_step_, strokes.size() - 1,
                  state.obj_, per_stroke_count);
      StrokeGraph before_graph = state.graph_.clone();

      // Visualize the state (disconnected junctions only) before adding the new stroke
      if (!viz_dir.empty() && (solve_type == RegionSolve)) {
        std::vector<Junction> junctions;
        junctions.reserve(state.candidates_.size());
        for (size_t i = 0; i < state.candidates_.size(); ++i) {
          if (state.connectivity_[i])
            continue;
          junctions.emplace_back(state.candidates_[i]);
        }
        std::vector<std::pair<ClassifierPrediction, VizProbabilities>> preds;
        junc2pred(state.graph_, junctions, preds, final_predictions);
        std::vector<std::string> connected_junc_strs;
        viz_connections(state, final_predictions, connected_junc_strs);
        viz_pdf(strokes, state.graph_,
                viz_dir + "/" + std::to_string(state.state_idx_) + "_begin" + ".pdf",
                preds, connected_junc_strs, state.face_colors_);
      }
      //

      // Add a stroke to the graph
      SPDLOG_DEBUG("Before add_stroke_incremental_topological");
      state.cache_.candidates_ = std::move(state.candidates_);
      StrokeSnapInfo snap_info;
      if (solve_type == CCSolve) {
        snap_info = increment_strokes(state.graph_, strokes, 0, strokes.size());
      } else {
        snap_info = add_stroke_incremental_topological(
          state.graph_, &state.cache_, strokes[state.curr_stroke_step_], false);
      }
      state.candidates_ = std::move(state.cache_.candidates_);
      SPDLOG_DEBUG("After add_stroke_incremental_topological");
      StrokeGraph added_graph = state.graph_.clone();

      // Update the state after the new stroke is added
      post_add_update(state);

      // Make decisions
      std::vector<std::vector<GraphState>> state_levels;
      {
        // Set candidates as the current vanilla
        if (solve_type == CCSolve) {
          std::vector<Junction> corner_predictions;
          vanilla_candidates(state.graph_, final_predictions, false);
          if (state.graph_.orig_strokes_.size() < strokes.size()) {
            StrokeGraph plane_graph;
            increment_strokes(plane_graph, strokes, 0, strokes.size());
            final_graph = std::move(plane_graph);
          } else
            final_graph = state.graph_.clone();
        }
        SPDLOG_DEBUG("Before region_solve");
        region_solve(strokes, final_graph, final_predictions, state, before_graph,
                     viz_dir, state.state_idx_, state.curr_stroke_step_, state_levels,
                     (solve_type == RegionSolve) ? Incremental : Final);
        SPDLOG_INFO("After region_solve");

        if (!viz_dir.empty() && (solve_type == RegionSolve)) {
          // Note that we haven't updated the face colors after adding the new stroke.
          // Update the face colors here. The unupdated values are used inside
          // region_solve.
          const auto before_face_colors = state.face_colors_;
          state.face_colors_.resize(state.graph_.faces_.size());
          color_by_reference((int)get_color_palette().size(), before_graph,
                             before_face_colors, state.graph_, state.face_colors_);
        }
      }

      for (size_t i = 0; i < state_levels.size(); ++i) {
        // Check if any of the previous queue has any state. If so, we don't need to
        // process any further
        bool prev_level_non_empty = false;
        for (size_t j = 0; j < i; ++j) {
          prev_level_non_empty = !connection_state_levels[j].empty();
          if (prev_level_non_empty)
            break;
        }
        if (prev_level_non_empty)
          break;

        std::vector<GraphState>& states = state_levels[i];
        // Expand or save the promising states
        for (GraphState& s : states) {
          // Skip the seen cases
          // Note that since state_levels[2] states are included in [0] or [1] (if they
          // are not empty), the connection_state_levels may not correspond to
          // state_levels. But since we always assign the next stroke round queue in the
          // [0] -> [2] order (in the outter iteration over connection_state_levels) and
          // stop when we see a non-empty queue stored in connection_state_levels, the
          // behavior is indeed correct.
          if (!s.candidates_.empty() &&
              seen_junction_assignment_record(state.curr_stroke_step_, s.candidates_,
                                              s.connectivity_))
            continue;

          s.prev_state_idx_ = (int)state.state_idx_;
          s.state_idx_ = state_count++;

          // Save the intermediate graphs
          intermediate_graphs[s.state_idx_] = s.graph_.clone();

          // Save the visualization info
          if (s.state_idx_ >= predictions.size())
            predictions.resize(s.state_idx_ + 1);
          predictions[s.state_idx_] = snap_info;
          for (size_t j = 0; j < s.candidates_.size(); ++j) {
            std::vector<Junction> junctions{s.candidates_[j]};
            std::vector<std::pair<ClassifierPrediction, VizProbabilities>> pred;
            junc2pred(added_graph, junctions, pred, final_predictions);
            pred.front().first.connected = s.connectivity_[j];
            predictions[s.state_idx_].predictions.emplace_back(pred.front().first);
          }

          {
            // Visualize the solution directly out of the region solve
            if (!viz_dir.empty() && (solve_type == RegionSolve)) {
              // Color
              s.face_colors_.resize(s.graph_.faces_.size());
              color_by_reference((int)get_color_palette().size(), state.graph_,
                                 state.face_colors_, s.graph_, s.face_colors_);

              std::vector<Junction> junctions;
              junctions.reserve(s.candidates_.size());
              for (size_t j = 0; j < s.candidates_.size(); ++j) {
                if (s.connectivity_[j])
                  continue;
                junctions.emplace_back(s.candidates_[j]);
              }
              std::vector<std::pair<ClassifierPrediction, VizProbabilities>> preds;
              junc2pred(s.graph_, junctions, preds, final_predictions);
              std::vector<std::string> connected_junc_strs;
              viz_connections(s, final_predictions, connected_junc_strs);

              viz_pdf(strokes, s.graph_,
                      viz_dir + "/" + std::to_string(state.state_idx_) + "_region_" +
                        std::to_string(s.state_idx_) + ".pdf",
                      preds, connected_junc_strs, s.face_colors_);
            }
            //

            // Visualize the solution after updates
            if (!viz_dir.empty() && (solve_type == RegionSolve)) {
              // Color
              s.face_colors_.resize(s.graph_.faces_.size());
              color_by_reference((int)get_color_palette().size(), state.graph_,
                                 state.face_colors_, s.graph_, s.face_colors_);

              std::vector<Junction> junctions;
              junctions.reserve(s.candidates_.size());
              for (size_t j = 0; j < s.candidates_.size(); ++j) {
                if (s.connectivity_[j])
                  continue;
                junctions.emplace_back(s.candidates_[j]);
              }
              std::vector<std::pair<ClassifierPrediction, VizProbabilities>> preds;
              junc2pred(s.graph_, junctions, preds, final_predictions);
              std::vector<std::string> connected_junc_strs;
              viz_connections(s, final_predictions, connected_junc_strs);

              viz_pdf(strokes, s.graph_,
                      viz_dir + "/" + std::to_string(state.state_idx_) +
                        "_regionupdate_" + std::to_string(s.state_idx_) + ".pdf",
                      preds, connected_junc_strs, s.face_colors_);
            }
            //

            // Compute and save obj values for prioritized processing
            // Ensure we have arclengths
            for (auto& ss : s.graph_.strokes_) {
              ss.compute_arclengths();
            }
            for (auto& ss : s.graph_.orig_strokes_) {
              ss.compute_arclengths();
            }
            /*s.obj_ = obj_function(s, (!viz_dir.empty())
                                       ? viz_dir + "/" + std::to_string(s.state_idx_) +
                                           "_region.txt"
                                       : "");*/
            Float region_obj;
            s.obj_ = obj_function(s, region_obj, "region.txt");

            if (!violation_check(s))
              s.obj_ += -1e3;
          }

          // If this is the last stroke, save the final result
          if (state.curr_stroke_step_ + 1 == strokes.size() ||
              state.curr_stroke_step_ + 1 == max_num_strokes) {
            prev_state[s.state_idx_] = s.prev_state_idx_;

            // Redo deformation if we used the faster deformation within the solve
            if (snapping_type == StrokeGraph::SnappingType::Deformation) {
              const auto& final_candidates = s.candidates_;
              std::vector<Junction> connected_junctions;
              connected_junctions.reserve(final_candidates.size());
              for (const auto& junc : final_candidates)
                if (!junc.repr.empty())
                  connected_junctions.emplace_back(junc);

              std::sort(connected_junctions.begin(), connected_junctions.end(),
                        [](const Junction& a, const Junction& b) -> bool {
                          assert(!a.repr.empty() && !b.repr.empty());
                          int a_idx, b_idx;
                          sscanf(a.repr.c_str(), "junc_%d", &a_idx);
                          sscanf(b.repr.c_str(), "junc_%d", &b_idx);
                          return a_idx < b_idx;
                        });
              StrokeGraph replayed;
              if (replay_deformations(strokes, connected_junctions, replayed)) {
                if (!viz_dir.empty()) {
                  std::vector<int> after_face_colors(replayed.faces_.size());
                  color_by_reference((int)get_color_palette().size(), s.graph_,
                                     s.face_colors_, replayed, after_face_colors);
                  s.graph_ = std::move(replayed);
                  s.face_colors_ = std::move(after_face_colors);
                }
              } else
                SPDLOG_WARN("Cannot replay the connection sequence with better "
                            "deformation method...");
            }

            final_states.emplace_back(s);

            if (!viz_dir.empty() && (solve_type == RegionSolve)) {
              std::vector<Junction> junctions;
              junctions.reserve(s.candidates_.size());
              for (size_t j = 0; j < s.candidates_.size(); ++j) {
                if (s.connectivity_[j])
                  continue;
                junctions.emplace_back(s.candidates_[j]);
              }
              std::vector<std::pair<ClassifierPrediction, VizProbabilities>> preds;
              junc2pred(s.graph_, junctions, preds, final_predictions);
              std::vector<std::string> connected_junc_strs;
              viz_connections(s, final_predictions, connected_junc_strs);

              viz_pdf(strokes, s.graph_,
                      viz_dir + "/" + std::to_string(s.state_idx_) + "_final.pdf", preds,
                      connected_junc_strs, s.face_colors_);
              viz_pdf(strokes, s.graph_,
                      viz_dir + "/" + std::to_string(s.state_idx_) + "_finalfaces.pdf",
                      std::vector<std::pair<ClassifierPrediction, VizProbabilities>>(),
                      std::vector<std::string>(), s.face_colors_, false);
            }
          } else {
            s.curr_stroke_step_++;
            connection_state_levels[i].push(std::move(s));
          }
        }
      }
    }

    // Pick the type of states
    for (auto& level_q : connection_state_levels) {
      if (level_q.empty())
        continue;

      while (!level_q.empty()) {
        connection_states.push(level_q.top());
        level_q.pop();
      }
      break;
    }
  } while (!connection_states.empty());

  if (final_states.empty())
    return -1;

  // Among all final states, pick the best one
  Float opt_obj = -std::numeric_limits<Float>::infinity();
  int opt_state_idx = -1;
  std::vector<Junction> opt_candidates;
  std::vector<bool> opt_connectivities;
  for (const auto& state : final_states) {
    Float obj = 0;
    obj = state.obj_;

    SPDLOG_INFO("Final state {}; Obj: {};", state.state_idx_, state.obj_);

    if (obj > opt_obj) {
      opt_obj = obj;
      opt_state_idx = (int)state.state_idx_;
      opt_candidates = state.candidates_;
      opt_connectivities = state.connectivity_;
    }
  }

  // Debug printout
  int ret_opt_state_idx = opt_state_idx;
  SPDLOG_INFO("Backtrack: ");
  while (prev_state.count(opt_state_idx)) {
    SPDLOG_INFO("{}", opt_state_idx);
    opt_state_idx = prev_state[opt_state_idx];
  }

  stroke_graphs.resize(intermediate_graphs.size());
  for (auto& [idx, g] : intermediate_graphs) {
    stroke_graphs[idx] = std::move(g);

    // Only replay the connection on the optimal graph
    if (idx == ret_opt_state_idx) {
      StrokeGraph replay_graph = final_graph.clone();
      std::vector<Junction> varying_candidates;
      bool succeed = replay_connection(opt_candidates, opt_connectivities, replay_graph,
                                       varying_candidates);
      if (succeed) {
        stroke_graphs[idx] = std::move(replay_graph);
        for (size_t j = 0; j < varying_candidates.size(); ++j) {
          predictions[idx].predictions[j].junction_repr = varying_candidates[j].repr;
        }
      }
    }
  }

  return ret_opt_state_idx;
}

int nonincremental_nonregion_solve(span<const Stroke> strokes,
                                   StrokeGraph::SnappingType snapping_type,
                                   FeatureType feature_type,
                                   std::vector<StrokeGraph>& stroke_graphs,
                                   std::vector<StrokeSnapInfo>& predictions,
                                   std::unordered_map<size_t, int>& prev_state,
                                   size_t max_num_strokes, size_t max_per_stroke_states,
                                   const std::string& viz_dir, bool to_bridge_) {
  // Disable region term
  region_term_ratio = 0.0;
  prediction_feature_type = feature_type;
  to_bridge = to_bridge_;
  int opt_idx = incremental_solve(strokes, snapping_type, stroke_graphs, predictions,
                                  prev_state, max_num_strokes, max_per_stroke_states,
                                  viz_dir, SolveType::CCSolve);
  for (auto& pred : predictions[opt_idx].predictions) {
    pred.alt_prob = serial_largest_non_region_gap;
  }

  return opt_idx;
}

void corner_solve(const StrokeGraph& plane_graph, StrokeGraph::SnappingType snapping_type,
                  FeatureType feature_type, StrokeSnapInfo& predictions,
                  StrokeSnapInfo& all_predictions, StrokeGraph& stroke_graph,
                  bool to_include_prev_connections_, Float largest_non_region_gap) {
  prediction_feature_type = feature_type;
  region_term_ratio = 0.0;
  include_corners = true;
  to_include_prev_connections = to_include_prev_connections_;

  if (stroke_graph.orig_strokes_.size() < plane_graph.orig_strokes_.size())
    stroke_graph = plane_graph.clone();

  GraphState init_state{stroke_graph.orig_strokes_.size() - 1,
                        StrokeGraph(snapping_type),
                        IncrementalCache(),
                        std::vector<Junction>(),
                        std::vector<bool>(),
                        std::unordered_map<std::string, Float>(),
                        std::unordered_map<std::string, Float>()};
  init_state.graph_ = plane_graph.clone();
  IncrementalCache cache;
  std::vector<RegionSolution> sols;
  precomputed_corner_decompose(init_state, stroke_graph, cache, plane_graph, sols, "",
                               largest_non_region_gap);
  if (!sols.empty()) {
    const auto& s = sols.front();

    // Dedup the corner candidates for visualization
    std::vector<Junction> dedup_candidates;
    {
      std::map<std::pair<int, int>, std::vector<size_t>> high_valence_junctions;
      for (size_t j = 0; j < s.candidates_.size(); ++j) {
        const auto& junc = s.candidates_[j];
        if (!is_corner_candidate(stroke_graph, junc)) {
          dedup_candidates.emplace_back(junc);
          // dedup_candidates.back().corner_type = dedup_candidates.back().type;
        } else {
          const auto junc_p =
            std::make_pair(std::min(to_v_idx(stroke_graph, junc.points[0]),
                                    to_v_idx(stroke_graph, junc.points[1])),
                           std::max(to_v_idx(stroke_graph, junc.points[0]),
                                    to_v_idx(stroke_graph, junc.points[1])));

          if (high_valence_junctions.count(junc_p) == 0) {
            high_valence_junctions[junc_p] = std::vector<size_t>();
          }

          high_valence_junctions[junc_p].emplace_back(j);
        }
      }

      for (const auto& [p, juncs] : high_valence_junctions) {
        for (const auto j_idx : juncs) {
          dedup_candidates.emplace_back(s.candidates_[j_idx]);
          break;
        }
      }
    }

    stroke_graph = sols.front().graph_.clone();
    for (size_t j = 0; j < dedup_candidates.size(); ++j) {
      std::vector<Junction> junctions{dedup_candidates[j]};
      std::vector<std::pair<ClassifierPrediction, VizProbabilities>> pred;
      junc2pred(stroke_graph, junctions, pred, std::vector<Junction>());
      predictions.predictions.emplace_back(pred.front().first);
      predictions.predictions.back().connected = !dedup_candidates[j].repr.empty();
    }

    for (size_t j = 0; j < s.candidates_.size(); ++j) {
      std::vector<Junction> junctions{s.candidates_[j]};
      std::vector<std::pair<ClassifierPrediction, VizProbabilities>> pred;
      junc2pred(stroke_graph, junctions, pred, std::vector<Junction>());
      all_predictions.predictions.emplace_back(pred.front().first);
      all_predictions.predictions.back().connected = !s.candidates_[j].repr.empty();
    }
  }
}

} // namespace sketching
