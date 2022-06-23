#include "incremental_region.h"

#include "../force_assert.h"
#include "../graph_color.h"
#include "../io/pdf.h"
#include "../stroke_graph_extra.h"
#include "incremental_decomposition.h"
#include "incremental_obj.h"
#include "incremental_param.h"
#include "incremental_region_util.h"
#include "incremental_util.h"

#include <queue>

namespace sketching {
size_t max_num_states_region = 3;
DecompositionType decomposition_type = DecompositionType::Probability;

namespace {

void pick_states(const std::vector<Junction>& candidates,
                 const std::vector<bool>& connectivity, const GraphState& in_state,
                 const std::set<std::pair<Float, size_t>,
                                std::greater<std::pair<Float, size_t>>>& sort_objs,
                 const std::vector<RegionSolution>& final_solutions,
                 std::vector<GraphState>& states) {
  auto find_junction_index = [&candidates](const Junction& junc) -> size_t {
    size_t j;
    for (j = 0; j < candidates.size(); ++j) {
      if (candidates[j] == junc)
        break;
    }
    return j;
  };

  // Add in the number of connected junctions
  std::set<std::tuple<Float, size_t, size_t>,
           std::greater<std::tuple<Float, size_t, size_t>>>
    sort_obj_conns;
  for (const auto [obj, i] : sort_objs) {
    const auto& sol = final_solutions[i];
    sort_obj_conns.emplace(obj, sol.junc_distance_map_.size(), i);
  }

  size_t count = 0;
  for (const auto [obj, num_conns, i] : sort_obj_conns) {
    if (count >= max_num_states_region)
      break;
    const auto& sol = final_solutions[i];

    if (!violation_check(sol))
      continue;

    // Save this state
    states.emplace_back();
    GraphState& state = states.back();
    state.curr_stroke_step_ = in_state.curr_stroke_step_;
    state.graph_ = sol.graph_.clone();
    for (auto& s : state.graph_.strokes_)
      s.ensure_arclengths();
    state.cache_ = sol.cache_;
    state.candidates_ = candidates;
    state.connectivity_ = connectivity;
    state.junc_distance_map_ = in_state.junc_distance_map_;
    state.junc_distance_map_.insert(sol.junc_distance_map_.begin(),
                                    sol.junc_distance_map_.end());
    state.region_size_cache_ = sol.region_size_cache_;

    for (size_t j = 0; j < sol.connectivity_.size(); ++j) {
      size_t k = find_junction_index(sol.candidates_[j]);
      assert(k < state.candidates_.size());
      state.candidates_[k].probability = sol.candidates_[j].probability;
      state.candidates_[k].repr = sol.candidates_[j].repr;
      if (!sol.connectivity_[j])
        continue;
      state.connectivity_[k] = true;
      // We only delete disconnected weak junctions later
      state.candidates_[k].is_weak = false;
    }

    // Update strokes. TODO: Is this necessary?
    auto& bvh = state.graph_.bvh_;
    const auto n_strokes = bvh.strokes().size();
    for (size_t si = 0; si < n_strokes; ++si) {
      bvh.full_update(si);
    }

    count++;
  }

  for (const auto [obj, num_conns, i] : sort_obj_conns) {
    if (count >= max_num_states_region)
      break;
    const auto& sol = final_solutions[i];

    if (violation_check(sol))
      continue;

    // Save this state
    states.emplace_back();
    GraphState& state = states.back();
    state.curr_stroke_step_ = in_state.curr_stroke_step_;
    state.graph_ = sol.graph_.clone();
    for (auto& s : state.graph_.strokes_)
      s.ensure_arclengths();
    state.cache_ = sol.cache_;
    state.candidates_ = candidates;
    state.connectivity_ = connectivity;
    state.junc_distance_map_ = in_state.junc_distance_map_;
    state.junc_distance_map_.insert(sol.junc_distance_map_.begin(),
                                    sol.junc_distance_map_.end());
    state.region_size_cache_ = sol.region_size_cache_;

    for (size_t j = 0; j < sol.connectivity_.size(); ++j) {
      size_t k = find_junction_index(sol.candidates_[j]);
      assert(k < state.candidates_.size());
      state.candidates_[k].probability = sol.candidates_[j].probability;
      state.candidates_[k].repr = sol.candidates_[j].repr;
      if (!sol.connectivity_[j])
        continue;
      state.connectivity_[k] = true;
      // We only delete disconnected weak junctions later
      state.candidates_[k].is_weak = false;
    }

    // Update strokes. TODO: Is this necessary?
    auto& bvh = state.graph_.bvh_;
    const auto n_strokes = bvh.strokes().size();
    for (size_t si = 0; si < n_strokes; ++si) {
      bvh.full_update(si);
    }

    count++;
  }
}

bool process_region_state(const StrokeGraph& stroke_graph, IncrementalCache& cache,
                          const std::vector<Junction>& candidates,
                          size_t current_junc_pos,
                          const std::vector<bool>& junction_connected,
                          const std::unordered_map<std::string, Float>& region_size_cache,
                          std::queue<RegionState>& connection_states, RegionSolution& sol,
                          const size_t max_pos = -1, const size_t start_pos = 0,
                          bool check_region_hard = true) {
  sol.region_size_cache_ = region_size_cache;

  std::string binary_str = "";
  size_t num_connections = 0;
  for (auto c : junction_connected) {
    binary_str += (c) ? "1" : "0";
    if (c)
      num_connections++;
  }

  // Verification
  SPDLOG_DEBUG(binary_str);
  SPDLOG_DEBUG("---------------");

  // This is an invalid candidate that is only added here for the high-valence check
  // inside modify_graph. We should never connect this candidate.
  if (junction_connected[current_junc_pos] &&
      candidates[current_junc_pos].must_disconnect)
    return false;

  // Early trimming
  // Junctions with prob = 0
  if (junction_connected[current_junc_pos] &&
      candidates[current_junc_pos].probability < std::numeric_limits<double>::epsilon()) {
    return false;
  }

  // Early trimming: impossible interior-interior connection
  if (junction_connected[current_junc_pos]) {
    std::vector<Junction> connected_candidates;
    for (size_t i = 0; i <= current_junc_pos; ++i) {
      if (junction_connected[i])
        connected_candidates.emplace_back(candidates[i]);
    }
    if (implies_interior_interior(stroke_graph, connected_candidates))
      return false;
  }

  std::vector<std::pair<size_t, size_t>> adj_faces;
  std::vector<Float> junc_distances;
  std::vector<StrokeGraph::VertexID> junc_vertex_ids;
  sol.candidates_ = candidates;
  auto varying_stroke_graph =
    modify_graph(stroke_graph, sol.candidates_, junction_connected, adj_faces,
                 junc_distances, junc_vertex_ids);

  if (!varying_stroke_graph) {
    // Avoid further branching this assignment since it creates geometric changes that
    // are too large
    // SPDLOG_INFO("modify_graph: Failed.");
    return false;
  }

  // Check if this assignment is legal (the junction distance < the adjacent maximum
  // inscribing circle radiuses)
  Index highest_violation = -1;
  for (size_t i = 0; check_region_hard && i < adj_faces.size(); ++i) {
    // Check if this candidate is in the range to check
    /*bool found_junc = false;
    for (size_t j = start_pos; j < sol.candidates_.size() && j < max_pos; ++j) {
      if (sol.candidates_[j].repr == junc_vertex_ids[i].repr()) {
        found_junc = true;
        break;
      }
    }
    if (!found_junc)
      continue;*/

    if (!hard_region_condition_check(*varying_stroke_graph, sol.region_size_cache_,
                                     adj_faces[i].first, junc_distances[i])) {
      highest_violation = i;
      SPDLOG_DEBUG("Failed: {}; junc: {} <= {} vs {}", binary_str, i + 1, f1_size,
                   junc_distances[i]);
      break;
    }
    if (!hard_region_condition_check(*varying_stroke_graph, sol.region_size_cache_,
                                     adj_faces[i].second, junc_distances[i])) {
      highest_violation = i;
      SPDLOG_DEBUG("Failed: {}; junc: {} <= {} vs {}", binary_str, i + 1, f2_size,
                   junc_distances[i]);
      break;
    }
  }
  const auto pass_check = highest_violation == -1;
  if (!pass_check) {
    SPDLOG_DEBUG("\t=> junc: {} / {}", highest_violation + 1, adj_faces.size());
    SPDLOG_DEBUG("\t=> junc: {} / {}", highest_violation + 1, num_connections);
  }

  // Add to the queue
  // Since we can only make regions smaller by connecting more, we don't branch the
  // states that already fail the test.
  if (current_junc_pos + 1 < max_pos &&
      current_junc_pos + 1 < junction_connected.size() && pass_check)
    connection_states.push(
      RegionState{current_junc_pos + 1, junction_connected, sol.region_size_cache_});

  if (pass_check) {
    std::unordered_map<std::string, Float> junc_distance_map;
    for (size_t i = 0; i < junc_vertex_ids.size(); ++i) {
      junc_distance_map[junc_vertex_ids[i].repr()] = junc_distances[i];
    }
    sol.graph_ = std::move(*varying_stroke_graph);
    sol.cache_ = cache;
    sol.connectivity_ = junction_connected;
    sol.junc_distance_map_ = std::move(junc_distance_map);
  }

  return pass_check;
}

} // namespace

bool highly_likely_junctions_connected(const RegionSolution& sol, Float sol_obj,
                                       const GraphState& in_state,
                                       Float highly_likely_prob, Float certain_prob) {
  std::unordered_map<std::string, Float> in_junc_distance_map = sol.junc_distance_map_;
  for (const auto& [k, v] : in_state.junc_distance_map_)
    in_junc_distance_map[k] = v;
  auto in_candidates = sol.candidates_;
  in_candidates.insert(in_candidates.end(), in_state.candidates_.begin(),
                       in_state.candidates_.end());
  auto in_connectivity = sol.connectivity_;
  in_connectivity.insert(in_connectivity.end(), in_state.connectivity_.begin(),
                         in_state.connectivity_.end());

  std::unordered_set<size_t> checked_indices;

  // Check all not connected junctions
  for (size_t i = 0; i < sol.candidates_.size(); ++i) {
    if (sol.connectivity_.at(i) ||
        sol.candidates_.at(i).probability < highly_likely_prob ||
        checked_indices.count(i))
      continue;

    // Check if all expanded candidates fulfill the condition
    std::set<size_t> binding;
    expand_trial_candidate(sol.graph_, sol.candidates_, i, binding);

    // This is either unable to be snapped or is topologically invalid
    if (!binding.count(i))
      continue;

    bool multi_likely = true;
    bool multi_certain = true;
    for (const auto idx : binding) {
      // TODO: This is disabled for variable decomposition
      // assert(!sol.connectivity_.at(idx));
      checked_indices.emplace(idx);
      if (sol.candidates_.at(idx).probability < highly_likely_prob) {
        multi_likely = false;
        multi_certain = false;
      } else if (sol.candidates_.at(idx).probability < certain_prob) {
        multi_certain = false;
      }
    }

    if (!multi_likely)
      continue;

    // Is it possible to connect this junction without violation
    auto candidates = in_candidates;
    std::unordered_map<std::string, Float> junc_distance_map = in_junc_distance_map;
    auto modified_graph = try_connect_graph(sol.graph_, sol.region_size_cache_,
                                            junc_distance_map, candidates, binding);
    if (!modified_graph)
      continue; // Connection not possible.

    // Check the entire graph
    bool is_valid = is_sol_graph_valid(*modified_graph, junc_distance_map, candidates,
                                       in_state, sol.region_size_cache_, certain_prob);
    if (!is_valid)
      continue;

    // If this connection is valid and has a high/certain probability. This should always
    // be chosen.
    if (multi_certain)
      return false;

    // Do we see a region term gain if we connect this junction instead?
    // Only consider the varying junction in the input solution and the newly connected
    // one.
    RegionSolution tmp_sol;
    tmp_sol.graph_ = std::move(*modified_graph);
    tmp_sol.candidates_ = candidates;
    tmp_sol.junc_distance_map_ = junc_distance_map;
    tmp_sol.connectivity_ = in_connectivity;
    tmp_sol.connectivity_[i] = true;
    Float region_obj;
    obj_function(tmp_sol, region_obj);

    // Saw a region term increase. We should always connect this junction. This state is
    // invalid.
    if (region_obj > sol_obj)
      return false;
  }

  return true;
}

void enumerate_subproblem(std::queue<RegionState>& connection_states,
                          const GraphState& in_state, const StrokeGraph& stroke_graph,
                          IncrementalCache& cache, const StrokeGraph& final_graph,
                          const std::vector<Junction>& varying_candidates,
                          const int max_num_sols, const size_t max_pos,
                          std::vector<RegionSolution>& sols, bool check_region_hard) {
  assert(!connection_states.empty());
  size_t start_pos = connection_states.front().cur_variable_;

  // Check hard constraints
  std::vector<RegionSolution> final_solutions;
  size_t start_q_size = connection_states.size();
  size_t itr = 0;
  while (!connection_states.empty()) {
    auto state = connection_states.front();
    connection_states.pop();

    // Check the two possibilities
    auto junction_connected = state.variable_state_;
    junction_connected[state.cur_variable_] = false;
    if (itr < start_q_size) {
      final_solutions.emplace_back();
      bool pass_check = process_region_state(
        stroke_graph, cache, varying_candidates, state.cur_variable_, junction_connected,
        state.region_size_cache_, connection_states, final_solutions.back(), max_pos,
        start_pos, check_region_hard);
      if (!pass_check) {
        final_solutions.pop_back();
      }
    }
    // We've already checked this state from previous rounds, only extend
    else if (state.cur_variable_ + 1 < max_pos &&
             state.cur_variable_ + 1 < junction_connected.size())
      connection_states.push(RegionState{state.cur_variable_ + 1, junction_connected,
                                         state.region_size_cache_});

    junction_connected[state.cur_variable_] = true;
    final_solutions.emplace_back();
    bool pass_check = process_region_state(
      stroke_graph, cache, varying_candidates, state.cur_variable_, junction_connected,
      state.region_size_cache_, connection_states, final_solutions.back(), max_pos,
      start_pos, check_region_hard);
    if (!pass_check) {
      final_solutions.pop_back();
    }

    itr++;
  }

  if (max_num_sols >= 0) {
    // Sort and pick the top max_num_sols intermidate states
    std::vector<
      std::set<std::pair<Float, size_t>, std::greater<std::pair<Float, size_t>>>>
      sort_objs;
    sort_objs.resize(3);

    // When we are at the end of a component (high-valence), we can check the validity.
    // Note the assumption is max_num_sub_region_sols != max_num_states_region (default:
    // 5)
    bool to_check_high_valence = (max_num_sols != max_num_sub_region_sols);
    sort_solutions(final_solutions, in_state, final_graph, sort_objs[0], sort_objs[2],
                   sort_objs[1], to_check_high_valence);

    sols.reserve(max_num_sols);
    // For now, keep all states (even the unlikely ones)
    auto const& objs = sort_objs[2];
    if (!objs.empty()) {
      // Add in the number of connected junctions
      std::set<std::tuple<Float, size_t, size_t>,
               std::greater<std::tuple<Float, size_t, size_t>>>
        sort_obj_conns;
      for (const auto [obj, i] : objs) {
        const auto& sol = final_solutions[i];
        sort_obj_conns.emplace(obj, sol.junc_distance_map_.size(), i);
      }
      for (auto const& [obj, num_conns, idx] : sort_obj_conns) {
        sols.emplace_back();
        sols.back() = std::move(final_solutions[idx]);
        if (sols.size() == max_num_sols)
          break;
      }
      return;
    }
  } else {
    // The check and sort is done outside this function
    sols = std::move(final_solutions);
  }
}

std::unique_ptr<StrokeGraph>
try_connect_graph(const StrokeGraph& varying_stroke_graph,
                  const std::unordered_map<std::string, Float>& in_region_size_cache,
                  std::unordered_map<std::string, Float>& junc_distance_map,
                  std::vector<Junction>& candidates, const std::set<size_t>& binding) {
  std::vector<std::pair<size_t, size_t>> adj_faces;
  std::vector<Float> junc_distances;
  std::vector<StrokeGraph::VertexID> junc_vertex_ids;
  std::vector<Junction> trial_candidate;
  std::unordered_map<size_t, size_t> to_cand;
  for (const auto idx : binding) {
    to_cand[trial_candidate.size()] = idx;
    trial_candidate.emplace_back(candidates[idx]);
  }
  std::vector<bool> connected;
  connected.resize(trial_candidate.size(), true);

  // Update bounding boxes
  auto out_graph = modify_graph(varying_stroke_graph, trial_candidate, connected,
                                adj_faces, junc_distances, junc_vertex_ids);
  if (!out_graph)
    return nullptr; // Connection not possible.

  for (size_t j = 0; j < junc_distances.size(); ++j) {
    junc_distance_map[junc_vertex_ids[j].repr()] = junc_distances[j];
  }
  for (size_t j = 0; j < trial_candidate.size(); ++j) {
    candidates[to_cand[j]] = trial_candidate[j];
  }
  std::unordered_map<std::string, Float> region_size_cache = in_region_size_cache;

  // Check if valid
  for (size_t j = 0; j < adj_faces.size(); ++j) {
    if (!hard_region_condition_check(*out_graph, region_size_cache, adj_faces[j].first,
                                     junc_distances[j])) {
      return nullptr;
    }
    if (!hard_region_condition_check(*out_graph, region_size_cache, adj_faces[j].second,
                                     junc_distances[j])) {
      return nullptr;
    }
  }

  return out_graph;
}

void region_solve(span<const Stroke> strokes, const StrokeGraph& final_graph,
                  const std::vector<Junction>& final_predictions,
                  const GraphState& in_state, const StrokeGraph& viz_before_graph,
                  const std::string& viz_dir, size_t state_count, size_t cur_stroke_step,
                  std::vector<std::vector<GraphState>>& state_levels,
                  const CandidateProposalType candidate_proposal_type) {
  const StrokeGraph& stroke_graph = in_state.graph_;
  IncrementalCache cache = in_state.cache_;

  // Create three versions of output
  state_levels.clear();
  state_levels.resize(3);

  // 1. Find all new candidates
  std::vector<Junction> new_candidates;
  if (candidate_proposal_type == Incremental)
    propose_candidates_incremental(strokes, final_graph, final_predictions, in_state,
                                   cur_stroke_step, new_candidates, false);
  else {
    propose_candidates_final(strokes, final_graph, final_predictions, in_state,
                             cur_stroke_step, new_candidates);
  }

  // Can't find candidate junctions, return the same input state (waiting for a new
  // stroke)
  if (new_candidates.empty()) {
    state_levels[0].emplace_back(in_state);
    if (state_levels[0].back().face_colors_.size() !=
        state_levels[0].back().graph_.faces_.size()) {
      state_levels[0].back().face_colors_.resize(
        state_levels[0].back().graph_.faces_.size());
      map_color(state_levels[0].back().graph_, (int)get_color_palette().size(),
                state_levels[0].back().face_colors_);
    }
    if (!viz_dir.empty() && (candidate_proposal_type == Incremental)) {
      std::vector<std::pair<ClassifierPrediction, VizProbabilities>> predictions;
      junc2pred(in_state.graph_, std::vector<Junction>(), predictions, final_predictions);
      std::vector<std::string> connected_junc_strs;
      viz_connections(in_state, final_predictions, connected_junc_strs);

      // Note here in_state.face_colors_ hasn't been updated since adding the new stroke
      viz_pdf(strokes, viz_before_graph,
              viz_dir + "/" + std::to_string(state_count) + "_new" + ".pdf", predictions,
              connected_junc_strs, in_state.face_colors_);
    }
    return;
  }

  // 2. Complete endpoint graphs given the past record and new candidate sets
  {
    std::vector<Junction> complementary_new_candidates;
    std::vector<Junction> to_complete_candidates;
    to_complete_candidates.reserve(in_state.candidates_.size() + new_candidates.size());
    to_complete_candidates.insert(to_complete_candidates.end(),
                                  in_state.candidates_.begin(),
                                  in_state.candidates_.end());
    to_complete_candidates.insert(to_complete_candidates.end(), new_candidates.begin(),
                                  new_candidates.end());
    complete_graph_candidates(stroke_graph, strokes, final_graph, cur_stroke_step,
                              to_complete_candidates, complementary_new_candidates);
    new_candidates.insert(new_candidates.end(), complementary_new_candidates.begin(),
                          complementary_new_candidates.end());
  }

  // 3. Add new variables to the record set and assemble the varying set with the current
  // predictions
  std::vector<Junction> varying_candidates, out_candidates;
  std::vector<bool> out_connectivity;
  update_candidate_record(stroke_graph, in_state.candidates_, in_state.connectivity_,
                          new_candidates, varying_candidates, out_candidates,
                          out_connectivity);

  // Can't find candidate junctions, return the same input state (waiting for a new
  // stroke)
  if (varying_candidates.empty()) {
    state_levels[0].emplace_back(in_state);
    if (state_levels[0].back().face_colors_.size() !=
        state_levels[0].back().graph_.faces_.size()) {
      state_levels[0].back().face_colors_.resize(
        state_levels[0].back().graph_.faces_.size());
      map_color(state_levels[0].back().graph_, (int)get_color_palette().size(),
                state_levels[0].back().face_colors_);
    }
    if (!viz_dir.empty() && (candidate_proposal_type == Incremental)) {
      std::vector<std::pair<ClassifierPrediction, VizProbabilities>> predictions;
      junc2pred(in_state.graph_, std::vector<Junction>(), predictions, final_predictions);
      std::vector<std::string> connected_junc_strs;
      viz_connections(in_state, final_predictions, connected_junc_strs);
      viz_pdf(strokes, viz_before_graph,
              viz_dir + "/" + std::to_string(state_count) + "_new" + ".pdf", predictions,
              connected_junc_strs, in_state.face_colors_);
    }
    return;
  }

  // Visualize the state: with the newly added stroke and the new junctions introduced
  // by it
  if (!viz_dir.empty() && (candidate_proposal_type == Incremental)) {
    std::vector<Junction> junctions;
    junctions.reserve(new_candidates.size());
    for (size_t i = 0; i < new_candidates.size(); ++i) {
      junctions.emplace_back(new_candidates[i]);
    }
    std::vector<std::pair<ClassifierPrediction, VizProbabilities>> predictions;
    junc2pred(in_state.graph_, junctions, predictions, final_predictions);
    std::vector<std::string> connected_junc_strs;
    viz_connections(in_state, final_predictions, connected_junc_strs);
    viz_pdf(strokes, viz_before_graph,
            viz_dir + "/" + std::to_string(state_count) + "_new" + ".pdf", predictions,
            connected_junc_strs, in_state.face_colors_);
  }
  if (!viz_dir.empty() && (candidate_proposal_type == Incremental)) {
    std::vector<Junction> junctions;
    junctions.reserve(varying_candidates.size());
    for (size_t i = 0; i < varying_candidates.size(); ++i) {
      junctions.emplace_back(varying_candidates[i]);
    }
    std::vector<std::pair<ClassifierPrediction, VizProbabilities>> predictions;
    junc2pred(in_state.graph_, junctions, predictions, final_predictions);
    std::vector<std::string> connected_junc_strs;
    viz_connections(in_state, final_predictions, connected_junc_strs);

    const auto before_face_colors = in_state.face_colors_;
    std::vector<int> after_face_colors(in_state.graph_.faces_.size());
    color_by_reference((int)get_color_palette().size(), viz_before_graph,
                       before_face_colors, in_state.graph_, after_face_colors);

    viz_pdf(strokes, in_state.graph_,
            viz_dir + "/" + std::to_string(state_count) + "_newunknown" + ".pdf",
            predictions, connected_junc_strs, after_face_colors);
  }
  //

  // 3. Enumerate local decisions and compute corresponding relative awards
  // Recursizely enumerate on the junction candiates
  std::queue<RegionState> connection_states;
  {
    std::vector<bool> junction_connected;
    junction_connected.resize(varying_candidates.size(), false);
    connection_states.push(
      RegionState{0, junction_connected, in_state.region_size_cache_});
  }

  size_t actual_varying_count = 0;
  for (auto const& junc : varying_candidates) {
    if (!junc.must_disconnect && junc.probability > 0)
      actual_varying_count++;
  }
  SPDLOG_INFO("\tVarying: {} / {}", actual_varying_count, varying_candidates.size());

  std::vector<RegionSolution> final_solutions;
  if (decomposition_type == DecompositionType::Connected)
    connected_component_decompose(in_state, stroke_graph, cache, final_graph,
                                  varying_candidates, max_num_sub_region_vars,
                                  max_num_sub_region_sols, final_solutions);
  else if (!include_corners) {
    print_junctions(varying_candidates);
    probability_decompose(in_state, stroke_graph, cache, final_graph, varying_candidates,
                          max_num_sub_region_vars, max_num_sub_region_sols,
                          final_solutions, viz_dir);
  } else {
    print_junctions(varying_candidates);
    corner_decompose(in_state, stroke_graph, cache, final_graph, varying_candidates,
                     final_solutions, viz_dir);
  }

  // Compute costs of feasible solutions and save the promising ones
  std::set<std::pair<Float, size_t>, std::greater<std::pair<Float, size_t>>> sort_objs,
    sort_objs_all, sort_objs_likely;
  sort_solutions(final_solutions, in_state, final_graph, sort_objs, sort_objs_all,
                 sort_objs_likely, true);

  pick_states(out_candidates, out_connectivity, in_state, sort_objs, final_solutions,
              state_levels[0]);
  pick_states(out_candidates, out_connectivity, in_state, sort_objs_likely,
              final_solutions, state_levels[1]);
  pick_states(out_candidates, out_connectivity, in_state, sort_objs_all, final_solutions,
              state_levels[2]);

  assert(state_levels.back().size() >= state_levels[0].size() &&
         state_levels.back().size() >= state_levels[1].size());
  assert(!state_levels.back().empty());
}

} // namespace sketching
