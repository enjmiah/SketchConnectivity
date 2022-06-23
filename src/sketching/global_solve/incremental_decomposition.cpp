#include "incremental_decomposition.h"

#include "../force_assert.h"
#include "../stroke_graph.h"
#include "../stroke_graph_extra.h"
#include "incremental_obj.h"
#include "incremental_param.h"
#include "incremental_region.h"
#include "incremental_region_util.h"

#include <numeric>

#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(disable : 4245) // conversion, signed/unsigned mismatch
#pragma warning(disable : 4244 4267) // conversion, possible loss of data
#endif

namespace sketching {
bool high_valence_prob_update = false;
bool to_include_prev_connections = false;

Float min_decomp_prob = 0.2;
size_t max_component_size = 16;

// Float largest_gap_ratio = 3;
Float largest_gap_ratio = std::numeric_limits<Float>::infinity();
Float serial_largest_non_region_gap = std::numeric_limits<Float>::infinity();

namespace {
void junction_decomposition_sort(std::vector<Junction>& candidates,
                                 std::vector<int>& junc_components) {
  if (candidates.empty())
    return;

  junc_components.clear();

  std::vector<int> components;
  components.reserve(candidates.size());
  for (const auto& junc : candidates) {
    components.emplace_back(junc.component_idx);
  }

  // Code for merging based on belonging faces
  /*int max_color = 0;
  components.resize(candidates.size(), -1);
  components[0] = max_color++;
  for (size_t i = 0; i + 1 < candidates.size(); ++i) {
    for (size_t j = i + 1; j < candidates.size(); ++j) {
    if (!(candidates[i].fi != candidates[j].fi &&
        candidates[i].component_idx != candidates[j].component_idx)) {
      if (components[i] == -1 && components[j] == -1) {
      components[i] = components[j] = max_color++;
      } else if (components[j] != components[i]) {
      int min_c = std::min(components[i], components[j]);
      int max_c = std::max(components[i], components[j]);
      if (min_c == -1)
        components[i] = components[j] = max_c;
      else {
        for (auto& c : components)
        if (c == min_c)
          c = max_c;
      }
      }
    }
    }
  }

  for (auto& c : components)
    if (c == -1)
    c = max_color++;*/

  std::vector<int> indices(candidates.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(),
            [&](int i, int j) -> bool { return components[i] < components[j]; });

  junc_components.reserve(candidates.size());
  std::vector<Junction> out_candidates;
  out_candidates.reserve(candidates.size());
  for (const auto idx : indices) {
    junc_components.emplace_back(components[idx]);
    out_candidates.emplace_back(candidates[idx]);
  }

  candidates = std::move(out_candidates);
}

void component_sort(const std::map<size_t, Float>& component_values,
                    std::vector<Junction>& candidates,
                    std::vector<int>& junc_components) {
  if (candidates.empty())
    return;
  std::set<std::tuple<Float, int, size_t>, std::greater<std::tuple<Float, int, size_t>>>
    sort_component;
  for (const auto [c, v] : component_values) {
    int latest_sid = -1;
    for (const auto& junc : candidates) {
      if (junc.component_idx == c) {
        latest_sid = std::max(latest_sid, junc.points[0].first);
        latest_sid = std::max(latest_sid, junc.points[1].first);
      }
    }
    // Flip the sign to pick the earlier ones
    sort_component.emplace(std::make_tuple(v, -latest_sid, c));
  }

  junc_components.clear();
  junc_components.reserve(candidates.size());
  std::vector<Junction> out_candidates;
  out_candidates.reserve(candidates.size());
  for (const auto [v, order, c] : sort_component) {
    std::vector<Junction> comp_candidates;
    for (const auto& junc : candidates) {
      if (junc.component_idx == c) {
        comp_candidates.emplace_back(junc);
        comp_candidates.back().repr = "";
        comp_candidates.back().component_idx = c;
        junc_components.emplace_back(c);
      }
    }

    // Sort within component: probability then drawing order
    junction_probability_sort(comp_candidates);
    out_candidates.insert(out_candidates.end(), comp_candidates.begin(),
                          comp_candidates.end());
  }
  assert(out_candidates.size() == candidates.size());
  candidates = std::move(out_candidates);
}

} // namespace

void remove_trivial_violation(const StrokeGraph& stroke_graph,
                              std::vector<Junction>& candidates) {
  // Try connect each junction one by one. If there's a violation, if it's the only
  // junction in the component, remove the candidate; otherwise, reduce it to a
  // must-be-disconnected weak candidate.
  std::map<size_t, std::vector<size_t>> group_indices;
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (candidates[i].component_idx < 0)
      continue;
    if (!group_indices.count(candidates[i].component_idx))
      group_indices[candidates[i].component_idx] = std::vector<size_t>();
    group_indices[candidates[i].component_idx].emplace_back(i);
  }

  std::unordered_set<size_t> trivial_violations;
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (candidates[i].component_idx < 0 || candidates[i].must_disconnect)
      continue;
    StrokeGraph varying_stroke_graph = stroke_graph.clone();
    std::vector<std::pair<size_t, size_t>> adj_faces;
    std::vector<Float> junc_distances;
    std::vector<StrokeGraph::VertexID> junc_vertex_ids;
    std::vector<Junction> trial_candidate = candidates;
    std::vector<bool> connected;
    connected.resize(trial_candidate.size(), false);
    connected[i] = true;

    auto out_graph = modify_graph(varying_stroke_graph, trial_candidate, connected,
                                  adj_faces, junc_distances, junc_vertex_ids);
    if (!out_graph) {
      if (!candidates[i].is_weak &&
          group_indices[candidates[i].component_idx].size() == 1)
        trivial_violations.emplace(i);
      else {
        candidates[i].is_weak = true;
        candidates[i].must_disconnect = true;
      }
    }

    std::unordered_map<std::string, Float> pseudo_cache;
    for (size_t j = 0; j < adj_faces.size(); ++j) {
      pseudo_cache.clear();
      if (!hard_region_condition_check(*out_graph, pseudo_cache, adj_faces[j].first,
                                       junc_distances[j])) {
        if (!candidates[i].is_weak &&
            group_indices[candidates[i].component_idx].size() == 1)
          trivial_violations.emplace(i);
        else {
          candidates[i].is_weak = true;
          candidates[i].must_disconnect = true;
        }
        break;
      }
      pseudo_cache.clear();
      if (!hard_region_condition_check(*out_graph, pseudo_cache, adj_faces[j].second,
                                       junc_distances[j])) {
        if (!candidates[i].is_weak &&
            group_indices[candidates[i].component_idx].size() == 1)
          trivial_violations.emplace(i);
        else {
          candidates[i].is_weak = true;
          candidates[i].must_disconnect = true;
        }
        break;
      }
    }
  }

  std::vector<Junction> out_candidates;
  out_candidates.reserve(candidates.size());
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (trivial_violations.count(i))
      continue;
    out_candidates.emplace_back(candidates[i]);
  }
  candidates = std::move(out_candidates);
}

void remove_trivial_corner_violation(const StrokeGraph& stroke_graph,
                                     std::vector<Junction>& candidates) {
  std::vector<Junction> out_candidates;
  std::map<std::pair<int, int>, std::vector<size_t>> high_valence_junctions;
  for (size_t i = 0; i < candidates.size(); ++i) {
    const auto& junc = candidates[i];
    if (!is_corner_candidate(stroke_graph, junc) || candidates[i].component_idx < 0) {
      out_candidates.emplace_back(junc);
    } else {
      const auto junc_p =
        std::make_pair(std::min(to_v_idx(stroke_graph, junc.points[0]),
                                to_v_idx(stroke_graph, junc.points[1])),
                       std::max(to_v_idx(stroke_graph, junc.points[0]),
                                to_v_idx(stroke_graph, junc.points[1])));

      if (high_valence_junctions.count(junc_p) == 0) {
        high_valence_junctions[junc_p] = std::vector<size_t>();
      }

      high_valence_junctions[junc_p].emplace_back(i);
    }
  }

  // The corner candidates with the same ancestor should have the same label
  for (const auto& [p, juncs] : high_valence_junctions) {
    bool seen_must_disconnect = false;
    for (const auto j_idx : juncs) {
      const auto& junc = candidates[j_idx];
      if (junc.must_disconnect) {
        seen_must_disconnect = true;
        break;
      }
    }
    if (seen_must_disconnect) {
      for (const auto j_idx : juncs) {
        auto& junc = candidates[j_idx];
        junc.is_weak = true;
        junc.must_disconnect = true;
      }
    }
  }

  // Remove the components that only have weak candidates
  std::map<size_t, std::vector<size_t>> group_indices;
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (candidates[i].component_idx < 0)
      continue;
    if (!group_indices.count(candidates[i].component_idx))
      group_indices[candidates[i].component_idx] = std::vector<size_t>();
    group_indices[candidates[i].component_idx].emplace_back(i);
  }

  std::unordered_set<size_t> to_remove_components;
  for (const auto& [c, indices] : group_indices) {
    bool seen_non_weak = false;
    for (const auto i : indices) {
      if (!candidates[i].is_weak) {
        seen_non_weak = true;
        break;
      }
    }
    if (!seen_non_weak) {
      to_remove_components.emplace(c);
    }
  }

  for (size_t i = 0; i < candidates.size(); ++i) {
    const auto& junc = candidates[i];
    if (is_corner_candidate(stroke_graph, junc) && candidates[i].component_idx >= 0 &&
        !to_remove_components.count(junc.component_idx)) {
      out_candidates.emplace_back(junc);
    }
  }
  candidates = std::move(out_candidates);
}

void decompose_candidates(const StrokeGraph& stroke_graph, span<Junction> candidates,
                          const Float min_prob) {
  // Decompose based on connected components
  std::vector<int> junc_components;
  color_endpoint_graph(stroke_graph, candidates, junc_components, min_prob);

  assert(junc_components.size() == candidates.size());
  for (size_t i = 0; i < candidates.size(); ++i) {
    candidates[i].component_idx = junc_components[i];

    // Decompose based on containing region
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
}

/// Limit the number of intermediate states to max_num_sols every max_num_vars variables.
void variable_decompose(const GraphState& in_state, const StrokeGraph& stroke_graph,
                        IncrementalCache& cache, const StrokeGraph& final_graph,
                        const std::vector<Junction>& varying_candidates,
                        const std::vector<size_t>& max_num_vars,
                        const std::vector<int>& max_num_sols,
                        std::vector<RegionSolution>& sols,
                        bool check_region_hard = true) {
  force_assert(std::accumulate(max_num_vars.begin(), max_num_vars.end(), size_t(0)) ==
               varying_candidates.size());
  std::queue<RegionState> connection_states;
  {
    std::vector<bool> junction_connected;
    junction_connected.resize(varying_candidates.size(), false);
    connection_states.push(
      RegionState{0, junction_connected, in_state.region_size_cache_});
  }

  std::vector<RegionSolution> final_solutions;
  size_t cur_pos = 0;
  size_t itr = 0;
  size_t next_pos = 0;
  do {
    // Prepare the next round
    next_pos += max_num_vars[itr];
    for (auto& sol : final_solutions) {
      connection_states.push(RegionState{cur_pos, std::move(sol.connectivity_),
                                         std::move(sol.region_size_cache_)});
    }

    // Avoid overly using the memory
    size_t actual_varying_count = 0;
    std::vector<Junction> debug_juncs;
    for (size_t i = cur_pos; i < next_pos; ++i) {
      const auto& junc = varying_candidates[i];
      debug_juncs.emplace_back(junc);
      if (!junc.must_disconnect && junc.probability > 0)
        actual_varying_count++;
    }
    SPDLOG_INFO("Round {} / {} inputs:", itr + 1, max_num_sols.size());
    print_junctions(debug_juncs);

    if (actual_varying_count > max_component_size)
      throw std::runtime_error("Large component");

    final_solutions.clear();
    int cur_max_num_sols = max_num_sols[itr++];
    enumerate_subproblem(connection_states, in_state, stroke_graph, cache, final_graph,
                         varying_candidates, cur_max_num_sols, next_pos, final_solutions,
                         check_region_hard);
    cur_pos = next_pos;

    {
      SPDLOG_INFO("Round {} results:", itr);
      for (const auto& sol : final_solutions) {
        print_junctions(sol.candidates_);
        break;
      }
    }
  } while (next_pos < varying_candidates.size());

  sols = std::move(final_solutions);
}

void variable_decompose(const GraphState& in_state, const StrokeGraph& stroke_graph,
                        IncrementalCache& cache, const StrokeGraph& final_graph,
                        const std::vector<Junction>& varying_candidates,
                        const size_t max_num_vars, const size_t max_num_sols,
                        std::vector<RegionSolution>& sols) {
  std::vector<size_t> max_num_var_list;
  std::vector<int> max_num_sol_list;
  max_num_var_list.resize(varying_candidates.size() / max_num_vars, max_num_vars);
  if (varying_candidates.size() % max_num_vars)
    max_num_var_list.emplace_back(varying_candidates.size() % max_num_vars);
  max_num_sol_list.resize(max_num_var_list.size(), max_num_sols);
  max_num_var_list.back() = -1;
  variable_decompose(in_state, stroke_graph, cache, final_graph, varying_candidates,
                     max_num_var_list, max_num_sol_list, sols, true);
}

/// Decompose variables based on the regions they belong to.
void connected_component_decompose(
  const GraphState& in_state, const StrokeGraph& stroke_graph, IncrementalCache& cache,
  const StrokeGraph& final_graph, std::vector<Junction>& varying_candidates,
  const size_t max_num_vars, const size_t max_num_sols, std::vector<RegionSolution>& sols,
  int last_num_states, Float* trivial_largest_gap) {
  // 1. Decompose based on the connected component and regions
  decompose_candidates(stroke_graph, varying_candidates);
  std::vector<int> junc_components;
  junction_decomposition_sort(varying_candidates, junc_components);

  // I. Solve without the region hard constraint
  std::vector<RegionSolution> first_sols;
  {
    // 2. Get the subproblem variable counts
    std::vector<size_t> max_num_var_list1;
    max_num_var_list1.emplace_back(1);
    for (size_t i = 1; i < junc_components.size(); ++i) {
      if (junc_components[i] != junc_components[i - 1])
        max_num_var_list1.emplace_back(1);
      else
        max_num_var_list1.back()++;
    }

    // 3. Further break down
    std::vector<size_t> max_num_var_list2;
    std::vector<int> max_num_sol_list2;
    for (const auto& vars : max_num_var_list1) {
      SPDLOG_INFO("\t\tComponent size: {}", vars);

      std::vector<size_t> max_num_var_list;
      std::vector<int> max_num_sol_list;
      max_num_var_list.resize(vars / max_num_vars, max_num_vars);
      if (vars % max_num_vars)
        max_num_var_list.emplace_back(vars % max_num_vars);
      max_num_sol_list.resize(max_num_var_list.size(), max_num_sols);
      // For individual parts, the enumerations can run separately so we don't need to
      // keep too many intermediate states.
      max_num_sol_list.back() = 1;

      max_num_var_list2.insert(max_num_var_list2.end(), max_num_var_list.begin(),
                               max_num_var_list.end());
      max_num_sol_list2.insert(max_num_sol_list2.end(), max_num_sol_list.begin(),
                               max_num_sol_list.end());
    }
    // Skip the last check: -1
    max_num_sol_list2.back() = 1;

    // Printout the actual varying variable counts
    for (size_t i = 0, start_i = 0; i < max_num_var_list1.size();
         start_i += max_num_var_list1[i++]) {
      size_t actual_varying_count = 0;
      for (size_t j = start_i; j < start_i + max_num_var_list1[i]; ++j) {
        auto const& junc = varying_candidates[j];
        if (!junc.must_disconnect && junc.probability > 0)
          actual_varying_count++;
      }
      SPDLOG_INFO("\t\tComponent varying size: {}", actual_varying_count);
    }

    // Avoid overly using the memory
    for (size_t i = 0, start_i = 0; i < max_num_var_list2.size();
         start_i += max_num_var_list2[i++]) {
      size_t actual_varying_count = 0;
      for (size_t j = start_i; j < start_i + max_num_var_list2[i]; ++j) {
        auto const& junc = varying_candidates[j];
        if (!junc.must_disconnect && junc.probability > 0)
          actual_varying_count++;
      }
      if (actual_varying_count > max_component_size)
        throw std::runtime_error("Large component");
    }

    // 4. Actually call the enumeration
    // Disable region term
    Float tmp_region_term_ratio = region_term_ratio;
    region_term_ratio = 0.0;
    IncrementalCache tmp_cache;
    variable_decompose(in_state, stroke_graph, tmp_cache, final_graph, varying_candidates,
                       max_num_var_list2, max_num_sol_list2, first_sols, false);
    region_term_ratio = tmp_region_term_ratio;
  }

  assert(first_sols.size() == 1);

  // II. Reorder and re-solve with region hard constraints on
  SPDLOG_INFO("Hard constraints on.");
  {
    // Sort based on component sum
    std::map<size_t, Float> component_values;
    varying_candidates = std::move(first_sols.front().candidates_);
    int max_component =
      *(std::max_element(junc_components.begin(), junc_components.end()));
    for (int i = 0; i <= max_component; ++i) {
      for (size_t j = 0; j < varying_candidates.size(); ++j) {
        const auto& junc = varying_candidates[j];
        if (junc.component_idx != i)
          continue;
        component_values[junc.component_idx] = 0;
        break;
      }
      for (size_t j = 0; j < varying_candidates.size(); ++j) {
        const auto& junc = varying_candidates[j];
        if (junc.component_idx != i || !first_sols.front().connectivity_[j])
          continue;
        component_values[junc.component_idx] += (junc.probability - 0.5);
        /*component_values[junc.component_idx] =
          std::max(component_values[junc.component_idx], junc.probability);*/
      }
    }

    // Get the largest gap connected in the region disabled setting
    if (trivial_largest_gap && *trivial_largest_gap < 0) {
      *trivial_largest_gap = 0;
      for (size_t i = 0; i < varying_candidates.size(); ++i) {
        const auto& junc = varying_candidates[i];
        if (first_sols.front().connectivity_[i])
          *trivial_largest_gap = std::max(*trivial_largest_gap, junc.orig_dist);
      }
      if (*trivial_largest_gap > 0)
        serial_largest_non_region_gap = *trivial_largest_gap;
      else
        *trivial_largest_gap = std::numeric_limits<Float>::infinity();
    }

    SPDLOG_INFO("Off solution.");
    print_junctions(varying_candidates);
    component_sort(component_values, varying_candidates, junc_components);

    SPDLOG_INFO("Reordered.");
    print_junctions(varying_candidates);

    std::vector<size_t> max_num_var_list1;
    max_num_var_list1.emplace_back(1);
    for (size_t i = 1; i < junc_components.size(); ++i) {
      if (junc_components[i] != junc_components[i - 1])
        max_num_var_list1.emplace_back(1);
      else
        max_num_var_list1.back()++;
    }

    std::vector<size_t> max_num_var_list2;
    std::vector<int> max_num_sol_list2;
    for (const auto& vars : max_num_var_list1) {
      SPDLOG_INFO("\t\tComponent size: {}", vars);

      std::vector<size_t> max_num_var_list;
      std::vector<int> max_num_sol_list;
      max_num_var_list.resize(vars / max_num_vars, max_num_vars);
      if (vars % max_num_vars)
        max_num_var_list.emplace_back(vars % max_num_vars);
      max_num_sol_list.resize(max_num_var_list.size(), max_num_sols);
      // For individual parts, the enumerations can run separately so we don't need to
      // keep too many intermediate states.
      max_num_sol_list.back() = max_num_states_region;

      max_num_var_list2.insert(max_num_var_list2.end(), max_num_var_list.begin(),
                               max_num_var_list.end());
      max_num_sol_list2.insert(max_num_sol_list2.end(), max_num_sol_list.begin(),
                               max_num_sol_list.end());
    }
    // Skip the last check: -1
    max_num_sol_list2.back() = last_num_states;

    // Printout the actual varying variable counts
    for (size_t i = 0, start_i = 0; i < max_num_var_list1.size();
         start_i += max_num_var_list1[i++]) {
      size_t actual_varying_count = 0;
      for (size_t j = start_i; j < start_i + max_num_var_list1[i]; ++j) {
        auto const& junc = varying_candidates[j];
        if (!junc.must_disconnect && junc.probability > 0)
          actual_varying_count++;
      }
      SPDLOG_INFO("\t\tComponent varying size: {}", actual_varying_count);
    }

    if (0) {
      std::vector<std::pair<size_t, size_t>> adj_faces;
      std::vector<Float> junc_distances;
      std::vector<StrokeGraph::VertexID> junc_vertex_ids;
      std::vector<bool> junction_connected{
        false, true,  true,  true,  true,  false, false, false, false, true,  true,
        true,  false, true,  false, false, false, false, true,  true,  false, true,
        false, true,  false, false, false, true,  true,  true,  true,  true,  false,
        true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,
        true,  true,  true,  true,  true,  true,  false, true,  true,  true,  false,
        false, true,  true,  true,  true,  true,  true,
        true, //
        false, false, false, false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false, false, false, false,
        false, false, false,
      };
      auto varying_stroke_graph =
        modify_graph(stroke_graph, varying_candidates, junction_connected, adj_faces,
                     junc_distances, junc_vertex_ids);
      if (varying_stroke_graph) {
        Index highest_violation = -1;
        std::unordered_map<std::string, Float> pseudo_cache;
        for (size_t i = 0; i < adj_faces.size(); ++i) {
          if (!hard_region_condition_check(*varying_stroke_graph, pseudo_cache,
                                           adj_faces[i].first, junc_distances[i])) {
            highest_violation = i;
            SPDLOG_DEBUG("Failed: {}; junc: {} <= {} vs {}", binary_str, i + 1, f1_size,
                         junc_distances[i]);
            break;
          }
          if (!hard_region_condition_check(*varying_stroke_graph, pseudo_cache,
                                           adj_faces[i].second, junc_distances[i])) {
            highest_violation = i;
            SPDLOG_DEBUG("Failed: {}; junc: {} <= {} vs {}", binary_str, i + 1, f2_size,
                         junc_distances[i]);
            break;
          }
        }
        const auto pass_check = highest_violation == -1;
      }
    }

    // 3.5 Assign must disconnect junctions based on the largest gap of initial result
    if (trivial_largest_gap) {
      for (auto& junc : varying_candidates) {
        if (junc.orig_dist > largest_gap_ratio * *trivial_largest_gap) {
          junc.is_weak = true;
          junc.must_disconnect = true;
        }
      }
    }

    // 4. Actually call the enumeration
    variable_decompose(in_state, stroke_graph, cache, final_graph, varying_candidates,
                       max_num_var_list2, max_num_sol_list2, sols, true);

    if (0) {
      SPDLOG_INFO("obj_function:");
      for (auto& s : sols) {
        s.connectivity_[74] = true;
        std::vector<std::pair<size_t, size_t>> adj_faces;
        std::vector<Float> junc_distances;
        std::vector<StrokeGraph::VertexID> junc_vertex_ids;
        s.graph_ = (modify_graph(stroke_graph, s.candidates_, s.connectivity_, adj_faces,
                                 junc_distances, junc_vertex_ids))
                     ->clone();
        print_junctions(s.candidates_);
        for (const auto& junc : s.candidates_) {
          s.junc_distance_map_[junc.repr] = junc.orig_dist;
        }
        Float region_obj;
        obj_function(s, region_obj, "region.txt");
      }
    }
  }
}

void probability_decompose(const GraphState& in_state, const StrokeGraph& stroke_graph,
                           IncrementalCache& cache, const StrokeGraph& final_graph,
                           std::vector<Junction>& varying_candidates,
                           const size_t max_num_vars, const size_t max_num_sols,
                           std::vector<RegionSolution>& sols, const std::string& viz_dir,
                           Float largest_non_region_gap) {
  // 1. Decompose into high probability and low probability
  // 1.5 Remove single-component candidates that trivially violate the hard constraint
  decompose_candidates(stroke_graph, varying_candidates, min_decomp_prob);
  remove_trivial_violation(stroke_graph, varying_candidates);
  remove_trivial_corner_violation(stroke_graph, varying_candidates);
  print_junctions(stroke_graph, varying_candidates);
  decompose_candidates(stroke_graph, varying_candidates, min_decomp_prob);

  {
    size_t actual_varying_count = 0;
    for (auto const& junc : varying_candidates) {
      if (!junc.must_disconnect && junc.probability > 0)
        actual_varying_count++;
    }
    SPDLOG_INFO("\tVarying: {} / {}", actual_varying_count, varying_candidates.size());
  }

  // 2. Solve the high probability subset with the regular decomposition
  std::vector<Junction> high_varying_candidates;
  for (const auto& junc : varying_candidates) {
    if (junc.component_idx >= 0)
      high_varying_candidates.emplace_back(junc);
  }
  print_junctions(high_varying_candidates);

  if (high_varying_candidates.empty()) {
    sols.emplace_back();
    sols.back().candidates_ = varying_candidates;
    sols.back().connectivity_.resize(varying_candidates.size());
    sols.back().graph_ = stroke_graph.clone();
    sols.back().cache_ = cache;

    return;
  }

  if (!viz_dir.empty()) {
    to_json(high_varying_candidates, viz_dir + "/high_prob_candidates.json");
  }

  // TODO: keep more states? E.g. max_num_states_region
  std::vector<RegionSolution> high_sols;

  // Disable outside region term
  /*outside_term_ratio = 0;
  Float tmp_largest_positive_region = largest_positive_region;
  largest_positive_region = 0.0;*/
  connected_component_decompose(in_state, stroke_graph, cache, final_graph,
                                high_varying_candidates, max_num_vars, max_num_sols,
                                high_sols, 1, &largest_non_region_gap);
  // Enable for the last round
  /*outside_term_ratio = 1;
  largest_positive_region = tmp_largest_positive_region;*/

  SPDLOG_INFO("largest_non_region_gap: {}", largest_non_region_gap);

  // 3. Read out
  std::vector<Junction> disconnected_varying_candidates,
    init_disconnected_varying_candidates;
  for (const auto& sol : high_sols) {
    // 4. Solve again with connected variables removed
    // Determine the still dangling ends
    disconnected_varying_candidates.clear();
    init_disconnected_varying_candidates.clear();
    for (const auto& junc : varying_candidates) {
      if (junc.component_idx < 0) {
        init_disconnected_varying_candidates.emplace_back(junc);
        init_disconnected_varying_candidates.back().repr = "new";
      } else {
        const auto itr = std::find(sol.candidates_.begin(), sol.candidates_.end(), junc);
        assert(itr != sol.candidates_.end());
        bool connected = sol.connectivity_[itr - sol.candidates_.begin()];
        if (!connected)
          init_disconnected_varying_candidates.emplace_back(junc);
      }
    }

    // Bound: A junction can at most get max_pos_weight region award
    decompose_candidates(stroke_graph, init_disconnected_varying_candidates, 0.1);
    std::unordered_map<int, bool> seen_new;
    for (const auto& junc : init_disconnected_varying_candidates) {
      if (junc.component_idx >= 0) {
        seen_new[junc.component_idx] |= !junc.repr.empty();
      }
    }
    for (const auto& junc : init_disconnected_varying_candidates) {
      if (junc.component_idx >= 0 && seen_new[junc.component_idx]) {
        disconnected_varying_candidates.emplace_back(junc);
        disconnected_varying_candidates.back().repr = "";
      }
    }
    print_junctions(disconnected_varying_candidates);

    std::vector<RegionSolution> low_sols;
    if (!disconnected_varying_candidates.empty()) {
      if (!viz_dir.empty()) {
        to_json(disconnected_varying_candidates, viz_dir + "/modified_candidates.json");
      }

      connected_component_decompose(in_state, sol.graph_, cache, final_graph,
                                    disconnected_varying_candidates, max_num_vars,
                                    max_num_sols, low_sols, max_num_states_region);
      assert(!low_sols.empty());

      // 5. Combine
      for (const auto& l_sol : low_sols) {
        sols.emplace_back();
        sols.back().candidates_ = varying_candidates;
        sols.back().connectivity_.resize(varying_candidates.size());
        sols.back().graph_ = l_sol.graph_.clone();
        sols.back().cache_ = l_sol.cache_;
        sols.back().junc_distance_map_ = sol.junc_distance_map_;
        sols.back().junc_distance_map_.insert(l_sol.junc_distance_map_.begin(),
                                              l_sol.junc_distance_map_.end());
        sols.back().region_size_cache_ = sol.region_size_cache_;
        sols.back().region_size_cache_.insert(l_sol.region_size_cache_.begin(),
                                              l_sol.region_size_cache_.end());
        for (size_t i = 0; i < varying_candidates.size(); ++i) {
          const auto& junc = varying_candidates[i];

          const auto l_itr =
            std::find(l_sol.candidates_.begin(), l_sol.candidates_.end(), junc);
          if (l_itr != l_sol.candidates_.end()) {
            sols.back().connectivity_[i] =
              l_sol.connectivity_[l_itr - l_sol.candidates_.begin()];
            sols.back().candidates_[i] =
              l_sol.candidates_[l_itr - l_sol.candidates_.begin()];
          } else {
            const auto h_itr =
              std::find(sol.candidates_.begin(), sol.candidates_.end(), junc);
            if (h_itr != sol.candidates_.end()) {
              sols.back().connectivity_[i] =
                sol.connectivity_[h_itr - sol.candidates_.begin()];
              sols.back().candidates_[i] =
                sol.candidates_[h_itr - sol.candidates_.begin()];
            }
          }
        }
      }
    } else {
      sols.emplace_back();
      sols.back().candidates_ = varying_candidates;
      sols.back().connectivity_.resize(varying_candidates.size());
      sols.back().graph_ = sol.graph_.clone();
      sols.back().cache_ = sol.cache_;
      sols.back().junc_distance_map_ = sol.junc_distance_map_;
      sols.back().region_size_cache_ = sol.region_size_cache_;
      for (size_t i = 0; i < varying_candidates.size(); ++i) {
        const auto& junc = varying_candidates[i];

        const auto h_itr =
          std::find(sol.candidates_.begin(), sol.candidates_.end(), junc);
        if (h_itr != sol.candidates_.end()) {
          sols.back().connectivity_[i] =
            sol.connectivity_[h_itr - sol.candidates_.begin()];
          sols.back().candidates_[i] = sol.candidates_[h_itr - sol.candidates_.begin()];
        }
      }

      print_junctions(sols.back().candidates_);
    }
  }
}

void precomputed_corner_decompose(const GraphState& in_state,
                                  const StrokeGraph& stroke_graph,
                                  IncrementalCache& cache, const StrokeGraph& final_graph,
                                  std::vector<RegionSolution>& sols,
                                  const std::string& viz_dir,
                                  Float largest_non_region_gap) {
  std::vector<Junction> corner_predictions;
  complete_predicted_corner_candidates(stroke_graph, final_graph, FeatureType::OrigStroke,
                                       corner_predictions, 0,
                                       to_include_prev_connections);
  probability_decompose(in_state, stroke_graph, cache, final_graph, corner_predictions,
                        max_num_sub_region_vars, max_num_sub_region_sols, sols, viz_dir,
                        largest_non_region_gap);
}

void corner_decompose(const GraphState& in_state, const StrokeGraph& stroke_graph,
                      IncrementalCache& cache, const StrokeGraph& final_graph,
                      std::vector<Junction>& varying_candidates,
                      std::vector<RegionSolution>& sols, const std::string& viz_dir) {
  // First, solve without corner variables
  std::vector<RegionSolution> regular_sols;
  probability_decompose(in_state, stroke_graph, cache, final_graph, varying_candidates,
                        max_num_sub_region_vars, max_num_sub_region_sols, regular_sols,
                        viz_dir);

  for (const auto& sol : regular_sols) {
    IncrementalCache sol_cache;
    std::vector<RegionSolution> corner_sols;
    GraphState corner_state = in_state;
    corner_state.region_size_cache_ = sol.region_size_cache_;
    precomputed_corner_decompose(corner_state, sol.graph_, sol_cache, final_graph,
                                 corner_sols, viz_dir);
    // Attach variables
    for (auto& c_sol : corner_sols) {
      /*c_sol.candidates_.insert(c_sol.candidates_.begin(), sol.candidates_.begin(),
                               sol.candidates_.end());
      c_sol.connectivity_.insert(c_sol.connectivity_.begin(), sol.connectivity_.begin(),
                                 sol.connectivity_.end());*/
      sols.emplace_back();
      sols.back() = std::move(c_sol);
    }
  }
}

void sort_solutions(
  std::vector<RegionSolution>& final_solutions, const GraphState& in_state,
  const StrokeGraph& final_graph,
  std::set<std::pair<Float, size_t>, std::greater<std::pair<Float, size_t>>>& sort_objs,
  std::set<std::pair<Float, size_t>, std::greater<std::pair<Float, size_t>>>&
    sort_objs_all,
  std::set<std::pair<Float, size_t>, std::greater<std::pair<Float, size_t>>>&
    sort_objs_likely,
  bool to_check_high_valence) {
  // Compute costs of feasible solutions and save the promising ones
  for (size_t j = 0; j < final_solutions.size(); ++j) {
    auto& sol = final_solutions[j];

    // Given the modified graph, check if the assignment is valid at endpoint where
    // multiple
    // junctions are assigned positive
    bool high_valence_valid = true;
    for (size_t i = 0; to_check_high_valence && i < sol.graph_.vertices_.size(); ++i) {
      if (!sol.graph_.vertex(i).is_active())
        continue;
      bool new_connection = false;
      for (const auto& vid : sol.graph_.vertex(i).vertex_ids()) {
        if (sol.junc_distance_map_.count(vid.repr())) {
          new_connection = true;
          break;
        }
      }
      if (new_connection && !is_assignment_valid(sol.candidates_, sol.connectivity_,
                                                 sol.graph_.vertex(i))) {
        high_valence_valid = false;
        break;
      }
    }
    if (!high_valence_valid)
      continue;

    // Can't have purely weak component
    {
      std::vector<Junction> connected_candidates;
      for (size_t i = 0; i < sol.candidates_.size(); ++i) {
        if (sol.connectivity_[i])
          connected_candidates.emplace_back(sol.candidates_[i]);
      }
      if (exists_weak_only_connection(sol.graph_, connected_candidates))
        continue;
    }

    // Update the probability of high-valence vertices before doing any obj computation
    if (include_corners) {
      // update_corner_probabilities(sol.graph_, final_graph, sol.candidates_);
    } else if (high_valence_prob_update) {
      update_high_valence_probabilities(sol.graph_, final_graph, sol.candidates_);
    }

    Float region_obj;
    Float obj = obj_function(sol, region_obj, "tmp.txt");

    sort_objs_all.emplace(obj, j);

    // Hacky filter of the hard constraint violating cases
    if (!violation_check(sol))
      continue;

    // Check if we've chosen all highly likely junctions
    if ((!region_junc_threshold.empty() &&
         !highly_likely_junctions_connected(
           sol, region_obj, in_state,
           (region_junc_threshold.size() >= 2) ? region_junc_threshold[0] : 1,
           (region_junc_threshold.size() >= 2) ? region_junc_threshold[1]
                                               : region_junc_threshold[0]))) {
      continue;
    }

    // If all past decisions are valid
    Float certain_prob =
      (region_junc_threshold.empty()) ? -1 : region_junc_threshold.back();
    bool is_valid =
      is_sol_graph_valid(sol.graph_, sol.junc_distance_map_, sol.candidates_, in_state,
                         sol.region_size_cache_, certain_prob);
    if (!is_valid) {
      sort_objs_likely.emplace(obj, j);
      continue;
    }

    sort_objs.emplace(obj, j);
  }
}

} // namespace sketching
