#pragma once

#include "../types.h"

#include <vector>

namespace sketching {
struct Junction;
struct GraphState;
struct StrokeGraph;
struct IncrementalCache;
struct RegionSolution;

void remove_trivial_violation(const StrokeGraph& stroke_graph,
                              std::vector<Junction>& candidates);
void remove_trivial_corner_violation(const StrokeGraph& stroke_graph,
                                     std::vector<Junction>& candidates);
void decompose_candidates(const StrokeGraph& stroke_graph, span<Junction> candidates,
                          const Float min_prob = 0.0);
void connected_component_decompose(
  const GraphState& in_state, const StrokeGraph& stroke_graph, IncrementalCache& cache,
  const StrokeGraph& final_graph, std::vector<Junction>& varying_candidates,
  const size_t max_num_vars, const size_t max_num_sols, std::vector<RegionSolution>& sols,
  int last_num_states = -1, Float* trivial_largest_gap = nullptr);

void probability_decompose(const GraphState& in_state, const StrokeGraph& stroke_graph,
                           IncrementalCache& cache, const StrokeGraph& final_graph,
                           std::vector<Junction>& varying_candidates,
                           const size_t max_num_vars, const size_t max_num_sols,
                           std::vector<RegionSolution>& sols,
                           const std::string& viz_dir = "",
                           Float largest_non_region_gap = -1);
void precomputed_corner_decompose(const GraphState& in_state,
                                  const StrokeGraph& stroke_graph,
                                  IncrementalCache& cache, const StrokeGraph& final_graph,
                                  std::vector<RegionSolution>& sols,
                                  const std::string& viz_dir = "",
                                  Float largest_non_region_gap = -1);
void corner_decompose(const GraphState& in_state, const StrokeGraph& stroke_graph,
                      IncrementalCache& cache, const StrokeGraph& final_graph,
                      std::vector<Junction>& varying_candidates,
                      std::vector<RegionSolution>& sols, const std::string& viz_dir = "");

void sort_solutions(
  std::vector<RegionSolution>& final_solutions, const GraphState& in_state,
  const StrokeGraph& final_graph,
  std::set<std::pair<Float, size_t>, std::greater<std::pair<Float, size_t>>>& sort_objs,
  std::set<std::pair<Float, size_t>, std::greater<std::pair<Float, size_t>>>&
    sort_objs_all,
  std::set<std::pair<Float, size_t>, std::greater<std::pair<Float, size_t>>>&
    sort_objs_likely,
  bool to_check_high_valence = true);
} // namespace sketching
