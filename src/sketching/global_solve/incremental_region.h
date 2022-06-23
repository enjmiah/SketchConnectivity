#pragma once

#include "../types.h"
#include "incremental_util.h"

#include <queue>
#include <vector>

namespace sketching {
struct Junction;
struct Stroke;
struct StrokeGraph;

enum CandidateProposalType : std::uint32_t { //
  Incremental = 0,
  Final = 1,
};

struct RegionState {
  size_t cur_variable_;
  std::vector<bool> variable_state_;
  std::unordered_map<std::string, Float> region_size_cache_;
};

bool highly_likely_junctions_connected(const RegionSolution& sol, Float sol_obj,
                                       const GraphState& in_state,
                                       Float highly_likely_prob, Float certain_prob);

void enumerate_subproblem(std::queue<RegionState>& connection_states,
                          const GraphState& in_state, const StrokeGraph& stroke_graph,
                          IncrementalCache& cache, const StrokeGraph& final_graph,
                          const std::vector<Junction>& varying_candidates,
                          const int max_num_sols, const size_t max_pos,
                          std::vector<RegionSolution>& sols,
                          bool check_region_hard = true);

std::unique_ptr<StrokeGraph>
try_connect_graph(const StrokeGraph& varying_stroke_graph,
                  const std::unordered_map<std::string, Float>& in_region_size_cache,
                  std::unordered_map<std::string, Float>& junc_distance_map,
                  std::vector<Junction>& candidates, const std::set<size_t>& binding);

void region_solve(span<const Stroke> strokes, const StrokeGraph& final_graph,
                  const std::vector<Junction>& final_predictions,
                  const GraphState& in_state, const StrokeGraph& viz_before_graph,
                  const std::string& viz_dir, size_t state_count, size_t cur_stroke_step,
                  std::vector<std::vector<GraphState>>& state_levels,
                  const CandidateProposalType candidate_proposal_type = Incremental);
} // namespace sketching
