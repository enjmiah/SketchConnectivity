#pragma once

#include "../stroke_graph.h"
#include "../types.h"

#include <vector>

namespace sketching {
struct Junction;
struct Stroke;
struct StrokeSnapInfo;

enum SolveType : std::uint32_t { //
  CCSolve = 1,
  RegionSolve = 2,
};

/**
 * Find the optimal solution (defined based on the local prediction) by incrementally
 * exploring the search space.
 * @param strokes Input strokes.
 * @param snapping_type Snapping method type.
 * @param stroke_graphs Path to the solution stroke graph.
 * @param predictions Junction candidates at the solution state with visualization info.
 * @param prev_state The state tree saved as indices to previous states.
 * @param max_num_strokes Optional limit to number of strokes used in the solve.
 * @param max_per_stroke_states Optional limit to number of states explored per stroke
 * added.
 * @param viz_dir Optional directory for intermediate visualizations.
 * @param use_baseline Whether to use the baseline algorithm.
 * @return The optimal solution state index. Use it with prev_state to recover the search
 * path.
 */
int incremental_solve(span<const Stroke> strokes, StrokeGraph::SnappingType snapping_type,
                      std::vector<StrokeGraph>& stroke_graphs,
                      std::vector<StrokeSnapInfo>& predictions,
                      std::unordered_map<size_t, int>& prev_state,
                      size_t max_num_strokes = std::numeric_limits<size_t>::max(),
                      size_t max_per_stroke_states = std::numeric_limits<size_t>::max(),
                      const std::string& viz_dir = "",
                      const SolveType solve_type = RegionSolve);

int nonincremental_nonregion_solve(
  span<const Stroke> strokes, StrokeGraph::SnappingType snapping_type,
  FeatureType feature_type, std::vector<StrokeGraph>& stroke_graphs,
  std::vector<StrokeSnapInfo>& predictions, std::unordered_map<size_t, int>& prev_state,
  size_t max_num_strokes = std::numeric_limits<size_t>::max(),
  size_t max_per_stroke_states = std::numeric_limits<size_t>::max(),
  const std::string& viz_dir = "", bool to_bridge = false);

void corner_solve(const StrokeGraph& plane_graph, StrokeGraph::SnappingType snapping_type,
                  FeatureType feature_type, StrokeSnapInfo& predictions,
                  StrokeSnapInfo& all_predictions, StrokeGraph& stroke_graph,
                  bool to_include_prev_connections = false,
                  Float largest_non_region_gap = -1);

} // namespace sketching
