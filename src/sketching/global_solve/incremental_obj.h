#pragma once

#include "../types.h"

namespace sketching {
struct Junction;
struct StrokeGraph;
struct GraphState;
struct RegionSolution;

Float get_face_size(const StrokeGraph& graph,
                    std::unordered_map<std::string, Float>& region_size_cache, size_t fi,
                    const Float check_junc_dist);
Float get_face_size_const(const StrokeGraph& graph,
                          const std::unordered_map<std::string, Float>& region_size_cache,
                          size_t fi);
Float get_junc_prob(const std::vector<Junction>& juncs, const std::string& junc_str);

void update_high_valence_probabilities(const StrokeGraph& graph,
                                       const StrokeGraph& final_graph,
                                       std::vector<Junction>& candidates);
void update_corner_probabilities(const StrokeGraph& graph, const StrokeGraph& final_graph,
                                 std::vector<Junction>& candidates);

bool is_sol_graph_valid(const StrokeGraph& graph,
                        const std::unordered_map<std::string, Float>& junc_distance_map,
                        const std::vector<Junction>& candidates,
                        const GraphState& in_state,
                        const std::unordered_map<std::string, Float>& region_size_cache,
                        Float certain_prob);

Float obj_function(const GraphState& state, Float& region_obj,
                   const std::string& txt_filename = "");
Float obj_function(const RegionSolution& state, Float& region_obj,
                   const std::string& txt_filename = "");

bool violation_check(const GraphState& state);
bool violation_check(const RegionSolution& state);

void find_junctions_in_face(const StrokeGraph& graph,
                            const std::vector<Junction>& candidates, const size_t fi,
                            std::vector<Junction>& in_face_junctions,
                            bool include_interior_junctions = false);

} // namespace sketching
