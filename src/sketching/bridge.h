#pragma once

#include "stroke_graph.h"
#include "types.h"

namespace sketching {
struct StrokeSnapInfo;

struct BridgeInfo {
  std::vector<Junction> bridges;
  std::vector<std::vector<CoordMat>> envelopes_visual;
  std::vector<CoordMat> bridge_intersections_visual;
  std::vector<CoordMat> other_intersections_visual;
  std::vector<Junction> largest_connections;
};

BridgeInfo find_bridge_locations(const StrokeGraph& graph);

void augment_with_bridges(StrokeGraph& graph, const BridgeInfo&);

/** round is 1-indexed. */
BridgeInfo find_final_bridge_locations(const StrokeGraph& graph,
                                       span<const ClassifierPrediction> candidates,
                                       int round);

std::vector<Junction> augment_with_final_bridges(StrokeGraph& graph, const BridgeInfo&,
                                                 int round);

void augment_with_final_bridges(StrokeGraph& graph, const BridgeInfo&);

void augment_with_overlapping_final_bridges(StrokeGraph& graph, const BridgeInfo&);

void multi_bridge(const StrokeGraph& plane_graph, const std::vector<Junction>& candidates,
                  StrokeGraph::SnappingType snapping_type, FeatureType feature_type,
                  StrokeSnapInfo& predictions, StrokeGraph& stroke_graph,
                  Float accept_ratio, Float lowest_p,
                  Float largest_non_region_gap = std::numeric_limits<Float>::infinity(),
                  Float accept_ratio_factor = -1);

std::vector<Junction> unbridge_interior(const StrokeGraph& stroke_graph,
                                        const std::vector<Junction>& candidates);

} // namespace sketching
