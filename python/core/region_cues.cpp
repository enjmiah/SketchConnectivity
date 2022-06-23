#include "drawing.h"

#include "cast.h"

#include <sketching/bridge.h>
#include <sketching/global_solve/incremental_obj.h>
#include <sketching/global_solve/incremental_param.h>
#include <sketching/global_solve/incremental_region_util.h>
#include <sketching/global_solve/incremental_solve.h>
#include <sketching/global_solve/incremental_util.h>
#include <sketching/junction.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace sketching;

void init_region_cues(py::module& m) {
  py::enum_<FeatureType>(m, "FeatureType")
    .value("Graph", FeatureType::Graph)
    .value("OrigStroke", FeatureType::OrigStroke)
    .export_values();

  m.def(
    "add_stroke_incremental",
    [](StrokeGraph& graph, IncrementalCache* const cache, const Stroke& new_stroke,
       FeatureType feature_type, bool to_dissolve) {
      prediction_feature_type = feature_type;
      return add_stroke_incremental(graph, cache, new_stroke, to_dissolve);
    },
    py::arg("graph"), py::arg("cache"), py::arg("new_stroke"), py::arg("feature_type"),
    py::arg("to_dissolve") = true);
  m.def(
    "finalize_incremental",
    [](StrokeGraph& graph, IncrementalCache* const cache, FeatureType feature_type,
       bool to_dissolve) {
      prediction_feature_type = feature_type;
      return finalize_incremental(graph, cache, to_dissolve);
    },
    py::arg("graph"), py::arg("cache"), py::arg("feature_type"),
    py::arg("to_dissolve") = true);

  m.def("junction_distance_sort",
        [](const StrokeGraph& stroke_graph, std::vector<Junction> candidates) {
          junction_distance_sort(stroke_graph, candidates);
          return candidates;
        });
  m.def(
    "incremental_solve",
    [](const Drawing& d, StrokeGraph::SnappingType snapping_type, size_t max_num_strokes,
       size_t max_per_stroke_states, const std::string& viz_dir) {
      std::vector<StrokeGraph> stroke_graphs;
      std::vector<StrokeSnapInfo> predictions;
      std::unordered_map<size_t, int> prev_state;
      int sol_state =
        incremental_solve(d.strokes(), snapping_type, stroke_graphs, predictions,
                          prev_state, max_num_strokes, max_per_stroke_states, viz_dir);
      return py::make_tuple(sol_state, std::move(stroke_graphs), predictions, prev_state);
    },
    py::return_value_policy::move, py::arg("d"),
    py::arg("snapping_type") = StrokeGraph::SnappingType::Deformation,
    py::arg("max_num_strokes") = std::numeric_limits<size_t>::max(),
    py::arg("max_per_stroke_states") = std::numeric_limits<size_t>::max(),
    py::arg("viz_dir") = "");
  m.def(
    "nonincremental_nonregion_solve",
    [](const Drawing& d, StrokeGraph::SnappingType snapping_type,
       FeatureType feature_type, size_t max_num_strokes, size_t max_per_stroke_states,
       const std::string& viz_dir, const bool to_include_t, const bool bridge) {
      std::vector<StrokeGraph> stroke_graphs;
      std::vector<StrokeSnapInfo> predictions;
      std::unordered_map<size_t, int> prev_state;
      include_ts = to_include_t;
      int sol_state = nonincremental_nonregion_solve(
        d.strokes(), snapping_type, feature_type, stroke_graphs, predictions, prev_state,
        max_num_strokes, max_per_stroke_states, viz_dir, bridge);
      return py::make_tuple(sol_state, std::move(stroke_graphs), predictions, prev_state);
    },
    py::return_value_policy::move, py::arg("d"),
    py::arg("snapping_type") = StrokeGraph::SnappingType::Connection,
    py::arg("feature_type") = FeatureType::OrigStroke,
    py::arg("max_num_strokes") = std::numeric_limits<size_t>::max(),
    py::arg("max_per_stroke_states") = std::numeric_limits<size_t>::max(),
    py::arg("viz_dir") = "", py::arg("to_include_t") = true, py::arg("bridge") = false);
  m.def(
    "corner_solve",
    [](const StrokeGraph& plane_graph, const StrokeGraph& in_graph,
       StrokeGraph::SnappingType snapping_type, FeatureType feature_type,
       size_t max_per_stroke_states, const bool to_include_t,
       const bool include_prev_connections, const Float largest_non_region_gap) {
      StrokeGraph stroke_graph = in_graph.clone();
      StrokeSnapInfo prediction, all_prediction;
      include_ts = to_include_t;
      corner_solve(plane_graph, snapping_type, feature_type, prediction, all_prediction,
                   stroke_graph, include_prev_connections, largest_non_region_gap);
      return py::make_tuple(std::move(stroke_graph), prediction, all_prediction);
    },
    py::return_value_policy::move, py::arg("plane_graph"), py::arg("in_graph"),
    py::arg("snapping_type") = StrokeGraph::SnappingType::Connection,
    py::arg("feature_type") = FeatureType::OrigStroke,
    py::arg("max_per_stroke_states") = std::numeric_limits<size_t>::max(),
    py::arg("to_include_t") = true, py::arg("include_prev_connections") = false,
    py::arg("largest_non_region_gap") = std::numeric_limits<Float>::infinity());
  m.def(
    "corner_solve",
    [](const StrokeGraph& plane_graph, const StrokeGraph& in_graph,
       StrokeGraph::SnappingType snapping_type, FeatureType feature_type,
       size_t max_per_stroke_states, const bool to_include_t,
       const Float largest_non_region_gap) {
      StrokeGraph stroke_graph = in_graph.clone();
      StrokeSnapInfo prediction, all_prediction;
      include_ts = to_include_t;
      corner_solve(plane_graph, snapping_type, feature_type, prediction, all_prediction,
                   stroke_graph, false, largest_non_region_gap);
      return py::make_tuple(std::move(stroke_graph), prediction, all_prediction);
    },
    py::return_value_policy::move, py::arg("plane_graph"), py::arg("in_graph"),
    py::arg("snapping_type") = StrokeGraph::SnappingType::Connection,
    py::arg("feature_type") = FeatureType::OrigStroke,
    py::arg("max_per_stroke_states") = std::numeric_limits<size_t>::max(),
    py::arg("to_include_t") = true,
    py::arg("largest_non_region_gap") = std::numeric_limits<Float>::infinity());
  m.def(
    "multi_bridge",
    [](const StrokeGraph& plane_graph, const std::vector<Junction>& candidates,
       const StrokeGraph& in_graph, Float accept_ratio, Float lowest_p,
       StrokeGraph::SnappingType snapping_type, FeatureType feature_type,
       const bool to_include_t, const Float largest_non_region_gap,
       const Float accept_ratio_factor) {
      StrokeGraph stroke_graph = in_graph.clone();
      StrokeSnapInfo prediction, all_prediction;
      include_ts = to_include_t;

      std::vector<Junction> varying_candidates = candidates;
      for (auto& junc : varying_candidates) {
        if (junc.orig_dist < 0) {
          junc.orig_dist = junction_distance_init(plane_graph, junc);
        }
      }

      multi_bridge(plane_graph, varying_candidates, StrokeGraph::SnappingType::Connection,
                   FeatureType::OrigStroke, prediction, stroke_graph, accept_ratio,
                   lowest_p, largest_non_region_gap, accept_ratio_factor);
      return py::make_tuple(std::move(stroke_graph), prediction);
    },
    py::return_value_policy::move, py::arg("plane_graph"), py::arg("candidates"),
    py::arg("in_graph"), py::arg("accept_ratio"), py::arg("lowest_p"),
    py::arg("snapping_type") = StrokeGraph::SnappingType::Connection,
    py::arg("feature_type") = FeatureType::OrigStroke, py::arg("to_include_t") = true,
    py::arg("largest_non_region_gap") = std::numeric_limits<Float>::infinity(),
    py::arg("accept_ratio_factor") = -1);

  m.def("build_plane_graph", [](const Drawing& d) {
    StrokeGraph stroke_graph;
    build_plane_graph(d.strokes(), stroke_graph);
    return stroke_graph;
  });
  m.def(
    "vanilla_candidates",
    [](const StrokeGraph& graph, bool train_time, int num_cand) {
      std::vector<Junction> dangling_predictions;
      vanilla_candidates(graph, dangling_predictions, train_time, num_cand);
      return std::move(dangling_predictions);
    },
    py::return_value_policy::move, py::arg("graph"), py::arg("train_time"),
    py::arg("num_cand") = -1);
  m.def(
    "predicted_corner_candidates",
    [](const StrokeGraph& graph, const StrokeGraph& plane_graph,
       const FeatureType feature_type, bool to_dedup, bool include_prev_connections) {
      std::vector<Junction> corner_predictions;
      predicted_corner_candidates(graph, plane_graph, feature_type, corner_predictions,
                                  to_dedup, include_prev_connections);
      return std::move(corner_predictions);
    },
    py::return_value_policy::move, py::arg("graph"), py::arg("plane_graph"),
    py::arg("feature_type") = FeatureType::OrigStroke, py::arg("to_dedup") = true,
    py::arg("include_prev_connections") = false);

  m.def(
    "update_disconnected_junction_predictions",
    [](const StrokeGraph& graph, const std::vector<Junction>& candidates,
       FeatureType feature_type) {
      std::vector<Junction> varying_candidates;
      varying_candidates.reserve(candidates.size());
      for (const auto& junc : candidates) {
        varying_candidates.emplace_back(junc);
      }
      update_disconnected_junction_predictions(graph, varying_candidates, feature_type,
                                               false);
      return std::move(varying_candidates);
    },
    py::return_value_policy::move);
  m.def(
    "update_high_valence_junction_predictions",
    [](const StrokeGraph& graph, const std::vector<Junction>& candidates,
       FeatureType feature_type) {
      std::vector<Junction> varying_candidates;
      varying_candidates.reserve(candidates.size());
      for (const auto& junc : candidates) {
        varying_candidates.emplace_back(junc);
      }
      update_high_valence_junction_predictions(graph, varying_candidates, feature_type);
      return std::move(varying_candidates);
    },
    py::return_value_policy::move);

  m.def(
    "visualize_obj_function",
    [](const StrokeGraph& stroke_graph, std::vector<Junction>& candidates,
       const std::string& txt_filename) {
      RegionSolution state;
      state.candidates_ = candidates;
      state.connectivity_.resize(state.candidates_.size(), false);
      for (size_t i = 0; i < candidates.size(); ++i) {
        state.candidates_[i].orig_dist =
          junction_distance_init(stroke_graph, state.candidates_[i]);
        state.connectivity_[i] = !state.candidates_[i].repr.empty();
        if (state.connectivity_[i]) {
          state.junc_distance_map_[state.candidates_[i].repr] =
            state.candidates_[i].orig_dist;
        }
      }
      state.graph_ = stroke_graph.clone();

      Float region_obj;
      obj_function(state, region_obj, txt_filename);
    },
    py::arg("stroke_graph"), py::arg("candidates"), py::arg("txt_filename") = "");

  m.def("viz_connected_junctions", [](const StrokeGraph& stroke_graph) {
    std::vector<std::pair<std::string, Vec2>> junctions;
    viz_connected_junctions(stroke_graph, junctions);
    return junctions;
  });

  m.def("debug_connectivity_state",
        [](const StrokeGraph& stroke_graph, std::vector<Junction>& connected_junctions) {
          std::vector<Junction> connectivities;
          bool succeeded =
            debug_connectivity_state(stroke_graph, connected_junctions, connectivities);
          return py::make_tuple(succeeded, connectivities);
          ;
        });
  m.def("face_id", [](StrokeGraph::FaceView f) { return face_id(f); });
  m.def("face_circle", [](StrokeGraph::FaceView f) {
    Eigen::Vector2d center;
    double radius = face_maximum_inscribing_circle_radius(*f.graph_, f.index_, center);
    return py::make_tuple(radius, center);
  });
  m.def("connect_graph", [](const StrokeGraph& stroke_graph,
                            std::vector<Junction>& candidates) { //
    std::vector<std::pair<size_t, size_t>> adj_faces;
    std::vector<Float> junc_distances;
    std::vector<bool> junction_connected;
    junction_connected.resize(candidates.size(), true);
    std::vector<StrokeGraph::VertexID> junc_vertex_ids;
    auto varying_stroke_graph = modify_graph(stroke_graph, candidates, junction_connected,
                                             adj_faces, junc_distances, junc_vertex_ids);
    const auto succeeded = (bool)varying_stroke_graph;
    return py::make_tuple(succeeded, std::move(varying_stroke_graph));
  });
  m.def(
    "connect_graph_candidates_simple",
    [](const StrokeGraph& stroke_graph, const std::vector<Junction>& candidates,
       bool include_interior_junctions) { //
      std::vector<Junction> varying_candidates = candidates;

      for (auto& junc : varying_candidates) {
        if (junc.probability < 0)
          junc.probability = 1;
        if (junc.orig_dist < 0) {
          junc.orig_dist = junction_distance_init(stroke_graph, junc);
        }
      }

      StrokeGraph varying_stroke_graph = stroke_graph.clone();
      connect_graph(varying_stroke_graph, varying_candidates);

      // Find face junction association
      std::map<size_t, std::vector<Junction>> f2juncs;
      for (size_t fi = 0; fi < varying_stroke_graph.faces_.size(); ++fi) {
        std::vector<Junction> in_face_junctions;
        find_junctions_in_face(varying_stroke_graph, varying_candidates, fi,
                               in_face_junctions, include_interior_junctions);
        if (!in_face_junctions.empty())
          f2juncs[fi] = std::move(in_face_junctions);
      }

      return py::make_tuple(std::move(varying_stroke_graph), std::move(f2juncs),
                            std::move(varying_candidates));
    },
    py::return_value_policy::move, py::arg("stroke_graph"), py::arg("candidates"),
    py::arg("include_interior_junctions") = false);
  m.def(
    "connect_graph_candidates",
    [](const StrokeGraph& stroke_graph, const std::vector<Junction>& candidates,
       bool include_interior_junctions) { //
      std::vector<std::pair<size_t, size_t>> adj_faces;
      std::vector<Float> junc_distances;
      std::vector<bool> junction_connected;
      std::vector<Junction> varying_candidates = candidates;

      for (auto& junc : varying_candidates) {
        if (junc.probability < 0)
          junc.probability = 1;
        if (junc.orig_dist < 0) {
          junc.orig_dist = junction_distance_init(stroke_graph, junc);
        }
      }

      junction_connected.resize(candidates.size(), true);
      std::vector<StrokeGraph::VertexID> junc_vertex_ids;
      auto varying_stroke_graph =
        modify_graph(stroke_graph, varying_candidates, junction_connected, adj_faces,
                     junc_distances, junc_vertex_ids);
      const auto succeeded = (bool)varying_stroke_graph;

      // Find face junction association
      std::map<size_t, std::vector<Junction>> f2juncs;
      if (succeeded) {
        for (size_t fi = 0; fi < varying_stroke_graph->faces_.size(); ++fi) {
          std::vector<Junction> in_face_junctions;
          find_junctions_in_face(*varying_stroke_graph, varying_candidates, fi,
                                 in_face_junctions, include_interior_junctions);
          if (!in_face_junctions.empty())
            f2juncs[fi] = std::move(in_face_junctions);
        }
      }

      return py::make_tuple(succeeded, std::move(*varying_stroke_graph),
                            std::move(f2juncs), std::move(varying_candidates));
    },
    py::return_value_policy::move, py::arg("stroke_graph"), py::arg("candidates"),
    py::arg("include_interior_junctions") = false);

  m.def(
    "find_junctions_in_face",
    [](const StrokeGraph& graph, const std::vector<Junction>& candidates, const size_t fi,
       bool include_interior_junctions) { //
      std::vector<Junction> in_face_junctions;
      find_junctions_in_face(graph, candidates, fi, in_face_junctions,
                             include_interior_junctions);
      return std::move(in_face_junctions);
    },
    py::return_value_policy::move, py::arg("graph"), py::arg("candidates"), py::arg("fi"),
    py::arg("include_interior_junctions") = false);

  // Region measurements
  // Accurate region radius
  m.def("face_maximum_inscribing_circle_radius_clipping", [](StrokeGraph::FaceView f) {
    Eigen::Vector2d center;
    double radius =
      face_maximum_inscribing_circle_radius_clipping(*f.graph_, f.index_, center);
    return py::make_tuple(radius, center);
  });
  // Region perimeter
  m.def(
    "face_perimeter",
    [](StrokeGraph::FaceView f, bool include_interior_strokes) {
      double perimeter = face_perimeter(*f.graph_, f.index_, include_interior_strokes);
      return perimeter;
    },
    py::arg("f"), py::arg("include_interior_strokes") = false);
  // Face area
  m.def("face_area", [](StrokeGraph::FaceView f) {
    double area = face_area(*f.graph_, f.index_);
    return area;
  });
  // Min stroke width surrounding the given face
  m.def("face_stroke_width_min", [](StrokeGraph::FaceView f) {
    double min_sw = face_stroke_width_min(*f.graph_, f.index_);
    return min_sw;
  });
}
