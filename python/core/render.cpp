#include "drawing.h"

#include <sketching/bridge.h>
#include <sketching/fitting.h>
#include <sketching/junction.h>
#include <sketching/mapping.h>
#include <sketching/render.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;
using namespace sketching;

namespace {

std::string array2string(const py::EigenDRef<const Vec>& arr) {
  std::stringstream ss;
  const auto n1 = arr.size() - 1;
  for (auto i = 0; i < n1; ++i) {
    ss << arr[i] << " ";
  }
  ss << arr[arr.size() - 1];
  return ss.str();
}

} // namespace

void init_render(py::module& m) {
  py::enum_<PolygonizeOptions::Flags>(m, "PolygonizeOptions", py::arithmetic())
    .value("None", PolygonizeOptions::None)
    .value("StartCapRound", PolygonizeOptions::StartCapRound)
    .value("EndCapRound", PolygonizeOptions::EndCapRound)
    .value("JoinsRound", PolygonizeOptions::JoinsRound)
    .value("JoinsBevel", PolygonizeOptions::JoinsBevel)
    .value("RoundCaps", PolygonizeOptions::RoundCaps)
    .export_values();

  m.def("set_render_decimation", &set_render_decimation);
  m.def("get_render_decimation", &get_render_decimation);
  m.def("set_render_n_cap_vertices", &set_render_n_cap_vertices);
  m.def("get_render_n_cap_vertices", &get_render_n_cap_vertices);
  m.def("outline_to_polygons", [](const ConstStrokeView& s) {
    std::vector<CoordMat> vertices;
    outline_to_polygons(s, vertices);
    return vertices;
  });
  m.def("outline_to_polygons", [](const Drawing& d) {
    std::vector<CoordMat> vertices;
    for (const auto& s : d.strokes()) {
      outline_to_polygons(s, vertices);
    }
    return vertices;
  });
  m.def("outline_to_triangulation", [](const ConstStrokeView& s) {
    std::vector<Float> coords;
    std::vector<unsigned> indices;
    outline_to_triangulation(s, coords, indices);
    assert(coords.size() % 2 == 0);
    assert(indices.size() % 3 == 0);
    CoordMat xy(coords.size() / 2, 2);
    {
      const auto n = coords.size() / 2;
      for (auto i = decltype(n){0}; i < n; ++i) {
        xy.row(i) = Eigen::Vector2d(coords[2 * i], coords[2 * i + 1]);
      }
    }
    Eigen::Matrix<unsigned, Eigen::Dynamic, 3> triangles(indices.size() / 3, 3);
    {
      const auto n = indices.size() / 3;
      for (auto i = decltype(n){0}; i < n; ++i) {
        triangles(i, 0) = indices[3 * i];
        triangles(i, 1) = indices[3 * i + 1];
        triangles(i, 2) = indices[3 * i + 2];
      }
    }
    return py::make_tuple(xy, triangles);
  });

  m.def(
    "rasterize",
    [](Drawing& d, int width, int height, BoundingBox* bb) {
      if (bb) {
        return rasterize(d.strokes(), width, height, *bb);
      }
      return rasterize(d.strokes(), width, height, BoundingBox());
    },
    "d"_a, "width"_a, "height"_a, "bb"_a = nullptr);
  m.def(
    "rasterize",
    [](Stroke& s, int width, int height, BoundingBox* bb) {
      if (bb) {
        return rasterize({&s, 1}, width, height, *bb);
      }
      return rasterize({&s, 1}, width, height, BoundingBox());
    },
    "s"_a, "width"_a, "height"_a, "bb"_a = nullptr);
  m.def(
    "rasterize_vertices",
    [](Drawing& d, int width, int height, Float radius_multiplier) {
      return rasterize_vertices(d.strokes(), width, height, radius_multiplier);
    },
    "d"_a, "width"_a, "height"_a, "radius_increase_px"_a = 0.0);
  m.def(
    "rasterize_regions",
    [](const StrokeGraph& graph, int width, int height,
       const Eigen::Ref<const Eigen::VectorXi>& face_colors) {
      return rasterize_regions(graph, width, height,
                               {face_colors.data(), (size_t)face_colors.rows()});
    },
    "graph"_a, "width"_a, "height"_a, "face_colors"_a);
  m.def(
    "rasterize_region_indices",
    [](const StrokeGraph& graph, int width, int height,
       const Eigen::Ref<const Eigen::VectorXi>& face_colors) {
      return rasterize_region_indices(graph, width, height,
                                      {face_colors.data(), (size_t)face_colors.rows()});
    },
    "graph"_a, "width"_a, "height"_a, "face_colors"_a);
  m.def("rasterize_scale", [](Drawing& d, int width, int height) {
    auto bb = BoundingBox();
    const auto scale = rasterize_scale(d.strokes(), width, height, bb);
    return py::make_tuple(scale, bb);
  });

  py::class_<BridgeInfo>(m, "BridgeInfo")
    .def("largest_connections",
         [](const BridgeInfo& info) { return info.largest_connections; })
    .def("envelopes_visual", [](const BridgeInfo& info) { return info.envelopes_visual; })
    .def("bridge_intersections_visual",
         [](const BridgeInfo& info) { return info.bridge_intersections_visual; })
    .def("other_intersections_visual",
         [](const BridgeInfo& info) { return info.other_intersections_visual; })
    .def_property(
      "bridges", [](const BridgeInfo& info) { return info.bridges; },
      [](BridgeInfo& info, const std::vector<Junction>& bridges) {
        info.bridges = bridges;
      });
  m.def("find_bridge_locations", &find_bridge_locations);
  m.def(
    "find_final_bridge_locations",
    [](const StrokeGraph& graph, const std::vector<ClassifierPrediction>& candidates,
       int round) { return find_final_bridge_locations(graph, candidates, round); },
    "graph"_a, "candidates"_a, "round"_a);
  m.def("augment_with_bridges", &augment_with_bridges);
  m.def(
    "augment_with_final_bridges",
    [](StrokeGraph& graph, const BridgeInfo& info, int round) {
      return augment_with_final_bridges(graph, info, round);
    }, //
    "graph"_a, "info"_a, "round"_a);
  m.def("augment_with_overlapping_final_bridges",
        &augment_with_overlapping_final_bridges);
  m.def(
    "augment_with_final_bridges_simple",
    [](StrokeGraph& graph, const BridgeInfo& info) {
      return augment_with_final_bridges(graph, info);
    }, //
    "graph"_a, "info"_a);
  m.def("unbridge_interior", &unbridge_interior);

  m.def("array2string", &array2string);
}
