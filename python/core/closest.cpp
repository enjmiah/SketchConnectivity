#include <sketching/closest.h>
#include <sketching/junction.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;
using namespace sketching;

namespace {
constexpr auto infinity = std::numeric_limits<Float>::infinity();
}

void init_closest(py::module& m) {
  m.def(
    "closest_point_to_own_stroke",
    [](const Stroke& polyline, bool head) {
      auto proj = Vec2::Empty();
      Float s;
      const auto dist = closest_point_to_own_stroke(polyline, head, proj, s);
      return (dist < infinity ? py::cast(s) : py::none());
    },
    "stroke"_a, "head"_a);
  m.def("pick", py::overload_cast<const PolylineBVH&, Float, Float, Float>(&pick), //
        "polylines"_a, "x"_a, "y"_a, "tolerance"_a);
  m.def("pick_junctions",
        py::overload_cast<const std::vector<Junction>&, const PolylineBVH&, Float, Float,
                          Float, std::size_t>(&pick_junctions),
        "junctions"_a, "strokes"_a, "x"_a, "y"_a, "tolerance"_a,
        "max_results"_a = std::numeric_limits<size_t>::max());
}
