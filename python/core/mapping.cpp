#include <sketching/mapping.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;
using namespace sketching;

void init_mapping(py::module& m) {
  py::class_<StrokeMapping>(m, "StrokeMapping")
    .def(py::pickle(
      [](const StrokeMapping& sm) { // __getstate__
        py::dict state;
        state["domain_arclens"] = sm.domain_arclens_;
        state["range_arclens"] = sm.range_arclens_;
        return state;
      },
      [](py::dict& state) { // __setstate__
        return StrokeMapping{state["domain_arclens"].cast<std::vector<Float>>(),
                             state["range_arclens"].cast<std::vector<Float>>()};
      }))
    .def_property_readonly(
      "domain_arclens",
      [](StrokeMapping& sm) {
        return Eigen::Map<const Vec>(sm.domain_arclens_.data(),
                                     sm.domain_arclens_.size());
      },
      py::keep_alive<1, 0>())
    .def_property_readonly(
      "range_arclens",
      [](StrokeMapping& sm) {
        return Eigen::Map<const Vec>(sm.range_arclens_.data(), sm.range_arclens_.size());
      },
      py::keep_alive<1, 0>())
    .def("map_clamp", &StrokeMapping::map_clamp)
    .def("is_linear", &StrokeMapping::is_linear)
    .def("is_valid", &StrokeMapping::is_valid)
    .def("is_range_empty", &StrokeMapping::is_range_empty)
    .def("inv_map_clamp_from_start", &StrokeMapping::inv_map_clamp_from_start)
    .def("inv_map_clamp_from_end", &StrokeMapping::inv_map_clamp_from_end)
    .def("inv_map_shortest_interval", &StrokeMapping::inv_map_shortest_interval);
}
