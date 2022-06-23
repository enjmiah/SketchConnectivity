#include "drawing.h"

#include <sketching/clipping.h>
#include <sketching/fairing.h>
#include <sketching/fitting.h>
#include <sketching/render.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <spdlog/spdlog.h>

namespace py = pybind11;
using namespace pybind11::literals;
using namespace ::sketching;

void init_fairing(py::module& m) {
  m.def(
    "smooth_drawing_box3",
    [](Drawing& d, int iterations) {
      for (auto& stroke : d.mut_strokes()) {
        for (int i = 0; i < iterations; ++i) {
          smooth_stroke_box3(stroke);
        }
      }
    },
    "drawing"_a, "iterations"_a = 1);
  m.def("dehooked_range", &dehooked_range);
  m.def(
    "dehook_strokes",
    [](Drawing& d, Float factor) { dehook_strokes(d.mut_strokes(), factor); },
    "drawing"_a, "factor"_a = 1);

  m.def(
    "remove_strokes_visual",
    [](Drawing& d, Float average_area_threshold_proportion,
       Float stroke_area_threshold_proportion) {
      Drawing out;
      out.strokes_ = remove_strokes_visual(d.strokes(), average_area_threshold_proportion,
                                           stroke_area_threshold_proportion);
      return out;
    },
    "drawing"_a, "average_area_threshold_proportion"_a,
    "stroke_area_threshold_proportion"_a);

  m.def("remove_hooks_visual", &remove_hooks_visual, "stroke"_a,
        "area_threshold_proportion"_a);
  m.def("cut_at_corners", [](const Drawing& d) {
    auto out = std::make_unique<Drawing>();
    cut_at_corners(d.strokes(), out->mut_strokes());
    return out;
  });
}
