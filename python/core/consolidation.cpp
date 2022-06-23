#include "cast.h"
#include "drawing.h"

#include <sketching/consolidation/consolidate.h>
#include <sketching/consolidation/snap_endpoints.h>
#include <sketching/stroke_graph.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace sketching;

void init_consolidation(py::module& m) {
#ifdef HAS_GUROBI
  m.def(
    "fit_width_cluster",
    [](const Drawing& input, bool cut_strokes, bool use_default_sampling) {
      std::vector<const Stroke*> clustered_strokes_ptr;
      for (auto const& s : input.strokes())
        clustered_strokes_ptr.emplace_back(&s);
      Float accuracy = 1;

      Stroke fit;
      fit_width_cluster(clustered_strokes_ptr, accuracy, cut_strokes, fit,
                        use_default_sampling);
      return std::move(fit);
    },
    py::return_value_policy::move, py::arg("clustered_strokes"),
    py::arg("cut_strokes") = true, py::arg("use_default_sampling") = true);
#endif

  m.def("consolidate_with_chaining", [](const Drawing& input) {
    auto output = std::make_unique<Drawing>();
    consolidate_with_chaining(input.strokes(), output->mut_strokes());
    return output;
  });
  m.def("consolidate_with_chaining_improved", [](const Drawing& input) {
    auto output = std::make_unique<Drawing>();
    consolidate_with_chaining_improved(input.strokes(), output->mut_strokes());
    return output;
  });
}
