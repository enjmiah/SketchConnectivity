#include <sketching/clipping.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace cl = ClipperLib;
using namespace sketching;

PYBIND11_MAKE_OPAQUE(cl::Paths)

void init_clipping(py::module& m) {
  py::class_<cl::Paths>(m, "ClipPaths")
    .def("__len__", &cl::Paths::size)
    .def("__get__", [](const cl::Paths& p, int i) {
      if (i < 0)
        i += (int)p.size();
      if (i < 0 || i >= (int)p.size())
        throw std::out_of_range("");
      return from_clip_path(p[i]);
    });

  m.def("to_clip_paths",
        [](const std::vector<CoordMat>& polygons) { return to_clip_paths(polygons); });
  m.def("from_clip_paths", &from_clip_paths);
  m.def("boolean_union", &boolean_union);
  m.def("boolean_difference", &boolean_difference);
  m.def("clip_area_scaled", &clip_area_scaled);
}
