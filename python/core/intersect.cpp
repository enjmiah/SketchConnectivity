#include <sketching/bvh.h>
#include <sketching/intersect.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace sketching;

void init_intersect(py::module& m) {
  py::class_<BoundingBox>(m, "BoundingBox")
    .def(py::init<>())
    .def_property("ymin", &BoundingBox::yMin,
                  [](BoundingBox& bb, Float v) { bb.yMin_ = v; })
    .def_property("ymax", &BoundingBox::yMax,
                  [](BoundingBox& bb, Float v) { bb.yMax_ = v; })
    .def_property("xmin", &BoundingBox::xMin,
                  [](BoundingBox& bb, Float v) { bb.xMin_ = v; })
    .def_property("xmax", &BoundingBox::xMax,
                  [](BoundingBox& bb, Float v) { bb.xMax_ = v; })
    .def("width", &BoundingBox::width)
    .def("height", &BoundingBox::height)
    .def("united", &BoundingBox::united)
    .def("largest_axis_length", &BoundingBox::largest_axis_length)
    .def("area", &BoundingBox::area);

  py::class_<PolylineBVHLeaf>(m, "PolylineBVHLeaf") //
    .def_property_readonly("bb", [](const PolylineBVHLeaf& l) { return l.bb; });
  py::class_<PolylineBVH>(m, "PolylineBVH")
    .def_property_readonly("bb", [](const PolylineBVH& b) { return b.bb; })
    .def_property_readonly("nodes", [](const PolylineBVH& bvh) { return bvh.nodes; })
    .def("__len__", [](const PolylineBVH& bvh) { return bvh.nodes.size(); });

  py::class_<EnvelopeBVHLeaf>(m, "EnvelopeBVHLeaf") //
    .def(py::init<const Stroke&>())
    .def_property_readonly("bb", [](const EnvelopeBVHLeaf& l) { return l.bb; });
  py::class_<EnvelopeBVH>(m, "EnvelopeBVH")
    .def_property_readonly("bb", [](const EnvelopeBVH& b) { return b.bb; })
    .def_property_readonly("nodes", [](const EnvelopeBVH& bvh) { return bvh.nodes; })
    .def("__len__", [](const EnvelopeBVH& bvh) { return bvh.nodes.size(); });

  py::class_<IntersectionEvent>(m, "IntersectionEvent")
    .def("__repr__", &IntersectionEvent::repr)
    .def_property_readonly("s_start", [](IntersectionEvent& e) { return e.s_start; })
    .def_property_readonly("t_start", [](IntersectionEvent& e) { return e.t_start; })
    .def_property_readonly("s_mid", [](IntersectionEvent& e) { return e.s_mid; })
    .def_property_readonly("t_mid", [](IntersectionEvent& e) { return e.t_mid; })
    .def_property_readonly("s_end", [](IntersectionEvent& e) { return e.s_end; })
    .def_property_readonly("t_end", [](IntersectionEvent& e) { return e.t_end; });

  m.def("intersect_self", //
        [](const EnvelopeBVHLeaf& node) {
          node.geometry->ensure_arclengths();
          auto arclens = std::vector<IntersectionEvent>();
          intersect_self(node, arclens);
          return arclens;
        });
  m.def("intersect_different",
        [](const EnvelopeBVHLeaf& node1, const EnvelopeBVHLeaf& node2) {
          node1.geometry->ensure_arclengths();
          node2.geometry->ensure_arclengths();
          auto arclens = std::vector<IntersectionEvent>();
          intersect_different(node1, node2, arclens);
          return arclens;
        });
}
