#include "cast.h"
#include "drawing.h"

#include <sketching/fitting.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <spdlog/spdlog.h>

namespace py = pybind11;
using namespace pybind11::literals;
using namespace ::sketching;

namespace {

#ifdef HAS_GUROBI
py::tuple fit_stroke_to_cluster_py(const py::list& strokes, const Float accuracy,
                                   const bool cut_strokes) {
  std::vector<const Stroke*> input;
  for (const auto obj : strokes) {
    input.push_back(obj.cast<Stroke*>());
  }
  std::vector<StrokeMapping> mappings;
  auto fit = std::make_unique<Stroke>(
    fit_stroke_to_cluster(input, accuracy, cut_strokes, mappings));
  return py::make_tuple(std::move(fit), mappings);
}

py::tuple fitted_drawing(const Drawing& d, const Clustering& clustering,
                         const Float accuracy, const bool cut_strokes,
                         const py::set& ignore_py) {
  auto out_d = std::make_unique<Drawing>();
  auto out_m = py::dict();
  std::unordered_set<int> ignored;
  for (const auto obj : ignore_py) {
    // Maybe we should add the unfitted strokes to the drawing?
    ignored.insert(obj.cast<int>());
  }
  for (const auto& [cluster, stroke_indices] : clustering.cluster2stroke()) {
    if (ignored.find(cluster) != ignored.end()) {
      continue;
    }
    if (stroke_indices.size() == 1) {
      out_d->mut_strokes().emplace_back(d.strokes()[stroke_indices.front()].clone());
    } else {
      std::vector<const Stroke*> strokes;
      for (const auto stroke_index : stroke_indices) {
        strokes.push_back(&d.strokes()[stroke_index]);
      }
      std::vector<StrokeMapping> mappings;
      try {
        out_d->mut_strokes().emplace_back(
          fit_stroke_to_cluster(strokes, accuracy, cut_strokes, mappings));
      } catch (const std::runtime_error&) {
        auto ss = std::stringstream();
        for (const auto si : stroke_indices) {
          ss << si << ", ";
        }
        SPDLOG_WARN("fitting failed for cluster {} with strokes {}", cluster, ss.str());
        throw;
      }
      out_m[py::cast(cluster)] = std::move(mappings);
    }
  }
  return py::make_tuple(std::move(out_d), out_m);
}
#endif

} // namespace

void init_fitting(py::module& m) {
  py::class_<Bezier>(m, "Bezier") //
    .def_property_readonly("pts",
                           [](Bezier& b) {
                             Eigen::MatrixXd out(4, 2);
                             out(0, 0) = b.pts[0].x_;
                             out(0, 1) = b.pts[0].y_;
                             out(1, 0) = b.pts[1].x_;
                             out(1, 1) = b.pts[1].y_;
                             out(2, 0) = b.pts[2].x_;
                             out(2, 1) = b.pts[2].y_;
                             out(3, 0) = b.pts[3].x_;
                             out(3, 1) = b.pts[3].y_;
                             return out;
                           })
    .def("pos", &Bezier::pos)
    .def("tangent", &Bezier::tangent)
    .def("normalized_head_tangent", &Bezier::normalized_head_tangent)
    .def("normalized_tail_tangent", &Bezier::normalized_tail_tangent);

  py::class_<BezierSpline>(m, "BezierSpline")
    .def("__len__", [](const BezierSpline& spline) { return spline.n_segments_; })
    .def("__getitem__",
         [](const BezierSpline& spline, Index i) {
           if (i < 0) {
             i += spline.n_segments_;
           }
           if (i < 0 || i >= spline.n_segments_) {
             throw std::out_of_range("");
           }
           return spline.segments_[i];
         })
    .def("pos", &BezierSpline::pos)
    .def("tangent", &BezierSpline::tangent)
    .def("length", &BezierSpline::length);

  m.def("fit_bezier_spline_with_corners", &fit_bezier_spline_with_corners, //
        "stroke"_a, "tolerance"_a);

  py::class_<ClusteringBuilder>(m, "ClusteringBuilder")
    .def(py::init<int>(), "n_strokes"_a)
    .def("__bool__", &ClusteringBuilder::operator bool)
    .def("add", &ClusteringBuilder::add, "stroke_index"_a, "cluster_index"_a)
    .def("finalize", &ClusteringBuilder::finalize,
         "The ClusteringBuilder instance becomes invalid after this.");

  py::class_<Clustering>(m, "Clustering")
    .def(py::init<>())
    .def(py::pickle(
      [](const Clustering& c) { // __getstate__
        const auto n = c.n_strokes();
        py::list state;
        for (auto i = decltype(n){0}; i < n; ++i) {
          state.append(c.get_cluster_index(i));
        }
        return state;
      },
      [](py::list& state) { // __setstate__
        const auto n = (int)state.size();
        auto cb = ClusteringBuilder(n);
        for (int si = 0; si < n; ++si) {
          cb.add(si, state[si].cast<int>());
        }
        return cb.finalize();
      }))
    .def("__bool__", &Clustering::operator bool)
    .def("__repr__", &Clustering::repr)
    .def("array", &Clustering::array, py::keep_alive<1, 0>())
    .def("clusters", &Clustering::clusters)
    .def("get_cluster_index",
         (int (Clustering::*)(int) const) & Clustering::get_cluster_index,
         "stroke_index"_a)
    .def("get_stroke_indices", &Clustering::get_stroke_indices, "cluster_index"_a);

#ifdef HAS_GUROBI
  m.def("fit_stroke_to_cluster", &fit_stroke_to_cluster_py, "strokes"_a, "accuracy"_a,
        "cut_strokes"_a);
  m.def("fitted_drawing", &fitted_drawing, "d"_a, "clustering"_a, "accuracy"_a = 1.0,
        "cut_strokes"_a = true, "ignore"_a = py::set());
#endif
}
