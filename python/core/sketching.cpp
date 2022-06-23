#include "cast.h"
#include "drawing.h"

#include <sketching/bvh.h>
#include <sketching/detail/util.h>
#include <sketching/eigen_compat.h>
#include <sketching/io.h>
#include <sketching/resample.h>
#include <sketching/stroke_view.h>

#include <pybind11/eigen.h>
#include <pybind11/stl.h>

using namespace sketching;

namespace {

py::list stroke_split_py(const Stroke& s, const std::vector<Float>& split_values) {
  std::vector<Stroke> splits;
  s.split(split_values, splits);
  py::list out;
  for (auto& split : splits) {
    out.append(new Stroke(std::move(split)));
  }
  return out;
}

void copy_eigen(const py::EigenDRef<const Eigen::VectorXd>& src, span<Float> dst) {
  const auto n = src.size();
  for (Eigen::Index i = 0; i < n; ++i) {
    dst[i] = src(i);
  }
}

std::unique_ptr<Stroke>
stroke_from_arrays_xyw(const py::EigenDRef<const Eigen::MatrixXd>& x,
                       const py::EigenDRef<const Eigen::MatrixXd>& y,
                       const py::EigenDRef<const Eigen::MatrixXd>& width) {
  auto stroke = std::make_unique<Stroke>(x.size(), false);
  copy_eigen(x, stroke->x());
  copy_eigen(y, stroke->y());
  copy_eigen(width, stroke->width());
  return stroke;
}

std::unique_ptr<Stroke>
stroke_from_arrays_xywt(const py::EigenDRef<const Eigen::MatrixXd>& x,
                        const py::EigenDRef<const Eigen::MatrixXd>& y,
                        const py::EigenDRef<const Eigen::MatrixXd>& width,
                        const py::EigenDRef<const Eigen::MatrixXd>& time) {
  auto stroke = std::make_unique<Stroke>(x.size(), true);
  copy_eigen(x, stroke->x());
  copy_eigen(y, stroke->y());
  copy_eigen(width, stroke->width());
  copy_eigen(time, stroke->time());
  return stroke;
}

std::unique_ptr<Stroke>
stroke_from_matrix(const py::EigenDRef<const Eigen::MatrixXd>& array) {
  bool has_time;
  if (array.cols() == 3) {
    has_time = false;
  } else if (array.cols() == 4) {
    has_time = true;
  } else {
    throw std::invalid_argument("polyline array must have either 3 or 4 columns");
  }
  if (has_time) {
    return stroke_from_arrays_xywt(array.col(0), array.col(1), array.col(2),
                                   array.col(3));
  } else {
    return stroke_from_arrays_xyw(array.col(0), array.col(1), array.col(2));
  }
}

CoordMat endpoints(const Drawing& d) {
  const auto n_strokes = d.size();
  CoordMat out(2 * n_strokes, 2);
  for (auto i = decltype(n_strokes)(0); i < n_strokes; ++i) {
    const auto& stroke = d.strokes()[i];
    assert(stroke.size() > 0);
    out.row(2 * i) = to_eigen(stroke.xy(0));
    out.row(2 * i + 1) = to_eigen(stroke.xy(stroke.size() - 1));
  }
  return out;
}

void remove_duplicate_vertices_py(Drawing& d) {
  for (auto& s : d.mut_strokes()) {
    remove_duplicate_vertices(s);
  }
}

std::unique_ptr<Drawing> duplicate_vertices_removed_py(const Drawing& d) {
  auto out = std::make_unique<Drawing>();
  for (const auto& s : d.strokes()) {
    out->mut_strokes().emplace_back(duplicate_vertices_removed(s));
  }
  return out;
}

Float avg_stroke_width(const Drawing& d) {
  auto sum = 0.0;
  auto denom = Index(0);
  for (const auto& s : d.strokes()) {
    sum += std::accumulate(s.width().begin(), s.width().end(), 0.0);
    denom += s.size();
  }
  return sum / denom;
}

Stroke fractional_slice_checked(const Stroke& s, const Float start, const Float stop) {
  if (start < 0.0) {
    throw std::out_of_range(std::to_string(start));
  } else if (start >= stop) {
    throw std::runtime_error("cannot return an empty slice");
  }
  return s.fractional_slice(start, stop);
}

} // namespace

void init_sketching(py::module& m) {
  py::class_<Stroke>(m, "Stroke")
    .def(py::init<Index, bool>(), "npoints"_a, "has_time"_a)
    .def(py::init(&stroke_from_arrays_xyw))
    .def(py::init(&stroke_from_arrays_xywt))
    .def_static("from_matrix", &stroke_from_matrix, "array"_a)
    .def("__len__", &Stroke::size)
    .def("__repr__", &Stroke::repr)
    .def("pos", &Stroke::pos, "Return the position at the given arc length value.")
    .def("pos_norm", &Stroke::pos_norm)
    .def(
      "xy",
      [](const Stroke& s, Index i) {
        if (i < 0)
          i += s.size();
        if (i < 0 || i >= s.size())
          throw std::out_of_range("");
        return s.xy(i);
      },
      "Return the position at the given index.")
    .def("length", &Stroke::length)
    .def("pen_width", &Stroke::pen_width)
    .def("compute_arclengths", &Stroke::compute_arclengths)
    .def("ensure_arclengths", &Stroke::ensure_arclengths)
    .def("invalidate_arclengths", &Stroke::invalidate_arclengths)
    .def("fractional_index", &Stroke::fractional_index)
    .def("fractional_slice", &fractional_slice_checked)
    .def("avg_sampling", &Stroke::avg_sampling)
    .def("reverse", &Stroke::reverse)
    .def("split", &stroke_split_py)
    .def("trim", py::overload_cast<Index, Index>(&Stroke::trim))
    .def("clone", &Stroke::clone)
    .def("__copy__", &Stroke::clone)
    .def("__deepcopy__", [](Stroke& s, py::object& /*memo*/) { return s.clone(); })
    .def_property_readonly("x", py::overload_cast<>(&Stroke::x))
    .def_property_readonly("y", py::overload_cast<>(&Stroke::y))
    .def_property_readonly("width", py::overload_cast<>(&Stroke::width))
    .def_property_readonly("time", py::overload_cast<>(&Stroke::time))
    .def_property_readonly("arclength", py::overload_cast<>(&Stroke::arclength));

  py::class_<ConstStrokeView>(m, "ConstStrokeView")
    .def(py::init<Stroke&>(), py::keep_alive<1, 2>())
    .def(py::init<>([](Stroke& s, Index start, Index stop) {
           if (start < 0)
             start += s.size();
           if (start >= s.size() || start < 0)
             throw std::out_of_range("");
           if (stop < 0)
             stop += s.size();
           if (stop > s.size() || stop < 0)
             throw std::out_of_range("");
           if (start >= stop)
             throw std::out_of_range("start >= stop");
           return ConstStrokeView(s, start, stop);
         }),
         py::keep_alive<1, 2>())
    .def("__len__", &ConstStrokeView::size)
    .def_property_readonly("x", py::overload_cast<>(&ConstStrokeView::x, py::const_))
    .def_property_readonly("y", py::overload_cast<>(&ConstStrokeView::y, py::const_))
    .def_property_readonly("width",
                           py::overload_cast<>(&ConstStrokeView::width, py::const_))
    .def("slice", &ConstStrokeView::slice)
    // So we can copy a stroke view and a stroke the same way.
    .def("clone", &ConstStrokeView::slice)
    .def("length", &ConstStrokeView::length)
    .def_property_readonly("stroke", &ConstStrokeView::stroke)
    .def_property(
      "start", [](ConstStrokeView& v) { return v.start_; },
      [](ConstStrokeView& v, Index start) { v.start_ = start; })
    .def_property(
      "end", [](ConstStrokeView& v) { return v.end_; },
      [](ConstStrokeView& v, Index end) { v.end_ = end; });
  py::implicitly_convertible<Stroke, ConstStrokeView>();

  py::class_<Drawing>(m, "Drawing")
    .def(py::init<>())
    .def_property_readonly("path",
                           [](const Drawing& d) -> std::optional<py::str> {
                             if (d.path_.length())
                               return d.path_;
                             return std::nullopt;
                           })
    .def("__getitem__", &Drawing::at, py::return_value_policy::reference_internal)
    .def("__len__", &Drawing::size)
    .def("add", &Drawing::add,
         "Add a stroke to the drawing. This copies the given stroke.")
    .def(
      "bvh", [](const Drawing& d) { return std::make_unique<PolylineBVH>(d.strokes()); },
      py::keep_alive<0, 1>())
    .def(
      "envelope_bvh",
      [](const Drawing& d) { return std::make_unique<EnvelopeBVH>(d.strokes()); },
      py::keep_alive<0, 1>())
    .def("visual_bounds", [](const Drawing& d) { return visual_bounds(d.strokes()); })
    .def("avg_sample_dist", &Drawing::avg_sample_dist, Drawing::avg_sample_dist_doc)
    .def("avg_stroke_width", &avg_stroke_width)
    .def("remove_duplicate_vertices", &remove_duplicate_vertices_py)
    .def("duplicate_vertices_removed", &duplicate_vertices_removed_py)
    .def("endpoints", &endpoints)
    .def(
      "save",
      [](const Drawing& d, const std::string& path) { save_vac(d.strokes(), path); },
      "path"_a);

  m.def(
    "load",
    [](const std::string& path) {
      auto d = std::make_unique<Drawing>();
      load_vac(path, d->mut_strokes());
      d->path_ = path;
      return d;
    },
    "path"_a);
}
