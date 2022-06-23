#include "cast.h"
#include "drawing.h"

#include <sketching/diff.h>
#include <sketching/global_solve/incremental_util.h>
#include <sketching/graph_color.h>
#include <sketching/incremental.h>
#include <sketching/io/json.h>
#include <sketching/io/pdf.h>
#include <sketching/io/svg.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;
using namespace sketching;

namespace {

using LabelImageMat =
  Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

Image<const int32_t> label_image_from_matrix(const LabelImageMat& mat) {
  auto im = Image<const int32_t>();
  im.data_ = mat.data();
  im.width_ = mat.cols();
  im.height_ = mat.rows();
  return im;
}

Eigen::VectorXi py_map_color(const StrokeGraph& graph, const int max_colors) {
  const auto nf = graph.faces_.size();
  auto coloring = Eigen::VectorXi(nf, 1);
  map_color(graph, max_colors, {coloring.data(), nf});
  return coloring;
}

std::pair<Eigen::VectorXi, bool> py_map_color_raster(const LabelImageMat& label_image_mat,
                                                     const int max_colors) {

  const auto n_labels = label_image_mat.maxCoeff() + 1;
  auto coloring = Eigen::VectorXi(n_labels, 1);
  const auto label_image = label_image_from_matrix(label_image_mat);
  const auto success =
    map_color_raster(label_image, max_colors, {coloring.data(), (size_t)n_labels});
  return {std::move(coloring), success};
}

bool py_color_by_reference( //
  int max_colors, //
  const StrokeGraph& ref_graph, const py::EigenDRef<const Eigen::VectorXi>& ref_coloring,
  const StrokeGraph& new_graph, Eigen::Ref<Eigen::VectorXi> out_coloring) {

  return color_by_reference(
    max_colors, //
    ref_graph, {ref_coloring.data(), (size_t)ref_coloring.rows()}, //
    new_graph, {out_coloring.data(), (size_t)out_coloring.rows()});
}

bool py_color_by_reference_raster( //
  int max_colors, //
  const LabelImageMat& ref_label,
  const py::EigenDRef<const Eigen::VectorXi>& ref_coloring,
  const LabelImageMat& new_label, Eigen::Ref<Eigen::VectorXi> out_coloring) {

  const auto n_labels = new_label.maxCoeff() + 1;
  auto coloring = Eigen::VectorXi(n_labels, 1);
  const auto ref_label_image = label_image_from_matrix(ref_label);
  const auto new_label_image = label_image_from_matrix(new_label);
  return color_by_reference_raster(
    max_colors, //
    ref_label_image, {ref_coloring.data(), (size_t)ref_coloring.rows()}, //
    new_label_image, {out_coloring.data(), (size_t)out_coloring.rows()});
}

void py_compute_correspondence_raster(const LabelImageMat& ref_label,
                                      const size_t n_ref_faces,
                                      const LabelImageMat& new_label,
                                      const size_t n_new_faces, std::vector<int>& new2ref,
                                      std::vector<int>& ref2new) {
  const auto ref_label_image = label_image_from_matrix(ref_label);
  const auto new_label_image = label_image_from_matrix(new_label);
  compute_correspondence_raster(ref_label_image, n_ref_faces, //
                                new_label_image, n_new_faces, new2ref, ref2new);
}

std::vector<std::unordered_set<size_t>>
py_compute_connectivity_raster(const LabelImageMat& label, const int n_labels) {
  const auto label_img = label_image_from_matrix(label);
  return compute_connectivity(label_img, n_labels);
}

void prediction_use_original_positions(const StrokeGraph& graph,
                                       ClassifierPrediction& pred_) {
  ClassifierPrediction pred = pred_;
  if (pred.key.type == JunctionType::T) {
    Junction junc(std::vector<StrokeTime>{pred.orig_a, pred.orig_b}, JunctionType::T);
    bool ok = reproject_t_junc_orig(graph, junc);
    if (ok) {
      pred.orig_a = junc.points[0];
      pred.orig_b = junc.points[1];
    } else {
      junc = Junction(std::vector<StrokeTime>{pred.orig_b, pred.orig_a}, JunctionType::T);
      ok = reproject_t_junc_orig(graph, junc);
      if (ok) {
        pred.orig_b = junc.points[0];
        pred.orig_a = junc.points[1];
      }
    }
  }
  pred_.p_a = graph.orig_strokes_[pred.orig_a.first].pos_norm(pred.orig_a.second);
  pred_.p_b = graph.orig_strokes_[pred.orig_b.first].pos_norm(pred.orig_b.second);
}

void prediction_use_graph_positions(const StrokeGraph& graph,
                                    ClassifierPrediction& pred) {
  StrokeTime end1(pred.orig_a.first, pred.orig_a.second);
  StrokeTime end2(pred.orig_b.first, pred.orig_b.second);
  auto ok = convert_orig2strokes(graph, end1);
  assert(ok && "couldn't map from orig to strokes");
  ok = convert_orig2strokes(graph, end2);
  assert(ok && "couldn't map from orig to strokes");

  pred.p_a = graph.strokes_[end1.first].pos_norm(end1.second);
  pred.p_b = graph.strokes_[end2.first].pos_norm(end2.second);
}

py::tuple py_changed_faces(const StrokeGraph& graph1, const StrokeGraph& graph2) {
  auto changed1 = std::vector<size_t>();
  auto changed2 = std::vector<size_t>();
  changed_faces(graph1, graph2, changed1, changed2);
  return py::make_tuple(std::move(changed1), std::move(changed2));
}

bool save_highlighted_diff(const PlotParams& params, //
                           const StrokeGraph& graph1, const StrokeGraph& graph2,
                           const py::str& path1, const py::str& path2,
                           const Eigen::Ref<Eigen::VectorXi>& face_colors1,
                           const Eigen::Ref<Eigen::VectorXi>& face_colors2) {
  auto changed1 = std::vector<size_t>();
  auto changed2 = std::vector<size_t>();
  changed_faces(graph1, graph2, changed1, changed2);

  save_pdf(graph1, path1, params, {nullptr, 0}, face_colors1, changed1);
  save_pdf(graph2, path2, params, {nullptr, 0}, face_colors2, changed2);

  return !changed1.empty() || !changed2.empty();
}

} // namespace

void init_io(py::module& m) {
  m.def("map_color", &py_map_color, "graph"_a, "max_colors"_a);
  m.def("map_color_raster", &py_map_color_raster, "label_image"_a, "max_colors"_a);
  m.def("color_by_reference", &py_color_by_reference, //
        "max_colors"_a, "ref_graph"_a, "ref_coloring"_a, "new_graph"_a,
        "out_coloring"_a.noconvert() // Needs contiguous storage for span compatibility.
  );
  m.def("color_by_reference_raster", &py_color_by_reference_raster, "max_colors"_a,
        "ref_label"_a, "ref_coloring"_a, "new_label"_a,
        "out_coloring"_a.noconvert() // Needs contiguous storage for span compatibility.
  );

  m.def("compute_correspondence",
        [](const StrokeGraph& ref_graph, const size_t n_ref_faces,
           const StrokeGraph& new_graph, const size_t n_new_faces) {
          std::vector<int> new2ref, ref2new;
          compute_correspondence(ref_graph, n_ref_faces, new_graph, n_new_faces, new2ref,
                                 ref2new);
          return py::make_tuple(new2ref, ref2new);
        });
  m.def("compute_correspondence_raster",
        [](const LabelImageMat& ref_label, const size_t n_ref_faces,
           const LabelImageMat& new_label, const size_t n_new_faces) {
          std::vector<int> new2ref, ref2new;
          py_compute_correspondence_raster(ref_label, n_ref_faces, new_label, n_new_faces,
                                           new2ref, ref2new);
          return py::make_tuple(new2ref, ref2new);
        });
  m.def("label_area",
        [](const LabelImageMat& label, const size_t fi, const size_t n_faces) {
          const auto label_image = label_image_from_matrix(label);
          return label_area(label_image, fi, n_faces);
        });

  m.def("compute_connectivity",
        [](const StrokeGraph& graph) { return compute_connectivity(graph); });
  m.def("compute_connectivity_raster", &py_compute_connectivity_raster);

  m.def("get_color_palette", []() {
    const auto palette = get_color_palette();
    auto out = py::list();
    for (const auto& color : palette) {
      out.append(py::make_tuple(color.r, color.g, color.b));
    }
    return out;
  });
  m.def("prediction_use_original_positions", &prediction_use_original_positions);
  m.def("prediction_use_graph_positions", &prediction_use_graph_positions);

  py::class_<PlotParams>(m, "PlotParams")
    .def(py::init<>())
    .def_property(
      "envelope_fill", [](const PlotParams& p) { return p.envelope_fill; },
      [](PlotParams& p, int color) { p.envelope_fill = Col3::from_hex_rgb(color); })
    .def_property(
      "centerline_color",
      [](const PlotParams& p) {
        return py::make_tuple(p.centerline_color.r, p.centerline_color.g,
                              p.centerline_color.b);
      },
      [](PlotParams& p, int color) { p.centerline_color = Col3::from_hex_rgb(color); })
    .def_property(
      "disconnection_color",
      [](const PlotParams& p) {
        return py::make_tuple(p.disconnection_color.r, p.disconnection_color.g,
                              p.disconnection_color.b);
      },
      [](PlotParams& p, int color) { p.disconnection_color = Col3::from_hex_rgb(color); })
    .def_property(
      "media_box", [](const PlotParams& p) { return p.media_box; },
      [](PlotParams& p, const BoundingBox& bb) { p.media_box = bb; })
    .def_property(
      "compress", [](const PlotParams& p) { return p.compress; },
      [](PlotParams& p, bool compress) { p.compress = compress; })
    .def_property(
      "viz_centerlines", [](const PlotParams& p) { return p.viz_centerlines; },
      [](PlotParams& p, bool viz_centerlines) { p.viz_centerlines = viz_centerlines; })
    .def_property(
      "viz_envelopes", [](const PlotParams& p) { return p.viz_envelopes; },
      [](PlotParams& p, bool viz_envelopes) { p.viz_envelopes = viz_envelopes; })
    .def_property(
      "viz_faces", [](const PlotParams& p) { return p.viz_faces; },
      [](PlotParams& p, bool state) { p.viz_faces = state; })
    .def_property(
      "opaque_faces", [](const PlotParams& p) { return p.opaque_faces; },
      [](PlotParams& p, bool state) { p.opaque_faces = state; })
    .def_property(
      "viz_dangling", [](const PlotParams& p) { return p.viz_dangling; },
      [](PlotParams& p, bool state) { p.viz_dangling = state; })
    .def_property(
      "viz_ends", [](const PlotParams& p) { return p.viz_ends; },
      [](PlotParams& p, bool state) { p.viz_ends = state; })
    .def_property(
      "connection_width", [](const PlotParams& p) { return p.connection_width; },
      [](PlotParams& p, Float w) { p.connection_width = w; })
    .def_property(
      "show_prediction_index",
      [](const PlotParams& p) { return p.show_prediction_index; },
      [](PlotParams& p, bool state) { p.show_prediction_index = state; });

  m.def("changed_faces", &py_changed_faces);
  m.def("save_highlighted_diff", &save_highlighted_diff);

  m.def(
    "save_pdf",
    [](const StrokeGraph& graph, const std::string& path, const PlotParams& params,
       const std::vector<ClassifierPrediction>& predictions,
       const Eigen::Ref<const Eigen::VectorXi>& face_colors,
       const std::vector<size_t>& highlighted_faces) {
      save_pdf(graph, path, params, predictions,
               {face_colors.data(), (size_t)face_colors.rows()}, highlighted_faces);
    },
    "graph"_a, "path"_a, "params"_a,
    "predictions"_a = std::vector<ClassifierPrediction>(),
    "face_colors"_a = Eigen::VectorXi(), "highlighted_faces"_a = std::vector<size_t>());
  m.def(
    "save_pdf",
    [](const Stroke& stroke, const std::string& path, const PlotParams& params,
       const StrokeSnapInfo* pred_info) {
      if (pred_info) {
        save_pdf(stroke, path, params, pred_info->predictions);
      } else {
        save_pdf(stroke, path, params);
      }
    },
    "stroke"_a, "path"_a, "params"_a, "pred_info"_a = nullptr);
  m.def(
    "save_pdf",
    [](const Drawing& d, const std::string& path, const PlotParams& params) {
      save_pdf(d.strokes(), path, params);
    },
    "d"_a, "path"_a, "params"_a);

  m.def("save_json", &save_json);
  m.def("load_json", [](const std::string& path) {
    auto graph = StrokeGraph();
    load_json(path, graph);
    return graph;
  });

  m.def("save_svg",
        [](const Drawing& d, const std::string& path) { save_svg(d.strokes(), path); });
}
