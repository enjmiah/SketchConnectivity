#include "cast.h"
#include "drawing.h"

#include <sketching/eigen_compat.h>
#include <sketching/incremental.h>
#include <sketching/io/vac.h>
#include <sketching/is_face_collapsible.h>
#include <sketching/junction.h>
#include <sketching/stroke_graph_extra.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;
using namespace sketching;

namespace {

/**
 * Output text labels and their positions for debug visualization.
 *
 * The first two columns of the output matrix are the positions in data space, and the
 * second two columns are a normalized offset vector, which should be applied in figure
 * space.
 */
std::pair<Eigen::Matrix<Float, Eigen::Dynamic, 4, Eigen::RowMajor>,
          std::vector<std::string>>
text_labels(const StrokeGraph& graph) {
  auto labels = std::vector<std::string>();
  labels.reserve(graph.hedges_.size());
  auto coords =
    Eigen::Matrix<Float, Eigen::Dynamic, 4, Eigen::RowMajor>(graph.hedges_.size(), 4);
  auto row = Index(0);
  for (size_t i = 0; i < graph.hedges_.size(); ++i) {
    const auto he = graph.hedge(i);
    if (he) {
      const auto& stroke = he.stroke();
      const auto l = stroke.length();
      coords.block(row, 0, 1, 2) = to_eigen(stroke.pos(0.5 * l)).transpose();
      const auto pen_width = stroke.pen_width();
      Vec2 tangent = (stroke.pos(std::min(l, 0.5 * l + 2 * pen_width)) -
                      stroke.pos(std::max(0.0, 0.5 * l - 2 * pen_width)));
      if (tangent.squaredNorm() > 1e-6) {
        tangent.normalize();
      } else {
        tangent =
          (stroke.pos(std::min(l, 0.5 * l + 0.1)) - stroke.pos(0.5 * l)).normalized();
      }
      auto normal = Vec2(tangent.y(), -tangent.x());
      if (he.forward())
        normal = -normal;
      coords.block(row, 2, 1, 2) = to_eigen(normal).transpose();
      row++;

      labels.emplace_back(fmt::format( //
        "#{} t{}\nn{} p{}\nf{} o{}", //
        he.index_, he.twin().index_, he.next().index_, he.prev().index_, //
        he.face().index_, he.origin().index_));
    }
  }
  if (row != coords.rows())
    coords.conservativeResize(row, Eigen::NoChange);
  return {coords, labels};
}

CoordMat face_positions(const StrokeGraph& graph, const size_t face_index) {
  std::vector<Vec2> poly = graph.face_positions(face_index);
  CoordMat out(poly.size(), 2);
  auto row = 0;
  for (size_t i = 0; i < poly.size(); ++i) {
    out.row(row++) = to_eigen(poly[i]);
  }
  return out;
}

Eigen::MatrixXd vertex_positions(const StrokeGraph& graph, const int min_valence,
                                 const int max_valence) {
  assert(min_valence <= max_valence);
  auto positions = Eigen::MatrixXd(graph.vertices_.size(), 3);
  auto row = Index(0);
  for (size_t i = 0; i < graph.vertices_.size(); ++i) {
    const auto v = graph.vertex(i);
    if (v.is_active() && (int)v.valence() >= min_valence &&
        (int)v.valence() <= max_valence) {
      if (max_valence <= 1 && (v.flags() & StrokeGraph::VertexRecord::Overlapping)) {
        continue;
      }

      positions(row, 0) = graph.vertices_[i].p_.x_;
      positions(row, 1) = graph.vertices_[i].p_.y_;
      positions(row, 2) = Float(i);
      row++;
    }
  }
  positions.conservativeResize(row, Eigen::NoChange_t());
  return positions;
}

Eigen::MatrixXd vertex_orig_positions(const StrokeGraph& graph, const int min_valence,
                                      const int max_valence) {
  assert(min_valence <= max_valence);
  auto positions = Eigen::MatrixXd(graph.vertices_.size(), 3);
  auto row = Index(0);
  for (size_t i = 0; i < graph.vertices_.size(); ++i) {
    const auto v = graph.vertex(i);
    if (v.is_active() && (int)v.valence() >= min_valence &&
        (int)v.valence() <= max_valence) {
      if (max_valence <= 1 && (v.flags() & StrokeGraph::VertexRecord::Overlapping)) {
        continue;
      }

      StrokeTime end((int)graph.hedge(graph.vertices_[i].hedge_).stroke_idx(),
                     (graph.hedge(graph.vertices_[i].hedge_).forward()) ? 0 : 1);
      auto ok = convert_strokes2orig(graph, end);
      assert(ok && "couldn't map from orig to strokes");
      auto p = graph.orig_strokes_[end.first].pos_norm(end.second);
      positions(row, 0) = p.x();
      positions(row, 1) = p.y();
      positions(row, 2) = Float(i);
      row++;
    }
  }
  positions.conservativeResize(row, Eigen::NoChange_t());
  return positions;
}

struct StrokeGraphStrokes {
  explicit StrokeGraphStrokes(StrokeGraph& g)
    : graph_(g) {}

  Stroke& operator[](Index i) const {
    const auto n = (Index)size();
    if (i < 0) {
      i += n;
    }
    if (i < 0 || i >= n) {
      throw std::out_of_range(std::to_string(i));
    }
    return graph_.strokes_[i];
  }

  size_t size() const { return graph_.strokes_.size(); }

  StrokeGraph& graph_;
};

std::unique_ptr<Drawing> stroke_graph_original_drawing(const StrokeGraph& graph) {
  return std::make_unique<Drawing>(graph.orig_strokes_);
}

std::unique_ptr<Drawing> stroke_graph_as_drawing(const StrokeGraph& graph) {
  return std::make_unique<Drawing>(graph.strokes_);
}

std::string snap_info_str(const SnapInfo& info) {
  switch (info.status) {
    case SnapInfo::Dangling:
      return "dangling; no snap candidates";
    case SnapInfo::Overlap:
      return "snapped due to overlap";
    case SnapInfo::PredictionPos:
      return fmt::format("accepted {} with probability {:.2f}",
                         info.prediction_type == JunctionType::R ? "end-end"
                                                                 : "end-stroke",
                         info.max_prob);
    case SnapInfo::PredictionDelay:
      return fmt::format("delayed decision; best candidate: {} with probability {:.2f}",
                         info.prediction_type == JunctionType::R ? "end-end"
                                                                 : "end-stroke",
                         info.max_prob);
    case SnapInfo::PredictionNeg:
      return fmt::format("rejected all; best candidate: {} with probability {:.2f}",
                         info.prediction_type == JunctionType::R ? "end-end"
                                                                 : "end-stroke",
                         info.max_prob);
    case SnapInfo::ReservedDelay:
      return "delayed; will be snapped to a later stroke";
    case SnapInfo::ReservedPos:
      return "snapped based on a constraint";
    case SnapInfo::Skip:
      return "skipped; degenerate stroke";
    default:
      throw std::logic_error("unknown status");
  }
}

} // namespace

void init_stroke_graph(py::module& m) {
  py::class_<StrokeGraph> py_stroke_graph(m, "StrokeGraph");
  py::enum_<StrokeGraph::SnappingType>(py_stroke_graph, "SnappingType")
    .value("Connection", StrokeGraph::SnappingType::Connection)
    .value("Deformation", StrokeGraph::SnappingType::Deformation)
    .export_values();
  py_stroke_graph.def(py::init<>())
    .def(py::init([](StrokeGraph::SnappingType type) { return StrokeGraph(type); }),
         "type"_a = StrokeGraph::SnappingType::Connection)
    .def(py::init([](Drawing& d, StrokeGraph::SnappingType type) {
           return StrokeGraph(d.strokes(), type);
         }),
         "d"_a, "type"_a = StrokeGraph::SnappingType::Connection)
    .def_property_readonly(
      "strokes", [](StrokeGraph& g) { return StrokeGraphStrokes(g); },
      py::keep_alive<0, 1>())
    .def_property_readonly("n_faces", [](StrokeGraph& g) { return g.faces_.size(); })
    .def_property_readonly("n_hedges", [](StrokeGraph& g) { return g.hedges_.size(); })
    .def_property_readonly("n_strokes", [](StrokeGraph& g) { return g.strokes_.size(); })
    .def_property_readonly("n_vertices",
                           [](StrokeGraph& g) { return g.vertices_.size(); })
    .def_property_readonly("n_orig_strokes",
                           [](StrokeGraph& g) { return g.orig_strokes_.size(); })
    .def_property_readonly("boundary_face",
                           [](StrokeGraph& g) { return g.boundary_face_; })
    .def_property_readonly("snap_history",
                           [](const StrokeGraph& graph) { return graph.snap_history_; })
    .def("__repr__", &StrokeGraph::repr)
    .def("__deepcopy__", &StrokeGraph::clone)
    .def("__copy__", &StrokeGraph::clone)
    .def("clone", &StrokeGraph::clone)
    .def(
      "orig_stroke", [](StrokeGraph& g, int i) { return &g.orig_strokes_[i]; },
      py::return_value_policy::reference_internal)
    .def("orig2strokes", [](StrokeGraph& g, int i) { return g.orig2strokes_[i]; })
    .def("face", &StrokeGraph::face, py::return_value_policy::reference_internal)
    .def("vertex", &StrokeGraph::vertex, py::return_value_policy::reference_internal)
    .def("vertex_ids",
         [](StrokeGraph& g, int i) {
           std::string vid_str;
           for (auto const& v : g.vertex(i).vertex_ids()) {
             vid_str += v.repr() + ";";
           }
           return vid_str;
         })
    .def("face_positions", &face_positions)
    .def("vertex_positions", &vertex_positions, "min_valence"_a = 0,
         "max_valence"_a = std::numeric_limits<int>::max())
    .def("vertex_orig_positions", &vertex_orig_positions, "min_valence"_a = 0,
         "max_valence"_a = std::numeric_limits<int>::max())
    .def("text_labels", &text_labels)
    .def("original_drawing", &stroke_graph_original_drawing, py::keep_alive<1, 0>())
    .def("as_drawing", &stroke_graph_as_drawing, py::keep_alive<1, 0>())
    .def("save", (void (*)(const StrokeGraph&, const std::string&)) & save_vac)
    .def("build_orig_bvh",
         [](StrokeGraph& g) {
           g.orig_bvh_ = std::make_unique<PolylineBVH>(g.orig_strokes_);
         })
    .def("is_face_collapsible_clipping", &is_face_collapsible_clipping)
    .def("get_forward_chain_original_indices", [](StrokeGraph& g, int he_idx) {
      std::vector<std::pair<size_t, bool>> orig_indices;
      get_forward_chain_original_indices(g, he_idx, orig_indices, true);
      return orig_indices;
    });

  py::class_<StrokeGraphStrokes>(py_stroke_graph, "Strokes")
    .def("__getitem__", &StrokeGraphStrokes::operator[],
         py::return_value_policy::reference_internal)
    .def("__len__", &StrokeGraphStrokes::size);

  py::class_<StrokeGraph::FaceView>(py_stroke_graph, "FaceView")
    .def("n_edges", &StrokeGraph::FaceView::n_edges)
    .def("n_neighbors", &StrokeGraph::FaceView::n_neighbors);

  py::class_<StrokeGraph::VertexView>(py_stroke_graph, "VertexView")
    .def("pos", &StrokeGraph::VertexView::pos)
    .def("is_active", &StrokeGraph::VertexView::is_active)
    .def("is_dangling", &StrokeGraph::VertexView::is_dangling)
    .def(
      "is_corner",
      [](const StrokeGraph::VertexView& v, bool include_t) {
        return v.is_active() && is_corner(*v.graph_, v.index_, include_t);
      },
      py::arg("include_t") = false)
    .def("hedge", [](const StrokeGraph::VertexView& v) { return v.hedge().index_; })
    .def("orig_pos", [](const StrokeGraph::VertexView& v) {
      std::vector<Vec2> positions;
      auto valence = 0;
      const auto he = v.hedge();
      auto it = he;
      do {
        if (!it.continuity_edge().is_valid()) {
          auto endp = StrokeTime((int)it.stroke_idx(), it.forward() ? 0.0 : 1.0);
          const auto ok = convert_strokes2orig(*v.graph_, endp);
          assert(ok && "couldn't map from strokes to orig");
          positions.emplace_back(
            v.graph_->orig_strokes_[endp.first].pos_norm(endp.second));
        }
        it = it.twin().next();
        valence++;
        assert(valence < 1024 && "likely infinite loop found");
      } while (it != he);
      return std::move(positions);
    });

  m.def("dehook_dangling_edges", &dehook_dangling_edges);
  m.def(
    "chained_drawing",
    [](const StrokeGraph& graph) {
      auto drawing = Drawing();
      auto mapping = std::vector<std::vector<std::pair<size_t, bool>>>();
      drawing.strokes_ = chained_drawing(graph, mapping);
      return py::make_tuple(std::move(drawing), std::move(mapping));
    },
    py::return_value_policy::move);
  m.def(
    "get_corner_original_positions",
    [](const StrokeGraph& graph, const size_t vid, bool include_t) {
      std::vector<Vec2> positions;
      get_corner_original_positions(graph, vid, positions, include_t);

      return std::move(positions);
    },
    py::return_value_policy::move, py::arg("graph"), py::arg("vid"),
    py::arg("include_t") = false);

  py::class_<IncrementalCache>(m, "IncrementalCache")
    .def(py::init(py::overload_cast<Float>(&make_incremental_cache)),
         "accept_threshold"_a = -1.0)
    .def("clear_allowed_connections",
         [](IncrementalCache& cache) { cache.allowed_connections_.clear(); })
    .def_property(
      "accept_threshold", [](IncrementalCache& cache) { return cache.accept_threshold_; },
      [](IncrementalCache& cache, Float thresh) { cache.accept_threshold_ = thresh; })
    .def_property(
      "trim_overshoots", [](IncrementalCache& cache) { return cache.trim_overshoots_; },
      [](IncrementalCache& cache, bool flag) { cache.trim_overshoots_ = flag; });

  py::class_<SnapInfo>(m, "SnapInfo") //
    .def("__str__", &snap_info_str);

  py::class_<FeatureVector> feature_vector(m, "FeatureVector");
  feature_vector.def_property_readonly("type",
                                       [](const FeatureVector& v) { return v.type_; });
  feature_vector.def_property_readonly("data", [](const FeatureVector& v) {
    if (v.type_ == JunctionType::R) {
      const size_t n_features = EndEndFeatures::n_features_;
      return span<const Float>{v.ee_fea_.data_, n_features};
    } else if (v.type_ == JunctionType::T) {
      const size_t n_features = EndStrokeFeatures::n_features_;
      return span<const Float>{v.es_fea_.data_, n_features};
    } else if (v.type_ == FeatureVector::Uninitialized) {
      return span<const Float>{nullptr, 0};
    } else {
      throw std::logic_error("unhandled junction type");
    }
  });

  py::enum_<decltype(FeatureVector::type_)>(feature_vector, "Type")
    .value("EndEnd", FeatureVector::EndEnd)
    .value("EndStroke", FeatureVector::EndStroke)
    .value("Uninitialized", FeatureVector::Uninitialized)
    .export_values();

  py::class_<ClassifierPrediction>(m, "ClassifierPrediction")
    .def(py::init<>())
    .def_property(
      "type", [](ClassifierPrediction& pred) { return pred.key.type; },
      [](ClassifierPrediction& pred, JunctionType::Type t) { pred.key.type = t; })
    .def_property(
      "corner_type", [](ClassifierPrediction& pred) { return pred.key.corner_type; },
      [](ClassifierPrediction& pred, JunctionType::Type t) { pred.key.corner_type = t; })
    .def_property(
      "cand1", [](ClassifierPrediction& pred) { return pred.key.cand1; },
      [](ClassifierPrediction& pred, int cand) { pred.key.cand1 = cand; })
    .def_property(
      "cand2", [](ClassifierPrediction& pred) { return pred.key.cand2; },
      [](ClassifierPrediction& pred, int cand) { pred.key.cand2 = cand; })
    .def_property(
      "prob", [](ClassifierPrediction& pred) { return pred.prob; },
      [](ClassifierPrediction& pred, Float v) { pred.prob = v; })
    .def_property(
      "alt_prob", [](ClassifierPrediction& pred) { return pred.alt_prob; },
      [](ClassifierPrediction& pred, Float v) { pred.alt_prob = v; })
    .def_property(
      "p_a", [](ClassifierPrediction& pred) { return pred.p_a; },
      [](ClassifierPrediction& pred, Vec2 p) { pred.p_a = p; })
    .def_property(
      "p_b", [](ClassifierPrediction& pred) { return pred.p_b; },
      [](ClassifierPrediction& pred, Vec2 p) { pred.p_b = p; })
    .def_property(
      "orig_a", [](ClassifierPrediction& pred) { return pred.orig_a; },
      [](ClassifierPrediction& pred, StrokeTime p) { pred.orig_a = p; })
    .def_property(
      "orig_b", [](ClassifierPrediction& pred) { return pred.orig_b; },
      [](ClassifierPrediction& pred, StrokeTime p) { pred.orig_b = p; })
    .def_property(
      "fea", [](ClassifierPrediction& pred) { return pred.fea; },
      [](ClassifierPrediction& pred, const FeatureVector& f) { pred.fea = f; })
    .def_property(
      "connected", [](ClassifierPrediction& pred) { return pred.connected; },
      [](ClassifierPrediction& pred, bool c) { pred.connected = c; })
    .def_property(
      "junction_repr", [](ClassifierPrediction& pred) { return pred.junction_repr; },
      [](ClassifierPrediction& pred, std::string r) {
        pred.junction_repr = std::move(r);
      })
    .def("__repr__", [](ClassifierPrediction& pred) {
      auto ss = std::stringstream();
      ss << "ClassifierPrediction(";
      ss << junction_type_to_char(pred.key.type) << ", ";
      ss << "cand1=(" << pred.key.cand1 << "), ";
      ss << "cand2=(" << pred.key.cand2 << "), ";
      ss << "p_a=(" << pred.p_a.x_ << ", " << pred.p_a.y_ << "), ";
      ss << "p_b=(" << pred.p_b.x_ << ", " << pred.p_b.y_ << "), ";
      ss << "orig_a=(" << pred.orig_a.first << ", " << pred.orig_a.second << "), ";
      ss << "orig_b=(" << pred.orig_b.first << ", " << pred.orig_b.second << "), ";
      ss << "prob=" << pred.prob << ", ";
      ss << "alt_prob=" << pred.alt_prob << ", ";
      ss << "connected=" << (pred.connected ? "True" : "False");
      ss << ")";
      return ss.str();
    });

  py::class_<StrokeSnapInfo>(m, "StrokeSnapInfo")
    .def_property_readonly("head", [](StrokeSnapInfo& info) { return info.head; })
    .def_property_readonly("tail", [](StrokeSnapInfo& info) { return info.tail; })
    .def_property_readonly("predictions",
                           [](StrokeSnapInfo& info) { return info.predictions; });

  m.def(
    "set_future_constraints",
    [](IncrementalCache& cache, const Drawing& d, bool train_time,
       const size_t num_candidates) {
      set_future_constraints(&cache, d.strokes(), num_candidates, train_time);
    },
    py::arg("cache"), py::arg("d"), py::arg("train_time"), py::arg("num_candidates") = 3);
  m.def("add_stroke_incremental", &add_stroke_incremental, py::arg("graph"),
        py::arg("cache"), py::arg("new_stroke"), py::arg("to_dissolve") = true);
  m.def("add_stroke_incremental_topological", &add_stroke_incremental_topological,
        py::arg("graph"), py::arg("cache"), py::arg("new_stroke"),
        py::arg("to_dissolve") = false);
  m.def("finalize_incremental", &finalize_incremental, py::arg("graph"), py::arg("cache"),
        py::arg("to_dissolve") = true);

  // Move these somewhere else if we decide to expose more classifier-related functions.
  m.def("get_end_end_feature_descriptions", &get_end_end_feature_descriptions);
  m.def("get_end_stroke_feature_descriptions", &get_end_stroke_feature_descriptions);
  m.def("human_readable_transform", [](const ClassifierPrediction& in_pred) {
    auto pred = in_pred;
    if (pred.fea.type_ == JunctionType::R) {
      human_readable_end_end_features(
        {pred.fea.ee_fea_.data_, EndEndFeatures::n_features_});
    } else if (pred.fea.type_ == JunctionType::T) {
      human_readable_end_stroke_features(
        {pred.fea.es_fea_.data_, EndStrokeFeatures::n_features_});
    } else if (pred.fea.type_ == FeatureVector::Uninitialized) {
      // Do nothing.
    } else {
      throw std::logic_error("unhandled junction type");
    }
    return pred;
  });
}
