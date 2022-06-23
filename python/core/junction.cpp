#include "cast.h"
#include "drawing.h"

#include <sketching/closest.h>
#include <sketching/eigen_compat.h>
#include <sketching/endpoint.h>
#include <sketching/fitting.h>
#include <sketching/force_assert.h>
#include <sketching/intersect.h>
#include <sketching/junction.h>
#include <sketching/junction_features.h>
#include <sketching/junction_type.h>
#include <sketching/stroke_graph_extra.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iomanip>
#include <unordered_map>

using Eigen::Dynamic;
using MatRowMaj = Eigen::Matrix<sketching::Float, Dynamic, Dynamic, Eigen::RowMajor>;
using Matchups = Eigen::Matrix<std::uint8_t, Dynamic, Dynamic>;
namespace py = pybind11;
using namespace pybind11::literals;
using namespace ::sketching;

namespace {

template <typename T>
bool contains(const std::unordered_set<T>& s, const T& t) {
  return s.find(t) != s.end();
}

py::str junction_repr(const Junction& junc) {
  std::stringstream ss;
  ss << "Junction([" << std::setprecision(13);
  for (const auto& p : junc.points) {
    ss << '(' << p.first << ", " << p.second << ')';
    if (&p != &junc.points.back()) {
      ss << ", ";
    }
  }

  ss << "], '" << junction_type_to_char(junc.type) << '\'';
  if (!junc.repr.empty())
    ss << ", id = '" << junc.repr << '\'';
  ss << ", c = '" << junction_type_to_char(junc.corner_type) << '\'';
  ss << ", p = " << junc.probability;
  if (junc.is_weak) {
    ss << ", weak";
  }
  if (junc.alt_probability >= 0) {
    ss << ", alt p = " << junc.alt_probability;
  }
  ss << ')';
  /*if (junc.type == JunctionType::R) {
    ss << " ";
    for (size_t i = 0; i < junc.fea.ee_fea_.n_features_; ++i) {
      if (i > 0)
        ss << ", ";
      ss << junc.fea.ee_fea_.data_[i];
    }
  } else {
    ss << " [";
    for (size_t i = 0; i < junc.fea.es_fea_.n_features_; ++i) {
      if (i > 0)
        ss << ", ";
      ss << junc.fea.es_fea_.data_[i];
    }
    ss << "]";
  }*/
  return ss.str();
}

CoordMat positions(const Junction& self, const Drawing& d) {
  const auto n = self.size();
  CoordMat out(n, 2);
  for (size_t i = 0; i < n; ++i) {
    const auto& stroke = d.strokes()[self.stroke(i)];
    out.row(i) = to_eigen(stroke.pos(self.arclength(i) * stroke.length()));
  }
  return out;
}

py::list junction_star(const Junction& junc, const Drawing& d) {
  py::list out;
  const auto& edges = d.strokes();
  const auto c = centroid(junc, edges);
  for (const auto& p : junc.points) {
    Eigen::Matrix2d line;
    line.row(0) = to_eigen(c);
    line.row(1) = to_eigen(edges[p.first].pos(p.second * edges[p.first].length()));
    out.append(std::move(line));
  }
  return out;
}

py::tuple
endpoint_classifier_candidates(const StrokeGraph& graph,
                               const std::unordered_set<std::size_t>& ignored_strokes,
                               const int n_closest) {
  if (n_closest < 1)
    throw std::invalid_argument("n_closest must be at least 1");
  const auto nverts = graph.vertices_.size();

  struct Hash {
    size_t operator()(const std::pair<Endpoint::IdType, Endpoint::IdType>& p) const {
      return size_t(p.first) * 7877 + size_t(p.second);
    };
  };
  auto candidates =
    std::unordered_set<std::pair<Endpoint::IdType, Endpoint::IdType>, Hash>();
  for (auto i = decltype(nverts){0}; i < nverts; ++i) {
    const auto vertex = graph.vertex(i);
    if (!vertex.is_active() || !vertex.is_dangling()) {
      continue;
    }
    const auto si = vertex.hedge().orig_stroke_idx();
    if (!contains(ignored_strokes, si)) {
      const auto endp = Endpoint(si, vertex.hedge().forward());
      const auto hits = pick_endpoints(graph, endp, n_closest, &ignored_strokes);
      for (const auto other_endp : hits) {
        const auto& other_stroke = graph.orig_strokes_[other_endp.stroke_idx()];
        const auto other_vertex_it = graph.endpoint2vertex_.find(other_endp);
        assert(other_vertex_it != graph.endpoint2vertex_.end());
        const auto q =
          (other_endp.is_head() ? other_stroke.xy(0) : other_stroke.xy(Back));
        if (line_of_sight(vertex.pos(), q, *graph.orig_bvh_)) {
          if (endp.as_int() < other_endp.as_int()) {
            candidates.emplace(endp.as_int(), other_endp.as_int());
          } else {
            candidates.emplace(other_endp.as_int(), endp.as_int());
          }
        }
      }
    }
  }

  auto first = Vector<Endpoint::IdType>(candidates.size(), 1);
  auto second = Vector<Endpoint::IdType>(candidates.size(), 1);
  auto row = Index(0);
  for (const auto& [cand1, cand2] : candidates) {
    first(row) = cand1;
    second(row) = cand2;
    row++;
  }

  return py::make_tuple(std::move(first), std::move(second));
}

py::tuple
tjunc_classifier_candidates(const StrokeGraph& graph,
                            const std::unordered_set<std::size_t>& ignored_strokes,
                            const int n_closest) {
  if (n_closest < 1)
    throw std::invalid_argument("n_closest must be at least 1");
  const auto nverts = graph.vertices_.size();
  auto first = Vector<Endpoint::IdType>(n_closest * nverts, 1);
  // double is OK if indices don't get too large.
  auto second = Eigen::Matrix<double, Dynamic, 2, Eigen::RowMajor>(n_closest * nverts, 2);
  auto row = Index(0);

  assert(ignored_strokes.empty() && "ignored_strokes no longer supported");

  for (auto i = decltype(nverts){0}; i < nverts; ++i) {
    const auto vertex = graph.vertex(i);
    if (!vertex.is_active() || !vertex.is_dangling() ||
        (vertex.flags() & StrokeGraph::VertexRecord::Overlapping)) {
      continue;
    }
    const auto si = vertex.hedge().orig_stroke_idx();
    const auto& s = graph.orig_strokes_[si];
    const auto endp = Endpoint(si, vertex.hedge().forward());
    const auto hits = pick(graph, endp, n_closest, &ignored_strokes);

    auto end1 =
      StrokeTime((int)vertex.hedge().stroke_idx(), vertex.hedge().forward() ? 0.0 : 1.0);
    const auto ok = convert_strokes2orig(graph, end1);
    force_assert(ok && "couldn't map from strokes to orig");
    assert((endp.is_head() && end1.second == 0.0) ||
           (!endp.is_head() && end1.second == 1.0));
    assert((size_t)end1.first == si);

    for (const auto& [other_si, other_norm_arclen] : hits) {
      const auto& other_stroke = graph.orig_strokes_[other_si];
      const auto other_arclen = other_norm_arclen * other_stroke.length();
      const auto p = (endp.is_head() ? s.xy(0) : s.xy(Back));
      const auto q = other_stroke.pos(other_arclen);
      const auto cen_dist = (p - q).norm();
      const auto env_dist =
        cen_dist - 0.5 * other_stroke.width_at(other_arclen) - vertex_radius(vertex);
      if ((env_dist < env_distance_over_pen_width_hard_threshold * 0.5 *
                        (s.pen_width() + other_stroke.pen_width()) ||
           env_dist < std::min(s.length(), other_stroke.length())) &&
          end_stroke_junction_type(other_stroke, other_arclen) != 0 &&
          // !intersecting_and_diverging_t(s, end1.second * s.length(), other_stroke,
          //                               other_arclen) &&
          line_of_sight(p, q, *graph.orig_bvh_)) {
        first(row) = endp.as_int();
        second(row, 0) = static_cast<double>(other_si);
        second(row, 1) = other_norm_arclen;
        row++;
      }
    }
  }

  first.conservativeResize(row);
  second.conservativeResize(row, Eigen::NoChange);
  return py::make_tuple(std::move(first), std::move(second));
}

Vector<std::uint8_t>
endpoint_classifier_labels(const py::EigenDRef<const Vector<Endpoint::IdType>>& cand1,
                           const py::EigenDRef<const Vector<Endpoint::IdType>>& cand2,
                           const py::EigenDRef<const Matchups>& matchups_head,
                           const py::EigenDRef<const Matchups>& matchups_tail) {
  if (cand1.size() != cand2.size()) {
    throw std::invalid_argument("both candidate vectors must be the same size");
  }
  if (2 * matchups_head.rows() != matchups_head.cols()) {
    throw std::invalid_argument(
      "matchups_head must have twice the number of cols as rows");
  }
  if (matchups_head.rows() != matchups_tail.rows() ||
      matchups_head.cols() != matchups_tail.cols()) {
    throw std::invalid_argument("sizes of matchup matrices must match");
  }

  auto labels = Vector<std::uint8_t>(cand1.size(), 1);
  for (auto row = Index(0); row < cand1.size(); ++row) {
    const auto e1 = Endpoint(cand1(row));
    const auto e2 = Endpoint(cand2(row));
    const auto& mat = (e1.is_head() ? matchups_head : matchups_tail);
    // Label: are they matched?
    const auto match_type =
      std::uint8_t{mat(e1.stroke_idx(), 2 * e2.stroke_idx() + (e2.is_head() ? 0 : 1))};
    labels(row) = match_type;
  }
  return labels;
}

Vector<std::uint8_t>
tjunc_classifier_labels(const py::EigenDRef<const Vector<Endpoint::IdType>>& cand1,
                        const py::EigenDRef<const MatRowMaj>& cand2,
                        const py::EigenDRef<const Matchups>& matchups_head,
                        const py::EigenDRef<const Matchups>& matchups_tail) {
  if (cand1.rows() != cand2.rows()) {
    throw std::invalid_argument("both candidate vectors must be the same size");
  }
  if (matchups_head.rows() != matchups_head.cols()) {
    throw std::invalid_argument("matchups_head must be square");
  }
  if (matchups_head.rows() != matchups_tail.rows() ||
      matchups_head.cols() != matchups_tail.cols()) {
    throw std::invalid_argument("sizes of matchup matrices must match");
  }

  auto labels = Vector<std::uint8_t>(cand1.size(), 1);
  for (auto row = Index(0); row < cand1.size(); ++row) {
    const auto e1 = Endpoint(cand1(row));
    const auto& mat = (e1.is_head() ? matchups_head : matchups_tail);
    const auto s2 = (Index)cand2(row, 0);
    // Label: are they matched?
    const auto match_type = std::uint8_t{mat(e1.stroke_idx(), s2)};
    labels(row) = match_type;
  }
  return labels;
}

MatRowMaj
endpoint_classifier_features(const StrokeGraph& graph,
                             const py::EigenDRef<const Vector<Endpoint::IdType>>& cand1,
                             const py::EigenDRef<const Vector<Endpoint::IdType>>& cand2,
                             const py::list& features) {
  if (cand1.size() != cand2.size()) {
    throw std::invalid_argument("both candidate vectors must be the same size");
  }

  const auto bvh = PolylineBVH(graph.orig_strokes_);
  for (const auto& node : bvh.nodes) {
    node.geometry->ensure_arclengths();
  }
  for (const auto& element : features) {
    auto& f = element.cast<JunctionFeature&>();
    f.init(bvh);
  }
  const auto n = features.size();
  auto feature_mat = MatRowMaj(cand1.size(), n);

  for (auto row = Index(0); row < cand1.size(); ++row) {
    const auto e1 = Endpoint(cand1(row));
    const auto e2 = Endpoint(cand2(row));
    const auto& s1 = *bvh.nodes[e1.stroke_idx()].geometry;
    const auto& s2 = *bvh.nodes[e2.stroke_idx()].geometry;
    for (auto col = decltype(n){0}; col < n; ++col) {
      const auto& f = features[col].cast<JunctionFeature&>();
      feature_mat(row, col) = f(s1, (e1.is_head() ? 0.0 : s1.length()), e1.stroke_idx(),
                                s2, (e2.is_head() ? 0.0 : s2.length()), e2.stroke_idx());
    }
  }
  return feature_mat;
}

MatRowMaj tjunc_classifier_features(
  const StrokeGraph& graph, const py::EigenDRef<const Vector<Endpoint::IdType>>& cand1,
  const py::EigenDRef<const Eigen::Matrix<double, Dynamic, 2>>& cand2,
  const py::list& features) {
  if (cand1.rows() != cand2.rows()) {
    throw std::invalid_argument("both candidate vectors must be the same size");
  }

  const auto bvh = PolylineBVH(graph.orig_strokes_);
  for (const auto& node : bvh.nodes) {
    node.geometry->ensure_arclengths();
  }
  for (const auto& element : features) {
    auto& f = element.cast<JunctionFeature&>();
    f.init(bvh);
  }
  const auto n = features.size();
  auto feature_mat = MatRowMaj(cand1.rows(), features.size());

  for (auto row = Index(0); row < cand1.size(); ++row) {
    const auto e1 = Endpoint(cand1(row));
    const auto& s1 = *bvh.nodes[e1.stroke_idx()].geometry;
    const auto stroke_idx2 = (std::size_t)cand2(row, 0);
    const auto& s2 = *bvh.nodes[stroke_idx2].geometry;
    for (auto col = decltype(n){0}; col < n; ++col) {
      const auto& f = features[col].cast<JunctionFeature&>();
      feature_mat(row, col) = f(s1, (e1.is_head() ? 0.0 : s1.length()), e1.stroke_idx(),
                                s2, cand2(row, 1) * s2.length(), stroke_idx2);
    }
  }
  return feature_mat;
}

Eigen::MatrixX2d junction_centroids(const py::list& junctions, const Drawing& d) {
  Eigen::MatrixX2d out(junctions.size(), 2);
  for (std::size_t i = 0; i < junctions.size(); ++i) {
    const auto& junc = junctions[i].cast<Junction&>();
    out.row(i) = to_eigen(centroid(junc, d.strokes()));
  }
  return out;
}

Eigen::MatrixX2d junction_positions(const py::list& junctions, const Drawing& d) {
  Index nrows = 0;
  for (const auto& item : junctions) {
    const auto& junc = item.cast<Junction&>();
    nrows += junc.size();
  }

  Eigen::MatrixX2d out(nrows, 2);
  Index row = 0;
  for (std::size_t i = 0; row < nrows; ++i) {
    const auto& junc = junctions[i].cast<Junction&>();
    const auto n = junc.size();
    for (std::size_t j = 0; j < n; ++j) {
      const auto& stroke = d.strokes()[junc.stroke(j)];
      out.row(row) = to_eigen(stroke.pos(junc.arclength(j) * stroke.length()));
      row++;
    }
  }
  return out;
}

static std::vector<Junction> convert_label_indices(const std::vector<Junction>& junctions,
                                                   const Drawing& old_drawing,
                                                   const Drawing& new_drawing) {
  auto new_bvh = PolylineBVH();
  for (const auto& stroke : new_drawing.strokes()) {
    stroke.ensure_arclengths();
    new_bvh.nodes.emplace_back(stroke, bounds(stroke));
  }
  for (const auto& stroke : old_drawing.strokes()) {
    stroke.ensure_arclengths();
  }

  auto out_junctions = std::vector<Junction>();

  for (const auto& junction : junctions) {
    auto out_junction = Junction(junction.type);
    for (size_t i = 0; i < junction.points.size(); ++i) {
      const auto [orig_si, orig_narclen] = junction.points[i];
      const auto p = old_drawing.strokes()[orig_si].pos_norm(orig_narclen);
      if (orig_narclen == 0 || orig_narclen == 1) {
        // Project to nearest endpoint.
        const auto hits = pick_endpoints(new_bvh, p);
        if (hits.empty()) {
          break; // Could not map. Skip.
        } else {
          out_junction.points.emplace_back(
            StrokeTime((int)hits[0].first, hits[0].second));
        }
      } else {
        // Project to nearest stroke.
        auto best_dist = Float(INFINITY);
        auto best_si = -1;
        auto best_arclen = Float(-1.0);
        const auto n_new_strokes = new_drawing.strokes().size();
        for (size_t si = 0; si < n_new_strokes; ++si) {
          Vec2 proj;
          Float s;
          const auto dist = closest_point(new_bvh.nodes[si], p, best_dist, proj, s);
          if (dist < best_dist) {
            best_dist = dist;
            best_si = (int)si;
            best_arclen = s / new_drawing.strokes()[si].length();
          }
        }
        if (best_si >= 0) {
          out_junction.points.emplace_back(StrokeTime(best_si, best_arclen));
        } else {
          break; // Could not map.
        }
      }
    }

    if (out_junction.points.size() == junction.points.size()) {
      out_junctions.emplace_back(std::move(out_junction));
    } else {
      SPDLOG_WARN("An endpoint was dropped");
    }
  }

  return out_junctions;
}

} // namespace

void init_junction(py::module& m) {
  py::class_<Junction> junction(m, "Junction");
  junction
    .def(
      py::init<std::vector<std::pair<int, double>>, const std::string&, bool, double>(),
      "points"_a, "type"_a, "is_weak"_a = false, "probability"_a = -1.0)
    .def(py::pickle(
      [](const Junction& p) { // __getstate__
        py::dict out;
        py::list points;
        for (const auto& [s, a] : p.points) {
          py::list pair;
          pair.append(s);
          pair.append(a);
          points.append(std::move(pair));
        }
        out["points"] = std::move(points);
        out["type"] = std::string(junction_type_to_char(p.type), 1);
        if (p.is_weak)
          out["is_weak"] = p.is_weak;
        if (p.probability)
          out["probability"] = p.probability;
        return out;
      },
      [](py::dict& state) { // __setstate__
        return Junction(state["points"].cast<decltype(Junction::points)>(),
                        state["type"].cast<std::string>(),
                        state.attr("get")("is_weak", false).cast<bool>(),
                        state.attr("get")("probability", -1).cast<double>());
      }))
    .def_property(
      "type", [](const Junction& j) { return j.type; },
      [](Junction& j, JunctionType::Type type) { j.type = type; })
    .def_property(
      "repr", [](const Junction& j) { return j.repr; },
      [](Junction& j, std::string repr) { j.repr = repr; })
    .def_property_readonly("corner_type", [](const Junction& j) { return j.corner_type; })
    .def_property_readonly("orig_dist", [](const Junction& j) { return j.orig_dist; })
    .def_property_readonly("fea", [](const Junction& j) { return j.fea; })
    .def_property_readonly("points", [](const Junction& junc) { return junc.points; })
    .def_property(
      "is_weak", [](const Junction& j) { return j.is_weak; },
      [](Junction& j, bool b) { j.is_weak = b; })
    .def_property(
      "probability", [](const Junction& j) { return j.probability; },
      [](Junction& j, double p) { j.probability = p; })
    .def_property(
      "alt_probability", [](const Junction& j) { return j.alt_probability; },
      [](Junction& j, double p) { j.alt_probability = p; })
    .def("__repr__", &junction_repr)
    .def("__len__", &Junction::size)
    .def("__lt__", &Junction::operator<)
    .def("__getitem__",
         [](const Junction& j, int i) {
           if (i < 0)
             i += static_cast<int>(j.size());
           if (i < 0 || i >= static_cast<int>(j.size())) {
             std::stringstream ss;
             ss << i << " was out of range";
             throw std::out_of_range(ss.str());
           }
           return j.points[i];
         })
    .def("strokes",
         [](const Junction& junc) {
           py::set out;
           for (const auto& p : junc.points) {
             const auto ok = out.add(p.first);
             assert(ok);
           }
           return out;
         })
    .def("clone", [](Junction j) { return j; })
    .def("add", &Junction::add)
    .def("merge", &Junction::merge)
    .def("merged", &Junction::merged)
    .def("normalize",
         [](Junction& junc, const Drawing& d) { junc.normalize(d.strokes()); })
    .def("normalized",
         [](Junction& junc, const Drawing& d) { return junc.normalized(d.strokes()); })
    .def("positions", &positions)
    .def("position",
         [](const Junction& j, const Drawing& d) { return centroid(j, d.strokes()); })
    .def("remove_duplicate_points", &Junction::remove_duplicate_points)
    .def("sort_entries", &Junction::sort_entries)
    .def("throw_on_invalid", &Junction::throw_on_invalid)
    .def("star", &junction_star, "d"_a);
  py::enum_<JunctionType::Type>(junction, "Type")
    .value("R", JunctionType::Type::R)
    .value("T", JunctionType::Type::T)
    .value("X", JunctionType::Type::X)
    .export_values();

  py::class_<ExtendedJunction::Range>(m, "Range")
    .def("__repr__", &ExtendedJunction::Range::repr)
    .def_property_readonly("start", [](ExtendedJunction::Range& j) { return j.start_; })
    .def_property_readonly("mid", [](ExtendedJunction::Range& j) { return j.mid_; })
    .def_property_readonly("end", [](ExtendedJunction::Range& j) { return j.end_; });

  py::class_<ExtendedJunction>(m, "ExtendedJunction")
    .def_property_readonly("ranges", [](ExtendedJunction& j) {
      py::list out;
      for (const auto& [i, r] : j.ranges_) {
        out.append(py::make_tuple(i, r));
      }
      return out;
    });

  m.def("intersection_junctions", [](PolylineBVH& bvh) {
    for (const auto& node : bvh.nodes) {
      node.geometry->ensure_arclengths();
    }
    auto junctions = std::vector<Junction>();
    intersection_junctions(bvh, junctions);
    return junctions;
  });

  m.def("convert_label_indices", &convert_label_indices);

  m.def("endpoint_classifier_candidates", &endpoint_classifier_candidates, //
        "graph"_a, "ignored_strokes"_a, "n_closest"_a = 3);
  m.def("tjunc_classifier_candidates", &tjunc_classifier_candidates, //
        "graph"_a, "ignored_strokes"_a, "n_closest"_a = 5);
  m.def("endpoint_classifier_labels", &endpoint_classifier_labels, //
        "cand1"_a, "cand2"_a, "matchups_head"_a, "matchups_tail"_a);
  m.def("tjunc_classifier_labels", &tjunc_classifier_labels, //
        "cand1"_a, "cand2"_a, "matchups_head"_a, "matchups_tail"_a);
  m.def("endpoint_classifier_features", &endpoint_classifier_features, //
        "Return a feature matrix for training an endpoint classifier.", //
        "bvh"_a, "cand1"_a, "cand2"_a, "features"_a);
  m.def("tjunc_classifier_features", &tjunc_classifier_features, //
        "Return a feature matrix for training a T-junction classifier.", //
        "bvh"_a, "cand1"_a, "cand2"_a, "features"_a);

  m.def("junction_centroids", &junction_centroids, "junctions"_a, "strokes"_a);
  m.def("junction_positions", &junction_positions, "junctions"_a, "strokes"_a);
}
