#include "json.h"

#include "../base64.h"
#include "../force_assert.h"
#include "../stroke_graph.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>

namespace sketching {

namespace nl = nlohmann;
using SnappingType = StrokeGraph::SnappingType;

// TODO: These functions will leak memory returned by the base64 encode/decode functions
//       if exceptions are thrown at certain places.

static Stroke decode_stroke(const nl::basic_json<>& stroke) {
  size_t decoded_len = 0;

  auto encoded = stroke["x"].get<std::string>();
  unsigned char* buf =
    base64_decode((unsigned char*)encoded.data(), encoded.length(), &decoded_len);
  force_assert(decoded_len % sizeof(Float) == 0);
  const auto n = decoded_len / sizeof(Float);
  auto out_stroke = Stroke(n, stroke.contains("time"));
  const auto* buf_as_float = (Float*)buf;
  std::copy(buf_as_float, buf_as_float + n, out_stroke.x_);
  free(buf);

  encoded = stroke["y"].get<std::string>();
  buf = base64_decode((unsigned char*)encoded.data(), encoded.length(), &decoded_len);
  force_assert(decoded_len == sizeof(Float) * n);
  buf_as_float = (Float*)buf;
  std::copy(buf_as_float, buf_as_float + n, out_stroke.y_);
  free(buf);

  encoded = stroke["width"].get<std::string>();
  buf = base64_decode((unsigned char*)encoded.data(), encoded.length(), &decoded_len);
  force_assert(decoded_len == sizeof(Float) * n);
  buf_as_float = (Float*)buf;
  std::copy(buf_as_float, buf_as_float + n, out_stroke.width_);
  free(buf);

  if (out_stroke.has_time()) {
    encoded = stroke["time"].get<std::string>();
    buf = base64_decode((unsigned char*)encoded.data(), encoded.length(), &decoded_len);
    force_assert(decoded_len == sizeof(Float) * n);
    buf_as_float = (Float*)buf;
    std::copy(buf_as_float, buf_as_float + n, out_stroke.time_);
    free(buf);
  }
  return out_stroke;
}

void load_json(const std::string& path, StrokeGraph& out_graph) {
  auto file = std::ifstream(path);
  auto obj = nl::json();
  file >> obj;
  file.close();

  const auto& version = obj["version"];
  force_assert(version.is_number_integer() && version.get<int>() == 0);

  const auto& boundary_face = obj["boundary_face"];
  force_assert(boundary_face.is_number_integer());
  out_graph.boundary_face_ = boundary_face.get<size_t>();

  const auto& snapping_method_type = obj["snapping_method_type"];
  force_assert(snapping_method_type.is_number_integer());
  out_graph.snapping_method_type_ = (SnappingType)snapping_method_type.get<size_t>();

  const auto& vertices = obj["vertices"];
  force_assert(vertices.is_array());
  const auto n_vertices = vertices.size();
  out_graph.vertices_.clear();
  out_graph.vertices_.reserve(n_vertices);
  for (size_t vi = 0; vi < n_vertices; ++vi) {
    const auto& vertex = vertices[vi];
    auto& out_vertex = out_graph.vertices_.emplace_back(Vec2());
    out_vertex.p_.x_ = vertex["x"].get<Float>();
    out_vertex.p_.y_ = vertex["y"].get<Float>();
    out_vertex.hedge_ = vertex["hedge"].get<size_t>();
    out_vertex.flags_ = (decltype(out_vertex.flags_))vertex["flags"].get<size_t>();
    if (vertex.contains("ids")) {
      const auto& ids = vertex["ids"];
      force_assert(ids.is_array());
      for (const auto& id : ids) {
        force_assert(id.is_array());
        auto& out_id = out_vertex.ids_.emplace_back();
        out_id.connection_type_ = id[0].get<decltype(out_id.connection_type_)>();
        out_id.connection_index_ = id[1].get<decltype(out_id.connection_index_)>();
      }
    }
  }

  const auto& hedges = obj["hedges"];
  force_assert(hedges.is_array());
  const auto n_hedges = hedges.size();
  out_graph.hedges_.resize(n_hedges);
  for (size_t hi = 0; hi < n_hedges; ++hi) {
    const auto& hedge = hedges[hi];
    auto& out_hedge = out_graph.hedges_[hi];
    out_hedge.next_ = hedge["next"].get<size_t>();
    out_hedge.prev_ = hedge["prev"].get<size_t>();
    out_hedge.origin_ = hedge["origin"].get<size_t>();
    out_hedge.continuity_ = hedge["continuity"].get<size_t>();
    out_hedge.face_ = hedge["face"].get<size_t>();
    out_hedge.flags_ = (decltype(out_hedge.flags_))hedge["flags"].get<size_t>();
  }

  const auto& faces = obj["faces"];
  force_assert(faces.is_array());
  const auto n_faces = faces.size();
  out_graph.faces_.resize(n_faces);
  for (size_t fi = 0; fi < n_faces; ++fi) {
    const auto& face = faces[fi];
    auto& out_face = out_graph.faces_[fi];
    out_face.cycles_ = face["cycles"].get<std::vector<size_t>>();
  }

  const auto& strokes = obj["strokes"];
  force_assert(strokes.is_array());
  const auto n_strokes = strokes.size();
  out_graph.bvh_.clear_all();
  for (size_t si = 0; si < n_strokes; ++si) {
    out_graph.bvh_.add(decode_stroke(strokes[si]));
  }

  out_graph.strokes2vid_.clear();
  out_graph.strokes2vid_.resize(out_graph.strokes_.size());

  if (obj.contains("orig_strokes")) {
    // The orig_strokes, orig2strokes, features, and snap_history fields were added
    // together, thus the presence of any one of these should imply any of the other.

    const auto& orig_strokes = obj["orig_strokes"];
    force_assert(orig_strokes.is_array());
    const auto n_orig_strokes = orig_strokes.size();
    out_graph.orig_strokes_.reserve(n_orig_strokes);
    for (size_t orig_si = 0; orig_si < n_orig_strokes; ++orig_si) {
      out_graph.orig_strokes_.emplace_back(decode_stroke(orig_strokes[orig_si]));
    }

    const auto& flat_orig2strokes = obj["orig2strokes"];
    force_assert(flat_orig2strokes.is_array());
    force_assert(flat_orig2strokes.size() % 6 == 0);
    const auto n_mappings = flat_orig2strokes.size() / 6;
    out_graph.orig2strokes_.resize(n_orig_strokes);
    out_graph.strokes2orig_.resize(n_strokes);
    for (size_t i = 0; i < n_mappings; ++i) {
      const auto orig_si = flat_orig2strokes[6 * i].get<size_t>();
      const auto si = flat_orig2strokes[6 * i + 1].get<size_t>();
      const auto domain_start = flat_orig2strokes[6 * i + 2].get<double>();
      const auto domain_stop = flat_orig2strokes[6 * i + 3].get<double>();
      const auto range_start = flat_orig2strokes[6 * i + 4].get<double>();
      const auto range_stop = flat_orig2strokes[6 * i + 5].get<double>();

      auto& record = out_graph.orig2strokes_[orig_si].emplace_back();
      record.first = si;
      record.second.domain_arclens_.resize(2);
      record.second.range_arclens_.resize(2);
      record.second.domain_arclens_[0] = domain_start;
      record.second.domain_arclens_[1] = domain_stop;
      record.second.range_arclens_[0] = range_start;
      record.second.range_arclens_[1] = range_stop;
      auto& s2o = out_graph.strokes2orig_[si];
      if (std::find(s2o.begin(), s2o.end(), orig_si) == s2o.end()) {
        s2o.push_back(orig_si);
      }
    }

    auto compatible_features = true;
    const auto& feature_schema = obj["features"];
    force_assert(feature_schema["r"].is_array() && feature_schema["t"].is_array());
    const auto& ee_schema = feature_schema["r"];
    const auto& es_schema = feature_schema["t"];
    if (ee_schema.size() != EndEndFeatures::n_features_ ||
        es_schema.size() != EndStrokeFeatures::n_features_) {
      compatible_features = false;
    } else {
      const auto ee_descriptions = get_end_end_feature_descriptions();
      for (size_t i = 0; i < ee_descriptions.size(); ++i) {
        if (ee_descriptions[i] != ee_schema[i].get<std::string>()) {
          compatible_features = false;
          break;
        }
      }
      const auto es_descriptions = get_end_stroke_feature_descriptions();
      for (size_t i = 0; i < es_descriptions.size(); ++i) {
        if (es_descriptions[i] != es_schema[i].get<std::string>()) {
          compatible_features = false;
          break;
        }
      }
    }

    const auto& snap_history = obj["snap_history"];
    force_assert(snap_history.is_array());
    const auto n_snaps = snap_history.size();
    out_graph.snap_history_.reserve(n_snaps);
    for (size_t i = 0; i < n_snaps; ++i) {
      const auto& snap = snap_history[i];
      force_assert(snap["key"].is_array() && snap["key"].size() == 3);
      auto& out_snap = out_graph.snap_history_.emplace_back();
      const auto type_str = snap["key"][0].get<std::string>();
      out_snap.key.type = junction_type_from_char(type_str[0]);
      out_snap.key.cand1 = snap["key"][1];
      out_snap.key.cand2 = snap["key"][2];
      out_snap.p_a.x_ = snap["p_a"][0];
      out_snap.p_a.y_ = snap["p_a"][1];
      out_snap.p_b.x_ = snap["p_b"][0];
      out_snap.p_b.y_ = snap["p_b"][1];
      out_snap.orig_a.first = snap["orig_a"][0];
      out_snap.orig_a.second = snap["orig_a"][1];
      out_snap.orig_b.first = snap["orig_b"][0];
      out_snap.orig_b.second = snap["orig_b"][1];
      out_snap.prob = snap["prob"];
      if (compatible_features && snap.contains("fea")) {
        const auto& fea = snap["fea"];
        const auto fea_type_str = fea["type"].get<std::string>();
        const auto type = (FeatureVector::Type)junction_type_from_char(fea[0]);
        out_snap.fea.type_ = type;
        const auto data = span<Float>(out_snap.fea);
        for (size_t k = 0; k < data.size(); ++k) {
          data[k] = fea[k + 1];
        }
      }
      out_snap.connected = snap["connected"];
    }
  }

  out_graph.orig_bvh_ = std::make_unique<PolylineBVH>(out_graph.orig_strokes_);
}

void save_json(const StrokeGraph& graph, const std::string& path) {
  auto file = std::ofstream();
  file.open(path);
  // Enough precision to round-trip a double precision float.
  file << std::scientific << std::setprecision(std::numeric_limits<double>::digits10 + 1);

  file << '{';
  // clang-format off
  file << "\"version\": 0, "
          "\"boundary_face\": " << graph.boundary_face_ << ", "
          "\"snapping_method_type\": " << (size_t)graph.snapping_method_type_ << ", ";
  // clang-format on

  file << "\"vertices\": [";
  for (size_t vi = 0; vi < graph.vertices_.size(); ++vi) {
    const auto& v = graph.vertices_[vi];
    file << "{";
    file << "\"x\": " << v.p_.x_ << ", ";
    file << "\"y\": " << v.p_.y_ << ", ";
    file << "\"hedge\": " << v.hedge_ << ", ";
    file << "\"flags\": " << v.flags_ << ", ";
    file << "\"ids\": [";
    for (size_t i = 0; i < v.ids_.size(); ++i) {
      file << "[" << v.ids_[i].connection_type_ << ", " << v.ids_[i].connection_index_
           << "]";
      if (i != v.ids_.size() - 1)
        file << ", ";
    }
    file << "]}";
    if (vi != graph.vertices_.size() - 1) {
      file << ", ";
    }
  }
  file << "], ";

  file << "\"hedges\": [";
  for (size_t hi = 0; hi < graph.hedges_.size(); ++hi) {
    const auto& he = graph.hedges_[hi];
    file << "{";
    file << "\"next\": " << he.next_ << ", ";
    file << "\"prev\": " << he.prev_ << ", ";
    file << "\"origin\": " << he.origin_ << ", ";
    file << "\"continuity\": " << he.continuity_ << ", ";
    file << "\"face\": " << he.face_ << ", ";
    file << "\"flags\": " << he.flags_ << "}";
    if (hi != graph.hedges_.size() - 1) {
      file << ", ";
    }
  }
  file << "], ";

  file << "\"faces\": [";
  for (size_t fi = 0; fi < graph.faces_.size(); ++fi) {
    const auto& f = graph.faces_[fi];
    file << "{\"cycles\": [";
    for (size_t ci = 0; ci < f.cycles_.size(); ++ci) {
      file << f.cycles_[ci];
      if (ci != f.cycles_.size() - 1) {
        file << ", ";
      }
    }
    file << "]}";
    if (fi != graph.faces_.size() - 1) {
      file << ", ";
    }
  }
  file << "], ";

  for (const auto& strokes :
       {graph.bvh_.strokes(), span<const Stroke>(graph.orig_strokes_)}) {
    if (&strokes[0] == &graph.strokes_[0]) {
      file << "\"strokes\": [";
    } else {
      file << "\"orig_strokes\": [";
    }
    for (size_t si = 0; si < strokes.size(); ++si) {
      const auto& stroke = strokes[si];
      const auto n = stroke.size();
      unsigned char* buf = nullptr;
      file << "{";

      file << "\"x\": \"";
      buf = base64_encode((unsigned char*)stroke.x_, sizeof(Float) * n, nullptr);
      file << buf;
      free(buf);
      file << "\", ";

      file << "\"y\": \"";
      buf = base64_encode((unsigned char*)stroke.y_, sizeof(Float) * n, nullptr);
      file << buf;
      free(buf);
      file << "\", ";

      file << "\"width\": \"";
      buf = base64_encode((unsigned char*)stroke.width_, sizeof(Float) * n, nullptr);
      file << buf;
      free(buf);

      if (stroke.has_time()) {
        file << "\", ";

        file << "\"time\": \"";
        buf = base64_encode((unsigned char*)stroke.time_, sizeof(Float) * n, nullptr);
        file << buf;
        free(buf);
      }

      file << "\"}";
      if (si != strokes.size() - 1) {
        file << ", ";
      }
    }
    file << "], ";
  }

  file << "\"orig2strokes\": [";
  auto first_iter = true;
  for (size_t orig_si = 0; orig_si < graph.orig2strokes_.size(); ++orig_si) {
    for (const auto& [si, map] : graph.orig2strokes_[orig_si]) {
      if (!first_iter) {
        file << ", ";
      }
      first_iter = false;
      file << orig_si << ", " << si << ", " //
           << map.domain_arclens_[0] << ", " << map.domain_arclens_[1] << ", " //
           << map.range_arclens_[0] << ", " << map.range_arclens_[1];
    }
  }
  file << "], ";

  file << "\"features\": {"
          "\"r\": [";
  const auto ee_descriptions = get_end_end_feature_descriptions();
  for (size_t i = 0; i < ee_descriptions.size(); ++i) {
    if (i != 0)
      file << ", ";
    file << '"' << ee_descriptions[i] << '"';
  }
  file << "], \"t\": [";
  const auto es_descriptions = get_end_end_feature_descriptions();
  for (size_t i = 0; i < es_descriptions.size(); ++i) {
    if (i != 0)
      file << ", ";
    file << '"' << es_descriptions[i] << '"';
  }
  file << "]}, ";

  file << "\"snap_history\": [";
  for (size_t i = 0; i < graph.snap_history_.size(); ++i) {
    const auto& info = graph.snap_history_[i];
    file << "{"
            "\"key\": [\""
         << junction_type_to_char(info.key.type) << "\", " << info.key.cand1 << ", "
         << info.key.cand2 << "], ";
    file << "\"p_a\": [" << info.p_a.x_ << ", " << info.p_a.y_ << "], ";
    file << "\"p_b\": [" << info.p_b.x_ << ", " << info.p_b.y_ << "], ";
    file << "\"orig_a\": [" << info.orig_a.first << ", " << info.orig_a.second << "], ";
    file << "\"orig_b\": [" << info.orig_b.first << ", " << info.orig_b.second << "], ";
    file << "\"prob\": " << info.prob << ", ";
    if (info.fea.type_ != FeatureVector::Uninitialized) {
      file << "\"fea\": [\"" << junction_type_to_char((JunctionType::Type)info.fea.type_)
           << '"';
      auto features = span<const Float>(info.fea);
      for (size_t f = 0; f < features.size(); ++f) {
        file << ", " << features[f];
      }
      file << "], ";
    }
    file << "\"connected\": " << (info.connected ? "true" : "false") << '}';
    if (i != graph.snap_history_.size() - 1)
      file << ", ";
  }
  file << "]";

  file << "}\n";
  file.close();
}

} // namespace sketching
