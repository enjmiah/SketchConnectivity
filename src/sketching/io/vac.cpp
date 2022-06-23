#include "vac.h"

#include "../sketching.h"
#include "../stroke_graph.h"
#include "error.h"

#include <pugixml.hpp>

#include <iomanip>
#include <unordered_set>

using namespace sketching;
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4455)
#endif
using std::literals::string_literals::operator""s;
#ifdef _MSC_VER
#pragma warning(pop)
#endif

namespace {

bool starts_with(const std::string& s, const std::string& prefix) {
  return s.rfind(prefix, 0) == 0;
}

// From https://github.com/imageworks/pystring/blob/master/pystring.cpp
void split_whitespace(const std::string& str, std::vector<Float>& result) {
  std::string::size_type i, j, len = str.size();
  for (i = j = 0; i < len;) {

    while (i < len && ::isspace(str[i]))
      i++;
    j = i;

    while (i < len && !::isspace(str[i]))
      i++;

    if (j < i) {
      result.push_back(std::stod(str.substr(j, i - j)));

      while (i < len && ::isspace(str[i]))
        i++;
      j = i;
    }
  }
  if (j < len) {
    result.push_back(std::stod(str.substr(j, len - j)));
  }
}

} // namespace

namespace sketching {

void load_vac(const std::string& path, std::vector<Stroke>& out) {
  pugi::xml_document doc;
  const pugi::xml_parse_result result = doc.load_file(path.c_str());
  if (!result) {
    auto ss = std::stringstream();
    ss << "could not open file " << path;
    throw io_error(ss.str());
  }

  const auto vec = doc.child("vec");
  if (vec.attribute("version").value() != "1.7"s) {
    throw io_error("unsupported version, only 1.7 is supported");
  }

  for (pugi::xml_node layer : vec.children("layer")) {
    if (layer.attribute("visible").value() == "false"s) {
      continue;
    }
    for (pugi::xml_node edge : layer.child("objects").children("edge")) {
      const pugi::char_t* curve_str = edge.attribute("curve").value();
      int num_components;
      if (starts_with(curve_str, "xywtdense(")) {
        num_components = 4;
        curve_str = &curve_str[10];
      } else if (starts_with(curve_str, "xywdense(")) {
        num_components = 3;
        curve_str = &curve_str[9];
      } else {
        throw io_error("unknown curve type");
      }
      auto curve_str2 = std::string(curve_str);
      if (curve_str2.empty()) {
        throw io_error("curve was empty");
      }
      curve_str2.pop_back(); // remove closing ')'
      for (auto& c : curve_str2) {
        if (c == ',')
          c = ' ';
      }
      auto entries = std::vector<Float>();
      split_whitespace(curve_str2, entries);
      // const auto sampling = entries[0];
      if (entries.size() % num_components != 1) {
        throw io_error("wrong number of entries for specified data type");
      }
      const auto n = entries.size() / num_components;
      auto& stroke = out.emplace_back(n, num_components == 4);
      for (std::size_t i = 0; i < n; ++i) {
        stroke.x(i) = entries[1 + i * num_components];
        stroke.y(i) = entries[1 + i * num_components + 1];
        stroke.width(i) = entries[1 + i * num_components + 2];
        if (num_components == 4) {
          stroke.time(i) = entries[1 + i * num_components + 3];
        }
      }
    }
  }
}

void save_vac(const span<const Stroke> strokes, const std::string& path) {
  pugi::xml_document doc;
  auto comment = doc.append_child(pugi::node_comment);
  comment.set_value("Saved with libsketching 0.1.0");
  auto vec = doc.append_child("vec");
  {
    vec.append_attribute("version") = "1.7";

    auto playback = vec.append_child("playback");
    playback.append_attribute("framerange") = "0 47";
    playback.append_attribute("fps") = "24";
    playback.append_attribute("subframeinbetweening") = "off";
    playback.append_attribute("playmode") = "normal";

    auto canvas = vec.append_child("canvas");
    canvas.append_attribute("position") = "0 0";
    canvas.append_attribute("size") = "1280 720";

    auto layer = vec.append_child("layer");
    {
      layer.append_attribute("name") = "Layer 1";
      layer.append_attribute("visible") = "true";

      auto background = layer.append_child("background");
      background.append_attribute("color") = "rgba(255,255,255,0)";
      background.append_attribute("image") = "";
      background.append_attribute("position") = "0 0";
      background.append_attribute("size") = "cover";
      background.append_attribute("repeat") = "norepeat";
      background.append_attribute("opacity") = "1";
      background.append_attribute("hold") = "yes";

      auto objects = layer.append_child("objects");
      int cell_id = 0;
      for (const auto& s : strokes) {
        auto v_l = objects.append_child("vertex");
        v_l.append_attribute("id").set_value(cell_id++);
        v_l.append_attribute("frame") = "0";
        v_l.append_attribute("color") = "rgba(0,0,0,1)";
        {
          auto ss = std::stringstream();
          ss << s.x(0) << " " << s.y(0);
          auto str = ss.str();
          v_l.append_attribute("position").set_value(str.c_str());
        }

        auto v_r = objects.append_child("vertex");
        v_r.append_attribute("id").set_value(cell_id++);
        v_r.append_attribute("frame") = "0";
        v_r.append_attribute("color") = "rgba(0,0,0,1)";
        {
          auto ss = std::stringstream();
          ss << s.x(s.size() - 1) << " " << s.y(s.size() - 1);
          auto str = ss.str();
          v_r.append_attribute("position").set_value(str.c_str());
        }

        auto edge = objects.append_child("edge");
        edge.append_attribute("id").set_value(cell_id++);
        edge.append_attribute("frame") = "0";
        edge.append_attribute("startvertex").set_value(cell_id - 3);
        edge.append_attribute("endvertex").set_value(cell_id - 2);
        edge.append_attribute("color") = "rgba(0,0,0,1)";
        {
          auto ss = std::stringstream();
          // Enough precision to round-trip a double precision float.
          ss << std::scientific
             << std::setprecision(std::numeric_limits<double>::digits10 + 1);
          if (s.has_time()) {
            ss << "xywtdense(5"; // Type and ds (sampling) (nonsense ds value since
                                 // strokes aren't necessarily uniformly sampled).
            for (auto i = 0; i < s.size(); ++i) {
              ss << ' ' << s.x(i) << ',' << s.y(i) << ',' << s.width(i) << ','
                 << s.time(i);
            }
          } else {
            ss << "xywdense(5"; // Same comment.
            for (auto i = 0; i < s.size(); ++i) {
              ss << ' ' << s.x(i) << ',' << s.y(i) << ',' << s.width(i);
            }
          }
          ss << ")";
          const auto str = ss.str();
          edge.append_attribute("curve").set_value(str.c_str());
        }
      }
    }
  }
  const auto ok = doc.save_file(path.c_str(), "  ");
  if (!ok) {
    auto ss = std::stringstream();
    ss << "could not save file to " << path;
    throw io_error(ss.str());
  }
}

void save_vac(const StrokeGraph& graph, const std::string& path) {
  pugi::xml_document doc;
  auto comment = doc.append_child(pugi::node_comment);
  comment.set_value("Saved with libsketching 0.1.0");
  auto vec = doc.append_child("vec");
  {
    vec.append_attribute("version") = "1.7";

    auto playback = vec.append_child("playback");
    playback.append_attribute("framerange") = "0 47";
    playback.append_attribute("fps") = "24";
    playback.append_attribute("subframeinbetweening") = "off";
    playback.append_attribute("playmode") = "normal";

    auto canvas = vec.append_child("canvas");
    canvas.append_attribute("position") = "0 0";
    canvas.append_attribute("size") = "1280 720";

    auto layer = vec.append_child("layer");
    {
      layer.append_attribute("name") = "Layer 1";
      layer.append_attribute("visible") = "true";

      auto background = layer.append_child("background");
      background.append_attribute("color") = "rgba(255,255,255,0)";
      background.append_attribute("image") = "";
      background.append_attribute("position") = "0 0";
      background.append_attribute("size") = "cover";
      background.append_attribute("repeat") = "norepeat";
      background.append_attribute("opacity") = "1";
      background.append_attribute("hold") = "yes";

      auto objects = layer.append_child("objects");
      int cell_id = 0;
      for (const auto& v : graph.vertices_) {
        // TODO: Skip inactive vertices.
        auto new_v = objects.append_child("vertex");
        new_v.append_attribute("id").set_value(cell_id++);
        new_v.append_attribute("frame") = "0";
        new_v.append_attribute("color") = "rgba(0,0,0,1)";
        {
          auto ss = std::stringstream();
          ss << v.p_.x() << " " << v.p_.y();
          const auto str = ss.str();
          new_v.append_attribute("position").set_value(str.c_str());
        }
      }
      auto hedge2cell = std::unordered_map<size_t, int>();
      for (size_t hi = 0; hi < graph.hedges_.size(); ++hi) {
        const auto& he = graph.hedge(hi);
        if (!he.forward())
          continue;
        const auto& s = he.stroke();
        auto edge = objects.append_child("edge");
        hedge2cell[hi] = cell_id;
        edge.append_attribute("id").set_value(cell_id++);
        edge.append_attribute("frame") = "0";
        edge.append_attribute("startvertex").set_value(he.origin().index_);
        edge.append_attribute("endvertex").set_value(he.dest().index_);
        edge.append_attribute("color") = "rgba(0,0,0,1)";
        {
          auto ss = std::stringstream();
          // Enough precision to round-trip a double precision float.
          ss << std::setprecision(std::numeric_limits<double>::digits10 + 1);
          if (s.has_time()) {
            ss << "xywtdense(5"; // Type and ds (sampling) (nonsense ds value since
                                 // strokes aren't necessarily uniformly sampled).
            for (auto i = 0; i < s.size(); ++i) {
              ss << ' ' << s.x(i) << ',' << s.y(i) << ',' << s.width(i) << ','
                 << s.time(i);
            }
          } else {
            ss << "xywdense(5"; // Same comment.
            for (auto i = 0; i < s.size(); ++i) {
              ss << ' ' << s.x(i) << ',' << s.y(i) << ',' << s.width(i);
            }
          }
          ss << ")";
          const auto str = ss.str();
          edge.append_attribute("curve").set_value(str.c_str());
        }
      }
      for (size_t fi = 0; fi < graph.faces_.size(); ++fi) {
        if (fi == graph.boundary_face_)
          continue;
        auto face = objects.append_child("face");
        face.append_attribute("id").set_value(cell_id++);
        face.append_attribute("frame") = "0";
        face.append_attribute("color") = "rgba(180,180,180,0.5)";
        {
          auto ss = std::stringstream();
          ss << '[';
          // TODO: Support multiple cycles.
          const auto he = graph.hedge(graph.faces_[fi].cycles_[0]);
          auto it = he;
          do {
            if (it.forward()) {
              assert(hedge2cell.find(it.index_) != hedge2cell.end());
              ss << hedge2cell[it.index_] << '+';
            } else {
              assert(hedge2cell.find(it.twin().index_) != hedge2cell.end());
              ss << hedge2cell[it.twin().index_] << '-';
            }
            if (it.next() != he) {
              ss << ' ';
            }
            it = it.next();
          } while (it != he);
          ss << ']';
          const auto str = ss.str();
          face.append_attribute("cycles").set_value(str.c_str());
        }
      }
    }
  }
  const auto ok = doc.save_file(path.c_str(), "  ");
  if (!ok) {
    auto ss = std::stringstream();
    ss << "could not save file to " << path;
    throw io_error(ss.str());
  }
}

} // namespace sketching
