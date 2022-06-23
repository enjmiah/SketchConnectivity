#include "clipping.h"

#include <polyclipping/clipper.hpp>

namespace cl = ClipperLib;

namespace sketching {

namespace {

constexpr double inv_precision_factor = 1 / clip_precision_factor;

void clip_path_coord_matrix(const cl::Path& path, CoordMat& m) {
  for (std::size_t row = 0; row < path.size(); ++row) {
    m(row, 0) = double(path[row].X) * inv_precision_factor;
    m(row, 1) = double(path[row].Y) * inv_precision_factor;
  }
}

} // namespace

std::unique_ptr<cl::Paths> to_clip_paths(span<const CoordMat> polygons) {
  auto out = std::make_unique<cl::Paths>(polygons.size());
  for (std::size_t i = 0; i < polygons.size(); ++i) {
    const auto& coords = polygons[i];
    assert(coords.rows() == 0 || coords.maxCoeff() * clip_precision_factor <
                                   (Float)std::numeric_limits<cl::cInt>().max());
    const auto n = coords.rows();
    for (auto row = 0; row < n; ++row) {
      (*out)[i].emplace_back(cl::cInt(coords(row, 0) * clip_precision_factor),
                             cl::cInt(coords(row, 1) * clip_precision_factor));
    }
  }
  return out;
}

std::vector<CoordMat> from_clip_paths(const cl::Paths& paths) {
  std::vector<CoordMat> out;
  for (const auto& polygon : paths) {
    auto& coords = out.emplace_back(polygon.size(), 2);
    clip_path_coord_matrix(polygon, coords);
  }
  return out;
}

CoordMat from_clip_path(const ClipperLib::Path& path) {
  CoordMat out(path.size(), 2);
  clip_path_coord_matrix(path, out);
  return out;
}

std::unique_ptr<cl::Paths> boolean_union(const cl::Paths& polygons) {
  auto solution = std::make_unique<cl::Paths>();
  cl::Clipper c;
  c.AddPaths(polygons, cl::ptSubject, true);
  c.Execute(cl::ctUnion, *solution, cl::pftNonZero, cl::pftNonZero);
  return solution;
}

std::unique_ptr<cl::Paths> boolean_difference(const cl::Paths& a, const cl::Paths& b) {
  auto solution = std::make_unique<cl::Paths>();
  cl::Clipper c;
  c.AddPaths(a, cl::ptSubject, true);
  c.AddPaths(b, cl::ptClip, true);
  c.Execute(cl::ctDifference, *solution, cl::pftNonZero, cl::pftNonZero);
  return solution;
}

std::unique_ptr<cl::Paths> boolean_intersection(const cl::Paths& a, const cl::Paths& b) {
  auto solution = std::make_unique<cl::Paths>();
  cl::Clipper c;
  c.AddPaths(a, cl::ptSubject, true);
  c.AddPaths(b, cl::ptClip, true);
  c.Execute(cl::ctIntersection, *solution, cl::pftNonZero, cl::pftNonZero);
  return solution;
}

Float clip_area_scaled(const ClipperLib::Paths& paths) {
  Float area = 0.0;
  for (const auto& p : paths) {
    area += std::abs(cl::Area(p));
  }
  return area;
}

} // namespace sketching
