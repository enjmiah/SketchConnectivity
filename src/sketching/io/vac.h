#pragma once

#include "../types.h"

namespace sketching {

struct Stroke;

/**
 * Load a VPaint 1.7 Vector Animation Complex file at the given path.  Not all features
 * are supported.
 */
void load_vac(const std::string& path, std::vector<Stroke>& out_strokes);

/**
 * Load a VPaint 1.7 Vector Animation Complex file at the given path.  Not all features
 * are supported.
 */
inline std::vector<Stroke> load_vac(const std::string& path) {
  auto strokes = std::vector<Stroke>();
  load_vac(path, strokes);
  return strokes;
}

/**
 * Save a collection of strokes as a VPaint 1.7 Vector Animation Complex file.
 */
void save_vac(span<const Stroke> strokes, const std::string& path);

struct StrokeGraph;
void save_vac(const StrokeGraph&, const std::string& path);

} // namespace sketching
