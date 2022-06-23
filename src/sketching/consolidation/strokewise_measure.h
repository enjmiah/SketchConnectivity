#pragma once

#include "../types.h"

#ifdef HAS_GUROBI
#include <StrokeStrip/Cluster.h>
#include <StrokeStrip/Context.h>
#include <StrokeStrip/SketchInfo.h>
#endif

namespace sketching {
struct Stroke;

#ifdef HAS_GUROBI
Vec2 range_tangent(const Stroke& s, Float t, Float dt, bool point_outside = true);

Float pointwise_angular_difference(Cluster::XSecPoint, Cluster::XSecPoint);
Float pointwise_euclidean_difference(Cluster::XSecPoint, Cluster::XSecPoint);

Float bipointwise_euclidean_change(Cluster::XSecPoint, Cluster::XSecPoint,
                                   Cluster::XSecPoint, Cluster::XSecPoint);

Stroke fit_stroke(Input& input, const Capture& capture, const glm::dvec2& center,
                  const Context& context, Float accuracy,
                  bool use_default_sampling = false);

Float strokewise_measure(
  const Stroke& s1, const Stroke& s2,
  std::function<Float(Cluster::XSecPoint, Cluster::XSecPoint)> point_measure,
  Float accuracy, bool cut_strokes, Float sampling_px_step, size_t min_samples,
  Float* overlapping_ratio = nullptr, bool find_max = false);

Float strokewise_binary_measure(
  const Stroke& s1, const Stroke& s2,
  std::function<Float(Cluster::XSecPoint, Cluster::XSecPoint, Cluster::XSecPoint,
                      Cluster::XSecPoint)>
    point_measure,
  Float accuracy, bool cut_strokes, Float sampling_px_step, size_t min_samples,
  Float* overlapping_ratio = nullptr, bool find_max = false);

Float strokewise_fit_measure(
  const Stroke& s1, const Stroke& s2,
  std::function<Float(Cluster::XSecPoint, Cluster::XSecPoint)> point_measure,
  Float accuracy, bool cut_strokes, Float sampling_px_step, size_t min_samples,
  bool find_max = false);
#endif
} // namespace sketching
