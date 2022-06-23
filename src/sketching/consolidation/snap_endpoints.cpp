#include "snap_endpoints.h"

#include "../deform.h"
#include "../fitting.h"
#include "../sketching.h"

#include <spdlog/spdlog.h>

#ifdef HAS_GUROBI
#include <StrokeStrip/FittingEigenSparse.h>
#include <StrokeStrip/Parameterization.h>
#include <StrokeStrip/SketchInfo.h>
#include <StrokeStrip/StrokeOrientation.h>
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(disable : 4018) // signed/unsigned mismatch
#pragma warning(disable : 4244 4267) // type conversions
#endif

namespace sketching {

namespace {

constexpr Float accuracy = 1;
constexpr Float division_angle_threshold = 30. / 180 * M_PI;
constexpr Float max_end_length_ratio = 4;
constexpr Float max_stroke_length_ratio = 0.4;

} // namespace

size_t determine_deformation_substroke(const Stroke& stroke2, const bool head) {
  // Find the sub stroke we want to run fitting and snapping on
  Index deformation_size = 1;

  const auto end_arclen = (head ? 0.0 : stroke2.length());
  Float length_threshold = std::min(max_end_length_ratio * stroke2.pen_width(),
                                    stroke2.length() * max_stroke_length_ratio);
#define LOOP_BODY                                                                        \
  do {                                                                                   \
    Index prev = i - 1;                                                                  \
    Index next = i + 1;                                                                  \
    deformation_size = i;                                                                \
                                                                                         \
    /* We've reached target length. */                                                   \
    Float s_len = std::abs(stroke2.arclength(i) - end_arclen);                           \
    if (s_len >= length_threshold) {                                                     \
      goto done;                                                                         \
    }                                                                                    \
                                                                                         \
    /* Check the local tangent change at i. */                                           \
    Vec2 t1 = (stroke2.xy(i) - stroke2.xy(prev)).normalized();                           \
    Vec2 t2 = (stroke2.xy(next) - stroke2.xy(i)).normalized();                           \
    Float dot_prod = t1.dot(t2);                                                         \
                                                                                         \
    /* TODO: Unify this angle threshold based on other angle thresholds. */              \
    if (dot_prod < 0 ||                                                                  \
        std::acos(std::clamp(dot_prod, 0., 1.)) > division_angle_threshold) {            \
      goto done;                                                                         \
    }                                                                                    \
  } while (0)

  if (head) {
    for (Index i = 1; i + 1 < stroke2.size(); ++i) {
      LOOP_BODY;
    }
  } else {
    for (Index i = stroke2.size() - 2; i > 0; --i) {
      LOOP_BODY;
    }
  }

done:
  assert(0 < deformation_size && deformation_size < stroke2.size());
  return (size_t)deformation_size;
#undef LOOP_BODY
}

#ifdef HAS_GUROBI

bool snap_endpoints(const Stroke& stroke, const Vec2 head_pos, const Vec2 tail_pos,
                    Stroke& stroke_snapped, bool use_default_sampling) {
  assert(&stroke != &stroke_snapped);

  const std::vector<Vec2> endpoints{head_pos, tail_pos};
  if (stroke.size() <= 1) {
    stroke_snapped = stroke.clone();
    if (stroke.size() > 0) {
      stroke_snapped.x(0) = endpoints.front().x();
      stroke_snapped.y(0) = endpoints.front().y();
    }
    return true;
  }

  // 1. Build common parameterization
  Capture capture_cut;
  std::vector<StrokeMapping> out_mappings;
  std::vector<const Stroke*> strokes{&stroke};
  strokes_to_capture(strokes, accuracy, false, out_mappings, capture_cut);

  // Reassign stroke indices to the after cut strokes
  // Create a new capture is ugly but we know indexing is definitely correct
  Capture capture;
  if (!use_default_sampling)
    capture.thickness = 1; // Note this parameter affects the fitting
  capture.sketchedPolylines.reserve(capture_cut.sketchedPolylines.size());
  for (auto& polyline : capture_cut.sketchedPolylines) {
    if (polyline.points.empty())
      continue;
    capture.sketchedPolylines.emplace_back(polyline);
    capture.sketchedPolylines.back().stroke_ind = capture.sketchedPolylines.size() - 1;
    if (!use_default_sampling)
      capture.sketchedPolylines.back().width = 1;
  }

  glm::dvec2 center;
  Input input = from_capture(capture, center);
  const Context context(false);
  Parameterization param(context);
  param.parameterize(&input);

  if (input.clusters[0].strokes.empty())
    return true;

  // 2. Sample strips
  // Determine the sampling
  // Currently using the fitting to better determine the sampling rate (since the input
  // rate is in px). Though this fitting may not be necessary for efficiency concerns.
  FittingEigenSparse fitting(context);
  // Determine the sampling rate
  double sampling_rate = 0.5;
  double sampling_rate_px = 0.5;

  if (use_default_sampling) {
    sampling_rate = 0.2;
    sampling_rate_px = 0.2;
  }

  {
    auto get_xsec_avg_pos = [](const Cluster::XSec& xsec) -> glm::dvec2 {
      glm::dvec2 pos(0, 0);
      for (const auto& p : xsec.points) {
        pos += p.point;
      }
      pos.x /= xsec.points.size();
      pos.y /= xsec.points.size();
      return pos;
    };
    double dp = -1;
    double du = -1;
    for (size_t i = 0; i + 1 < input.clusters[0].xsecs.size(); ++i) {
      if (!input.clusters[0].xsecs[i + 1].points.empty()) {
        glm::dvec2 cur = get_xsec_avg_pos(input.clusters[0].xsecs[i]);
        glm::dvec2 next = get_xsec_avg_pos(input.clusters[0].xsecs[i + 1]);
        dp = glm::distance(cur, next);
        du = input.clusters[0].xsecs[i + 1].u - input.clusters[0].xsecs[i].u;
      }
    }
    sampling_rate = std::abs(sampling_rate_px / dp * du);
  }

  std::vector<glm::dvec2> glm_endpoints;
  glm_endpoints.reserve(endpoints.size());
  for (const auto& p : endpoints) {
    glm_endpoints.emplace_back(glm::dvec2((p.x() - center.x) / capture.thickness,
                                          (p.y() - center.y) / capture.thickness));
  }

  std::map<int, std::unique_ptr<std::vector<glm::dvec2>[]>> out;
  if (!use_default_sampling)
    out = fitting.fit(&input, false, sampling_rate, 0, glm_endpoints);
  else
    out = fitting.fit(&input, false, -1, 0, glm_endpoints);

  const auto& arrays = out.begin()->second;
  const auto& fitted = arrays[capture.sketchedPolylines.size()];

  // Dedup
  std::vector<glm::dvec2> dedup_fitted;
  dedup_fitted.reserve(fitted.size());
  {
    for (size_t i = 0; i < fitted.size(); ++i) {
      if (i > 0) {
        size_t j = i - 1;
        if (glm::distance(fitted[i], fitted[j]) < std::numeric_limits<double>::epsilon())
          continue;
      }

      dedup_fitted.emplace_back(fitted[i]);
    }
  }

  if (dedup_fitted.empty())
    return false;

  Stroke stroke_snapped_orig(dedup_fitted.size(), false);

  // Copy points.
  for (size_t i = 0; i < dedup_fitted.size(); ++i) {
    const auto& p = dedup_fitted[i];
    stroke_snapped_orig.x(i) = p.x * capture.thickness + center.x;
    stroke_snapped_orig.y(i) = p.y * capture.thickness + center.y;
  }
  stroke_snapped_orig.compute_arclengths();

  // Create the output with a sampling extractly copied based on the input
  stroke_snapped = Stroke(stroke.size(), false);
  for (size_t i = 0; i < stroke.size(); ++i) {
    stroke_snapped.width(i) = stroke.width(i);
    auto snapped_pos = stroke_snapped_orig.pos_norm((Float)i / (stroke.size() - 1));
    stroke_snapped.x(i) = snapped_pos.x();
    stroke_snapped.y(i) = snapped_pos.y();
  }

  bool succeeded =
    ((stroke_snapped.xy(0) - endpoints[0]).norm() < 1e-2) ||
    ((stroke_snapped.xy(stroke_snapped.size() - 1) - endpoints[1]).norm() < 1e-2);

  stroke_snapped.x(0) = endpoints[0].x();
  stroke_snapped.y(0) = endpoints[0].y();
  stroke_snapped.x(stroke_snapped.size() - 1) = endpoints[1].x();
  stroke_snapped.y(stroke_snapped.size() - 1) = endpoints[1].y();

  return succeeded;
}

bool snap_endpoints_divided(Stroke& stroke, const Vec2 head_pos, const Vec2 tail_pos,
                            bool use_default_sampling) {
  int deformation_size = -1;
  bool forward;
  return snap_endpoints_adaptive(stroke, head_pos, tail_pos, use_default_sampling,
                                 deformation_size, forward);
}

bool snap_endpoints_adaptive(Stroke& stroke, Vec2 head_pos, Vec2 tail_pos,
                             bool use_default_sampling, int& deformation_size,
                             bool& forward) {
  assert(deformation_size < stroke.size());

  forward = true;
  assert(stroke.size() > 1);
  if (stroke.size() <= 2 ||
      (stroke.xy(0).isApprox(head_pos) && stroke.xy(Back).isApprox(tail_pos))) {
    stroke.x(0) = head_pos.x();
    stroke.y(0) = head_pos.y();
    stroke.x(Back) = tail_pos.x();
    stroke.y(Back) = tail_pos.y();
    stroke.invalidate_arclengths();
    deformation_size = stroke.size() - 1;
    return true;
  }

  Vec2 head_pos2 = head_pos;
  Vec2 tail_pos2 = tail_pos;
  Stroke stroke2 = stroke.clone();
  // Flip if necessary so we always have the tail fixed.
  if ((head_pos - stroke2.xy(0)).norm() < std::numeric_limits<Float>::epsilon()) {
    stroke2.reverse();
    head_pos2 = tail_pos;
    tail_pos2 = head_pos;
    forward = false;
  }

  if (deformation_size < 0) {
    stroke2.ensure_arclengths();
    deformation_size = (int)determine_deformation_substroke(stroke2, true);
  }

  Stroke stroke_divided;
  assert(deformation_size >= 0);
  stroke_divided.reserve((size_t)deformation_size + 1);
  for (size_t i = 0; i <= deformation_size; ++i) {
    stroke_divided.push_back(stroke2.x(i), stroke2.y(i), stroke2.width(i));
  }
  Stroke stroke_divided_snapped;
  if (!snap_endpoints(stroke_divided, head_pos2, stroke2.xy(deformation_size),
                      stroke_divided_snapped, use_default_sampling)) {
    return false;
  }

  // Since snap_endpoints maintains the sampling size between the input and the output, we
  // can directly modify the positions and widths
  for (size_t i = 0; i <= deformation_size; ++i) {
    stroke2.x(i) = stroke_divided_snapped.x(i);
    stroke2.y(i) = stroke_divided_snapped.y(i);
    stroke2.width(i) = stroke_divided_snapped.width(i);
  }

  stroke = stroke2.clone();
  if (!forward)
    stroke.reverse();
  return true;
}

#endif

} // namespace sketching
