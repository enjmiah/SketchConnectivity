#include "strokewise_measure.h"

#include "../closest.h"
#include "../fitting.h"

#ifdef HAS_GUROBI
#include <StrokeStrip/FittingEigenSparse.h>
#include <StrokeStrip/Parameterization.h>
#include <StrokeStrip/SketchInfo.h>
#include <StrokeStrip/StrokeOrientation.h>
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(disable : 4018 4389) // signed/unsigned mismatch
#pragma warning(disable : 4244 4267) // type conversions
#pragma warning(disable : 4456) // declaration hides previous local declaration
#pragma warning(disable : 4459) // declaration hides global declaration
#endif

namespace sketching {
namespace {
Float min_sub_stroke_length = 5;

#ifdef HAS_GUROBI
void common_parameterized_samples(const Stroke& s1, const Stroke& s2, Float accuracy,
                                  bool cut_strokes, Float sampling_px_step,
                                  size_t min_samples,
                                  std::unordered_map<size_t, size_t>& cut2original,
                                  std::vector<Cluster::XSec>& samples) {
  std::vector<const Stroke*> strokes({&s1, &s2});

  // 1. Build common parameterization
  Capture capture_cut;
  std::vector<StrokeMapping> out_mappings;
  strokes_to_capture(strokes, accuracy, cut_strokes, out_mappings, capture_cut);
  // Reassign stroke indices to the after cut strokes
  // Create a new capture is ugly but we know indexing is definitely correct
  Capture capture;
  capture.thickness = 1; // Note this parameter affects the fitting
  capture.sketchedPolylines.reserve(capture_cut.sketchedPolylines.size());
  for (auto& polyline : capture_cut.sketchedPolylines) {
    if (polyline.points.empty())
      continue;
    capture.sketchedPolylines.emplace_back(polyline);
    capture.sketchedPolylines.back().stroke_ind = capture.sketchedPolylines.size() - 1;
    // capture.sketchedPolylines.back().width = 1;
    capture.sketchedPolylines.back().width = polyline.width;
  }

  // Remove small sub strokes
  {
    Capture capture_cut_filter;
    for (auto& polyline : capture_cut.sketchedPolylines) {
      if (polyline.points.empty())
        continue;

      if (polyline.totalLen() < min_sub_stroke_length) {
        bool found_main_sub = false;
        for (auto& polyline2 : capture_cut.sketchedPolylines) {
          if (polyline.stroke_ind == polyline2.stroke_ind ||
              polyline.additional_ind != polyline2.additional_ind)
            continue;
          if (polyline.totalLen() < polyline2.totalLen()) {
            found_main_sub = true;
            break;
          }
        }
        if (found_main_sub)
          continue;
      }
      capture_cut_filter.sketchedPolylines.emplace_back(polyline);
      capture_cut_filter.sketchedPolylines.back().stroke_ind =
        capture_cut_filter.sketchedPolylines.size() - 1;
      // capture_cut_filter.sketchedPolylines.back().width = 1;
      capture_cut_filter.sketchedPolylines.back().width = polyline.width;
    }
    capture_cut = capture_cut_filter;
  }

  // Building the mapping between the original strokes and its largest sub-stroke (there
  // can be multiple of sub-strokes if we do cutting).
  std::unordered_map<size_t, std::vector<std::pair<size_t, Float>>> sub_lengths;
  for (const auto& polyline : capture.sketchedPolylines) {
    size_t orig_idx = (size_t)polyline.additional_ind;
    if (!sub_lengths.count(orig_idx))
      sub_lengths[orig_idx] = std::vector<std::pair<size_t, Float>>();
    sub_lengths[orig_idx].emplace_back(
      std::make_pair(polyline.stroke_ind, polyline.totalLen()));
  }
  for (auto& sort_sub : sub_lengths) {
    std::sort(
      sort_sub.second.begin(), sort_sub.second.end(),
      [](const std::pair<size_t, Float>& a, const std::pair<size_t, Float>& b) -> bool {
        return a.second > b.second;
      });
    cut2original[sort_sub.second.front().first] = sort_sub.first;
  }

  if (capture.sketchedPolylines.empty() || capture.sketchedPolylines.size() == 1 ||
      cut2original.size() < 2) {
    return;
  }

  glm::dvec2 center;
  Input input = from_capture(capture, center);
  const Context context(true);
  auto orientations = std::vector<int>();
  StrokeOrientation orientation(context);
  orientation.orient_strokes(&input);
  if (orientation.orientations.empty())
    return;
  orientation.flip_strokes(&input);

  orientations = std::move(orientation.orientations.begin()->second);
  Parameterization param(context);
  param.parameterize(&input);

  if (input.clusters[0].strokes.empty())
    return;

  // 2. Sample strips
  // Determine the sampling
  auto cluster = input.clusters[0];
  double max_u = cluster.max_u();

  sampling_px_step /= capture.thickness;
  Float sample_rate = std::min(sampling_px_step, cluster.max_u() / min_samples);
  auto num_xsecs = (size_t)std::ceil(max_u / sample_rate);
  num_xsecs = std::max(num_xsecs, min_samples);

  for (size_t n = 0; n < num_xsecs; ++n) {
    double t = double(n) / double(num_xsecs - 1);
    double u;
    u = t * max_u;

    auto xsec = param.xsec_at_u(cluster, u);
    if (xsec.points.size() > 0) {
      samples.push_back(xsec);
    }
  }

  // Transform the samples back to the original coord
  for (auto& xsec : samples) {
    for (auto& p : xsec.points) {
      p.point.x = p.point.x * capture.thickness + center.x;
      p.point.y = p.point.y * capture.thickness + center.y;
    }
  }
}
#endif

} // namespace

#ifdef HAS_GUROBI
Vec2 range_tangent(const Stroke& s, Float t, Float dt, bool point_outside) {
  Float t1, t2;
  if ((t - dt < 0) || (t + dt) > 1) {
    t2 = t;
    t1 = (t - dt < 0) ? t + dt : t - dt;
    if (!point_outside && t1 > t2) {
      t2 = t1;
      t1 = t;
    }
  } else {
    if (t > 0.5) {
      t2 = t + dt;
      t1 = t - dt;
    } else {
      t2 = t - dt;
      t1 = t + dt;
    }
    if (!point_outside && t1 > t2) {
      t2 = t + dt;
      t1 = t - dt;
    }
  }
  Vec2 end_p = s.pos_norm(t2);
  Vec2 endoff_p = s.pos_norm(t1);
  return (end_p - endoff_p).normalized();
}

Float pointwise_angular_difference(Cluster::XSecPoint p0, Cluster::XSecPoint p1) {
  Float angle =
    std::atan2(p0.tangent.y, p0.tangent.x) - atan2(p1.tangent.y, p1.tangent.x);
  angle = std::abs(angle);
  return angle;
}

Float pointwise_euclidean_difference(Cluster::XSecPoint p0, Cluster::XSecPoint p1) {
  Float dist = glm::distance(p0.point, p1.point);
  return dist;
}

Float bipointwise_euclidean_change(Cluster::XSecPoint p00, Cluster::XSecPoint p01,
                                   Cluster::XSecPoint p10, Cluster::XSecPoint p11) {
  Float dist0 = glm::distance(p00.point, p01.point);
  Float dist1 = glm::distance(p10.point, p11.point);
  return std::abs(dist0 - dist1) / dist0;
}

Float strokewise_measure(
  const Stroke& s1, const Stroke& s2,
  std::function<Float(Cluster::XSecPoint, Cluster::XSecPoint)> point_measure,
  Float accuracy, bool cut_strokes, Float sampling_px_step, size_t min_samples,
  Float* overlapping_ratio, bool find_max) {
  if (overlapping_ratio)
    *overlapping_ratio = -1;

  std::unordered_map<size_t, size_t> cut2original;
  std::vector<Cluster::XSec> samples;
  common_parameterized_samples(s1, s2, accuracy, cut_strokes, sampling_px_step,
                               min_samples, cut2original, samples);

  if (samples.empty())
    return std::numeric_limits<Float>::infinity();

  // 3. Integral
  // If cut, only use the largest component from the original stroke
  auto pick_samples_after_cut =
    [&cut2original](const Cluster::XSec& xsec) -> std::vector<size_t> {
    std::vector<size_t> picked_indices;
    picked_indices.reserve(2);
    for (size_t i = 0; i < xsec.points.size(); ++i) {
      if (cut2original.count(xsec.points[i].stroke_idx))
        picked_indices.emplace_back(i);
    }

    return picked_indices;
  };
  // Note that we are ignoring the half lengths/weights at the two endpoints of the
  // fitting curve and just using the even weighting.
  Float measure_sum = 0;
  int overlapping_sample_count = 0;
  for (const auto& xsec : samples) {
    std::vector<size_t> picked_indices = pick_samples_after_cut(xsec);

    // Not overlapping
    if (picked_indices.size() < 2)
      continue;

    size_t idx0 = picked_indices[0];
    size_t idx1 = picked_indices[1];

    // We call the difference comparison with the two stroke samples ordered based on the
    // stroke indices
    Float local_measure = point_measure(
      (xsec.points[idx0].stroke_idx < xsec.points[idx1].stroke_idx) ? xsec.points[idx0]
                                                                    : xsec.points[idx1],
      (xsec.points[idx0].stroke_idx < xsec.points[idx1].stroke_idx) ? xsec.points[idx1]
                                                                    : xsec.points[idx0]);
    if (!find_max) {
      measure_sum += local_measure;
    } else {
      measure_sum = std::max(measure_sum, local_measure);
    }
    overlapping_sample_count++;
  }

  if (!find_max && overlapping_sample_count > 0)
    measure_sum /= overlapping_sample_count;
  else if (!find_max) // There's no overlapping, use inf as an indicator.
    measure_sum = std::numeric_limits<Float>::infinity();

  if (overlapping_ratio)
    *overlapping_ratio = (Float)overlapping_sample_count / samples.size();

  return measure_sum;
}

Float strokewise_binary_measure(
  const Stroke& s1, const Stroke& s2,
  std::function<Float(Cluster::XSecPoint, Cluster::XSecPoint, Cluster::XSecPoint,
                      Cluster::XSecPoint)>
    point_measure,
  Float accuracy, bool cut_strokes, Float sampling_px_step, size_t min_samples,
  Float* overlapping_ratio, bool find_max) {
  if (overlapping_ratio)
    *overlapping_ratio = -1;

  std::unordered_map<size_t, size_t> cut2original;
  std::vector<Cluster::XSec> samples;
  common_parameterized_samples(s1, s2, accuracy, cut_strokes, sampling_px_step,
                               min_samples, cut2original, samples);

  if (samples.empty())
    return std::numeric_limits<Float>::infinity();

  // 3. Integral
  // If cut, only use the largest component from the original stroke
  auto pick_samples_after_cut =
    [&cut2original](const Cluster::XSec& xsec) -> std::vector<size_t> {
    std::vector<size_t> picked_indices;
    picked_indices.reserve(2);
    for (size_t i = 0; i < xsec.points.size(); ++i) {
      if (cut2original.count(xsec.points[i].stroke_idx))
        picked_indices.emplace_back(i);
    }

    return picked_indices;
  };
  // Note that we are ignoring the half lengths/weights at the two endpoints of the
  // fitting curve and just using the even weighting.
  Float measure_sum = 0;
  int overlapping_sample_count = 0;
  for (size_t i = 0; i + 1 < samples.size(); ++i) {
    const auto& xsec = samples[i];
    const auto& xsec_next = samples[i + 1];

    std::vector<size_t> picked_indices = pick_samples_after_cut(xsec);
    std::vector<size_t> picked_indices_next = pick_samples_after_cut(xsec_next);

    // Not overlapping
    if (picked_indices.size() < 2 || picked_indices_next.size() < 2)
      continue;

    size_t idx00 = picked_indices[0];
    size_t idx01 = picked_indices[1];
    size_t idx10 = picked_indices_next[0];
    size_t idx11 = picked_indices_next[1];

    // We call the difference comparison with the two stroke samples ordered based on the
    // stroke indices
    Float local_measure = bipointwise_euclidean_change(
      (xsec.points[idx00].stroke_idx < xsec.points[idx01].stroke_idx)
        ? xsec.points[idx00]
        : xsec.points[idx01],
      (xsec.points[idx00].stroke_idx < xsec.points[idx01].stroke_idx)
        ? xsec.points[idx01]
        : xsec.points[idx00],
      (xsec_next.points[idx10].stroke_idx < xsec_next.points[idx11].stroke_idx)
        ? xsec_next.points[idx10]
        : xsec_next.points[idx11],
      (xsec_next.points[idx10].stroke_idx < xsec_next.points[idx11].stroke_idx)
        ? xsec_next.points[idx11]
        : xsec_next.points[idx10]);

    if (!find_max) {
      measure_sum += local_measure;
    } else {
      measure_sum = std::max(measure_sum, local_measure);
    }
    overlapping_sample_count++;
  }

  if (!find_max && overlapping_sample_count > 0)
    measure_sum /= overlapping_sample_count;
  else if (!find_max) // There's no overlapping, use inf as an indicator.
    measure_sum = std::numeric_limits<Float>::infinity();

  if (overlapping_ratio)
    *overlapping_ratio = (Float)overlapping_sample_count / samples.size();

  return measure_sum;
}

Stroke fit_stroke(Input& input, const Capture& capture, const glm::dvec2& center,
                  const Context& context, Float accuracy, bool use_default_sampling) {
  FittingEigenSparse fitting(context);
  // Determine the sampling rate
  double sampling_rate = 0.5;
  double sampling_rate_px = 0.5;
  double k_weight = 0.1;

  if (use_default_sampling) {
    sampling_rate = 0.2;
    sampling_rate_px = 0.2;
    k_weight = 1;
  }

  std::pair<double, double> u_range(std::numeric_limits<double>::infinity(),
                                    -std::numeric_limits<double>::infinity());
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
    std::vector<double> du_dps;
    for (size_t i = 0; i < input.clusters[0].xsecs.size(); ++i) {
      if (i + 1 < input.clusters[0].xsecs.size() &&
          !input.clusters[0].xsecs[i + 1].points.empty()) {
        glm::dvec2 cur = get_xsec_avg_pos(input.clusters[0].xsecs[i]);
        glm::dvec2 next = get_xsec_avg_pos(input.clusters[0].xsecs[i + 1]);
        dp = glm::distance(cur, next);
        du = input.clusters[0].xsecs[i + 1].u - input.clusters[0].xsecs[i].u;
        du_dps.emplace_back(std::abs(du / dp));
      }
      u_range.first = std::min(u_range.first, input.clusters[0].xsecs[i].u);
      u_range.second = std::max(u_range.second, input.clusters[0].xsecs[i].u);
    }
    assert(!du_dps.empty());
    size_t median_n = du_dps.size() / 2;
    std::nth_element(du_dps.begin(), du_dps.begin() + median_n, du_dps.end());

    double med_du_dp = du_dps[median_n];
    // std::cout << "sampling_rate_px: " << sampling_rate_px << std::endl;
    // std::cout << "med_du_dp: " << med_du_dp << std::endl;
    sampling_rate = std::abs(sampling_rate_px * med_du_dp);
  }

  // std::cout << "Sampling rate: " << sampling_rate << std::endl;
  auto out = fitting.fit(&input, true, sampling_rate, k_weight);

  auto to_stroke = [&capture, &center, &accuracy](
                     const std::map<int, std::unique_ptr<std::vector<glm::dvec2>[]>>& out,
                     Stroke& strokefit) {
    strokefit.clear();
    const auto& arrays = out.begin()->second;
    const auto& fitted = arrays[capture.sketchedPolylines.size()];

    // Dedup
    std::vector<glm::dvec2> dedup_fitted;
    dedup_fitted.reserve(fitted.size());
    {
      for (size_t i = 0; i < fitted.size(); ++i) {
        if (i > 0) {
          size_t j = i - 1;
          if (glm::distance(fitted[i], fitted[j]) <
              std::numeric_limits<double>::epsilon())
            continue;
        }

        dedup_fitted.emplace_back(fitted[i]);
      }
    }

    strokefit = Stroke(dedup_fitted.size(), false);

    // Copy points.
    for (size_t i = 0; i < dedup_fitted.size(); ++i) {
      const auto& p = dedup_fitted[i];
      strokefit.x(i) = p.x * capture.thickness + center.x;
      strokefit.y(i) = p.y * capture.thickness + center.y;
      strokefit.width(i) =
        accuracy * capture.thickness; // TODO: Transfer thickness better.
    }
    strokefit.compute_arclengths();
  };

  Stroke strokefit;
  to_stroke(out, strokefit);

  auto get_out_size =
    [&capture](
      const std::map<int, std::unique_ptr<std::vector<glm::dvec2>[]>>& out) -> size_t {
    return out.begin()->second[capture.sketchedPolylines.size()].size();
  };

  size_t max_num_samples = 1000;
  if (strokefit.size() == 0) {
    sampling_rate = 0.2;
    auto out2 = fitting.fit(&input, true, sampling_rate, k_weight);

    if (get_out_size(out2) > get_out_size(out)) {
      to_stroke(out2, strokefit);
    }
  } else {
    // In case the med_du_dp is >> 1 which may cause undersampling
    size_t min_num_samples = strokefit.length() / sampling_rate_px;

    // std::cout << "Before fit size: " << strokefit.size() << std::endl;
    // std::cout << "min_num_samples: " << min_num_samples << " <= " << strokefit.length()
    // << std::endl;
    if (min_num_samples < max_num_samples && get_out_size(out) < min_num_samples) {
      sampling_rate = (u_range.second - u_range.first) / min_num_samples * 2;
      // std::cout << "Sampling rate2: " << sampling_rate << std::endl;
      auto out2 = fitting.fit(&input, true, sampling_rate, k_weight);

      if (get_out_size(out2) > get_out_size(out)) {
        to_stroke(out2, strokefit);
      }
    } else if (get_out_size(out) > max_num_samples) {
      Float downsample_ratio = (Float)get_out_size(out) / max_num_samples;
      sampling_rate *= downsample_ratio;
      // std::cout << "Sampling rate2: " << sampling_rate << std::endl;
      auto out2 = fitting.fit(&input, true, sampling_rate, k_weight);

      if (get_out_size(out2) < get_out_size(out)) {
        to_stroke(out2, strokefit);
      }
    }
  }

  return strokefit;
}

Float strokewise_fit_measure(
  const Stroke& s1, const Stroke& s2,
  std::function<Float(Cluster::XSecPoint, Cluster::XSecPoint)> point_measure,
  Float accuracy, bool cut_strokes, Float sampling_px_step, size_t min_samples,
  bool find_max) {
  std::vector<const Stroke*> strokes({&s1, &s2});

  // 1. Build common parameterization
  Capture capture_cut;
  std::vector<StrokeMapping> out_mappings;
  strokes_to_capture(strokes, accuracy, cut_strokes, out_mappings, capture_cut);
  // Reassign stroke indices to the after cut strokes
  // Create a new capture is ugly but we know indexing is definitely correct
  Capture capture;
  capture.thickness = 1; // Note this parameter affects the fitting
  capture.sketchedPolylines.reserve(capture_cut.sketchedPolylines.size());
  for (auto& polyline : capture_cut.sketchedPolylines) {
    if (polyline.points.empty())
      continue;
    capture.sketchedPolylines.emplace_back(polyline);
    capture.sketchedPolylines.back().stroke_ind = capture.sketchedPolylines.size() - 1;
    capture.sketchedPolylines.back().width = 1;
  }

  // Building the mapping between the original strokes and its largest sub-stroke (there
  // can be multiple of sub-strokes if we do cutting).
  std::unordered_map<size_t, size_t> cut2original;
  std::unordered_map<size_t, std::vector<std::pair<size_t, Float>>> sub_lengths;
  for (const auto& polyline : capture.sketchedPolylines) {
    size_t orig_idx = (size_t)polyline.additional_ind;
    if (!sub_lengths.count(orig_idx))
      sub_lengths[orig_idx] = std::vector<std::pair<size_t, Float>>();
    sub_lengths[orig_idx].emplace_back(
      std::make_pair(polyline.stroke_ind, polyline.totalLen()));
  }
  for (auto& sort_sub : sub_lengths) {
    std::sort(
      sort_sub.second.begin(), sort_sub.second.end(),
      [](const std::pair<size_t, Float>& a, const std::pair<size_t, Float>& b) -> bool {
        return a.second > b.second;
      });
    cut2original[sort_sub.second.front().first] = sort_sub.first;
  }

  if (capture.sketchedPolylines.empty() || capture.sketchedPolylines.size() == 1 ||
      cut2original.size() < 2) {
    auto ss = std::stringstream();
    throw std::runtime_error("no strokes in cluster were long enough");
  }

  glm::dvec2 center;
  Input input = from_capture(capture, center);
  const Context context(true);
  auto orientations = std::vector<int>();
  StrokeOrientation orientation(context);
  orientation.orient_strokes(&input);
  if (orientation.orientations.empty())
    return std::numeric_limits<Float>::infinity();
  orientation.flip_strokes(&input);

  orientations = std::move(orientation.orientations.begin()->second);
  Parameterization param(context);
  param.parameterize(&input);

  if (input.clusters[0].strokes.empty())
    return std::numeric_limits<Float>::infinity();

  // 2. Sample strips
  // Determine the sampling
  auto cluster = input.clusters[0];
  double max_u = cluster.max_u();

  sampling_px_step /= capture.thickness;
  Float sample_rate = std::min(sampling_px_step, cluster.max_u() / min_samples);
  auto num_xsecs = (size_t)std::ceil(max_u / sample_rate);
  num_xsecs = std::max(num_xsecs, min_samples);

  std::vector<Cluster::XSec> samples;
  for (size_t n = 0; n < num_xsecs; ++n) {
    double t = double(n) / double(num_xsecs - 1);
    double u;
    u = t * max_u;

    auto xsec = param.xsec_at_u(cluster, u);
    if (xsec.points.size() > 0) {
      samples.push_back(xsec);
    }
  }

  // 2.5 Fit the curve for the fit-input comparison
  Stroke strokefit = fit_stroke(input, capture, center, context, accuracy);

  if (strokefit.size() == 0)
    return std::numeric_limits<Float>::infinity();

  // 3. Integral
  // If cut, only use the largest component from the original stroke
  auto pick_samples_after_cut =
    [&cut2original](const Cluster::XSec& xsec) -> std::vector<size_t> {
    std::vector<size_t> picked_indices;
    picked_indices.reserve(2);
    for (size_t i = 0; i < xsec.points.size(); ++i) {
      if (cut2original.count(xsec.points[i].stroke_idx))
        picked_indices.emplace_back(i);
    }

    return picked_indices;
  };
  // Note that we are ignoring the half lengths/weights at the two endpoints of the
  // fitting curve and just using the even weighting.
  Float measure_sum = 0;
  int overlapping_sample_count = 0;

  auto sample_on_fit = [](const Stroke& strokefit, glm::dvec2 p) -> Cluster::XSecPoint {
    Float tan_dt = std::min(0.05, strokefit.length() / 1);
    Vec2 out_proj;
    Float out_s;
    closest_point(strokefit, Vec2(p.x, p.y), out_proj, out_s);

    Vec2 fit_pos = strokefit.pos(out_s);

    // Determine the sign of the v value
    Vec2 fit_tangent =
      range_tangent(strokefit, out_s / strokefit.length(), tan_dt, false);

    Cluster::XSecPoint fit_p;
    fit_p.point = glm::dvec2(fit_pos.x(), fit_pos.y());
    fit_p.tangent = glm::dvec2(fit_tangent.x(), fit_tangent.y());
    return fit_p;
  };

  // Transform the samples back to the original coord
  for (auto& xsec : samples) {
    for (auto& p : xsec.points) {
      p.point.x = p.point.x * capture.thickness + center.x;
      p.point.y = p.point.y * capture.thickness + center.y;
    }
  }

  for (const auto& xsec : samples) {
    std::vector<size_t> picked_indices = pick_samples_after_cut(xsec);

    if (picked_indices.empty())
      continue;

    glm::dvec2 center_p(0, 0);
    for (auto idx : picked_indices) {
      center_p = center_p + xsec.points[idx].point;
    }
    center_p.x /= picked_indices.size();
    center_p.y /= picked_indices.size();

    Float local_measure = -std::numeric_limits<Float>::infinity();
    auto fit_p = sample_on_fit(strokefit, center_p);
    for (auto idx : picked_indices) {
      // We call the difference comparison with the two stroke samples ordered based on
      // the stroke indices
      Float local_measure1 = point_measure(xsec.points[idx], fit_p);
      local_measure = std::max(local_measure1, local_measure);
    }

    if (!find_max) {
      measure_sum += local_measure;
    } else {
      measure_sum = std::max(measure_sum, local_measure);
    }
    overlapping_sample_count++;
  }

  if (!find_max && overlapping_sample_count > 0)
    measure_sum /= overlapping_sample_count;
  else if (!find_max) // There's no overlapping, use inf as an indicator.
    measure_sum = std::numeric_limits<Float>::epsilon();

  return measure_sum;
}

#endif
} // namespace sketching
