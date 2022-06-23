#include "consolidate.h"

#include "../deform.h"
#include "../fitting.h"
#include "../intersect.h"
#include "../is_face_collapsible.h"
#include "../resample.h"
#include "../stroke_graph.h"
#include "strokewise_measure.h"

#ifdef HAS_GUROBI
#include <StrokeStrip/FittingEigenSparse.h>
#include <StrokeStrip/Parameterization.h>
#include <StrokeStrip/SketchInfo.h>
#include <StrokeStrip/StrokeOrientation.h>
#endif

#include <fstream>
#include <vector>

#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(disable : 4018 4389) // signed/unsigned mismatch
#pragma warning(disable : 4244 4267) // type conversions
#pragma warning(disable : 4456) // declaration hides previous local declaration
#pragma warning(disable : 4459) // declaration hides global declaration
#endif

namespace sketching {

#ifdef HAS_GUROBI

namespace {

double continuity_angle_threshold = 30. / 180 * M_PI;
Float sampling_px_step = 2;
size_t min_samples = 5;

Float min_sub_stroke_length = 5;

void get_hausdorff_width(const std::vector<const Stroke*>& strokes, const Stroke& fit,
                         Float& hausdorff, Float& width) {
  hausdorff = -1;
  width = -1;
  {
    for (auto s : strokes) {
      for (size_t i = 0; i < s->size(); ++i) {
        Vec2 out_proj;
        Float out_s;
        Float d = closest_point(fit, s->xy(i), out_proj, out_s);
        if (d > hausdorff) {
          hausdorff = d;
          width = s->width(i);
        }
      }
    }
  }
}

void get_hausdorff_width(const Stroke& fit, const std::vector<const Stroke*>& strokes,
                         Float& hausdorff, Float& width) {
  hausdorff = -1;
  width = -1;
  {
    for (size_t i = 0; i < fit.size(); ++i) {
      Float group_d = std::numeric_limits<Float>::infinity();
      Float fit_width = fit.width(i);
      for (auto s : strokes) {
        Vec2 out_proj;
        Float out_s;
        Float d = closest_point(*s, fit.xy(i), out_proj, out_s);
        if (d < group_d) {
          group_d = d;
          fit_width = fit.width(i);
        }
      }

      if (group_d > hausdorff) {
        hausdorff = group_d;
        width = fit_width;
      }
    }
  }
}

void adaptive_fit(const std::vector<const Stroke*>& strokes, Float accuracy,
                  bool cut_strokes, Stroke& fit) {
  std::vector<Stroke> fit_strokes;
  std::map<Float, size_t> fit_strokes_sort;
  // Strokes => Fitting hausdorff

  fit_strokes.emplace_back();
  fit_width_cluster(strokes, accuracy, cut_strokes, fit_strokes.back());

  Float hausdorff, width;
  get_hausdorff_width(strokes, fit_strokes.back(), hausdorff, width);
  width = std::max(strokes.front()->pen_width(), width);
  fit_strokes_sort[hausdorff] = fit_strokes.size() - 1;

  if (hausdorff > width) {
    SPDLOG_INFO("\tRefit: {} vs {}", hausdorff, width);
    fit_strokes.emplace_back();
    fit_width_cluster(strokes, accuracy, cut_strokes, fit_strokes.back(), true);

    get_hausdorff_width(strokes, fit_strokes.back(), hausdorff, width);
    width = std::max(strokes.front()->pen_width(), width);
    fit_strokes_sort[hausdorff] = fit_strokes.size() - 1;
  }

  if (hausdorff > width) {
    SPDLOG_INFO("\tRefit without cut: {} vs {}", hausdorff, width);
    fit_strokes.emplace_back();
    fit_width_cluster(strokes, accuracy, false, fit_strokes.back(), true);

    get_hausdorff_width(strokes, fit_strokes.back(), hausdorff, width);
    width = std::max(strokes.front()->pen_width(), width);
    fit_strokes_sort[hausdorff] = fit_strokes.size() - 1;
  }

  fit = fit_strokes[fit_strokes_sort.begin()->second].clone();
  SPDLOG_INFO("fit_width_cluster done: {}", fit_strokes_sort.begin()->first);
}

} // namespace

void fit_width_cluster(const std::vector<const Stroke*>& strokes, Float accuracy,
                       bool cut_strokes, Stroke& strokefit, bool use_default_sampling) {
  if (strokes.size() <= 1) {
    if (strokes.size() == 1) {
      strokefit = strokes[0]->clone();
    }
    return;
  }

  // 1. Build common parameterization
  Capture capture_cut;
  std::vector<StrokeMapping> out_mappings;
  strokes_to_capture(strokes, accuracy, cut_strokes, out_mappings, capture_cut);

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
      capture_cut_filter.sketchedPolylines.back().width = polyline.width;
    }
    capture_cut = capture_cut_filter;
  }

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
    capture.sketchedPolylines.back().width = polyline.width;
  }

  // Building the mapping between the original strokes and all its sub-stroke (there can
  // be multiple of sub-strokes if we do cutting).
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
    for (auto cut_idx : sort_sub.second)
      cut2original[cut_idx.first] = sort_sub.first;
  }

  if (capture.sketchedPolylines.size() <= 1 || cut2original.size() <= 1) {
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
  // Currently using the fitting to better determine the sampling rate (since the input
  // rate is in px). Though this fitting may not be necessary for efficiency concerns.
  strokefit = fit_stroke(input, capture, center, context, accuracy, use_default_sampling);
  // std::cout << "Fit size: " << strokefit.size() << std::endl;
  auto cluster = input.clusters[0];
  double max_u = cluster.max_u();

  // Keep the sampling as the fitting
  std::vector<Cluster::XSec> samples;
  samples.reserve(strokefit.size());
  double fit_length = strokefit.length();
  for (size_t n = 0; n < strokefit.size(); ++n) {
    double t = strokefit.arclength(n) / fit_length;
    double u;
    u = t * max_u;

    auto xsec = param.xsec_at_u(cluster, u);
    samples.push_back(xsec);
  }

  // Transform the samples back to the original coord
  for (auto& xsec : samples) {
    for (auto& p : xsec.points) {
      p.point.x = p.point.x * capture.thickness + center.x;
      p.point.y = p.point.y * capture.thickness + center.y;
    }
  }

  // 3. Integral
  // If cut, only use the largest component from the original stroke
  auto pick_samples_after_cut =
    [&cut2original](const Cluster::XSec& xsec) -> std::vector<size_t> {
    std::vector<size_t> picked_indices;
    for (size_t i = 0; i < xsec.points.size(); ++i) {
      if (cut2original.count(xsec.points[i].stroke_idx))
        picked_indices.emplace_back(i);
    }

    return picked_indices;
  };
  Float tan_dt = std::min(0.05, strokefit.length() / 5);
  auto xsec_width = [&strokefit, &cut2original, &strokes, &pick_samples_after_cut,
                     &fit_length,
                     &tan_dt](const Cluster::XSec& xsec, const Float pos_t) -> Float {
    std::vector<size_t> picked_indices = pick_samples_after_cut(xsec);
    // SPDLOG_INFO("i: {}; #samples: {}", i, picked_indices.size());
    std::vector<std::pair<Float, Float>> intervals;
    auto add_interval = [](const std::pair<Float, Float> interval,
                           std::vector<std::pair<Float, Float>>& intervals) {
      for (auto& i : intervals) {
        if ((i.first >= interval.first && i.first <= interval.second) ||
            (interval.first >= i.first && interval.first <= i.second)) {
          i = std::make_pair(std::min(i.first, interval.first),
                             std::max(i.second, interval.second));
          return;
        }
      }
      intervals.emplace_back(interval);
    };
    for (auto sample_idx : picked_indices) {
      // Convert to the original stroke before cutting
      auto s_idx = cut2original[xsec.points[sample_idx].stroke_idx];
      const auto& orig_s = *strokes[s_idx];
      Vec2 sample_pos(xsec.points[sample_idx].point.x, xsec.points[sample_idx].point.y);
      Vec2 out_proj;
      Float out_s;
      closest_point(orig_s, sample_pos, out_proj, out_s);

      Vec2 orig_pos = orig_s.pos(out_s);
      Vec2 fit_pos = strokefit.pos_norm(pos_t);

      Float local_w = strokes[s_idx]->width_at(out_s) * 0.5;

      // Determine the sign of the v value
      Vec2 fit_tangent = range_tangent(strokefit, pos_t, tan_dt, false);
      Float det = counter_clockwise(sample_pos, fit_pos, fit_pos + fit_tangent);
      Float v = ((det > 0) ? 1 : -1) * (sample_pos - fit_pos).norm();

      std::pair<Float, Float> interval(v - local_w, v + local_w);
      add_interval(interval, intervals);
    }

    // Merge the ones in the set until no change
    bool updated = false;
    std::vector<std::pair<Float, Float>> intervals_updated;
    do {
      updated = false;

      for (const auto& i : intervals) {
        add_interval(i, intervals_updated);
      }

      if (intervals_updated.size() != intervals.size()) {
        updated = true;
        intervals = intervals_updated;
        intervals_updated.clear();
      }
    } while (updated);

    // Since we only have a single width parameter per position, take the average of the
    // upper and lower bounds.
    if (!intervals.empty()) {
      Float width = 0;
      for (const auto& i : intervals) {
        width += i.second - i.first;
      }

      return width;
    } else
      return -1;
  };
  // Note that we are ignoring the half lengths/weights at the two endpoints of the
  // fitting curve and just using the even weighting.
  std::vector<Float> fit_width;
  fit_width.reserve(strokefit.size());
  Cluster width_cluster;
  width_cluster.periodic = cluster.periodic;
  width_cluster.strokes = cluster.strokes;
  std::map<size_t, size_t> width_cluster_output;
  for (size_t i = 0; i < strokefit.size(); ++i) {
    const auto& xsec = samples[i];
    Vec2 debug_fit_pos = strokefit.xy(i);

    Float t = xsec.u / std::max(samples.front().u, samples.back().u);
    Float w = xsec_width(xsec, t);
    fit_width.emplace_back(w);

    if (!xsec.points.empty()) {
      width_cluster_output[width_cluster.xsecs.size()] = fit_width.size() - 1;
      width_cluster.xsecs.emplace_back(xsec);
      width_cluster.xsecs.back().width = w;
    }
  }

  // Find the width for location missing samples
  auto fill_width = [](std::vector<Float>& fit_width) {
    for (size_t i = 0; i < fit_width.size(); ++i) {
      if (fit_width[i] >= 0)
        continue;

      int next = i;
      for (; next < fit_width.size(); ++next) {
        if (fit_width[next] >= 0)
          break;
      }
      int prev = i;
      for (; prev >= 0; --prev) {
        if (fit_width[prev] >= 0)
          break;
      }

      int prev_dist = i - prev;
      int next_dist = next - i;
      prev_dist = (prev_dist == 0) ? fit_width.size() : prev_dist;
      next_dist = (next_dist == 0) ? fit_width.size() : next_dist;
      assert(!(prev_dist == fit_width.size() && next_dist == fit_width.size()));

      size_t copy_i = (prev_dist < next_dist) ? prev : next;
      fit_width[i] = fit_width[copy_i];
    }
  };

#ifdef HAS_GUROBI
  // Smooth width
  FittingEigenSparse fitting(context);
  std::vector<double> opt_widths = fitting.fit_widths(width_cluster);
  for (size_t i = 0; i < width_cluster_output.size(); ++i) {
    size_t j = width_cluster_output[i];
    fit_width[j] = opt_widths[i];
  }
#endif

  fill_width(fit_width);
  assert(strokefit.size() == fit_width.size());
  for (size_t i = 0; i < strokefit.size(); ++i) {
    strokefit.width(i) = fit_width[i];
  }
}

#endif // HAS_GUROBI

void consolidate_with_chaining_improved(const span<const Stroke> in_strokes,
                                        std::vector<Stroke>& strokes) {
#ifdef HAS_GUROBI
  std::vector<std::unordered_set<size_t>> final_clusters;
  std::vector<Stroke> temp_strokes;
  consolidate_with_chaining(in_strokes, temp_strokes, final_clusters);

  std::vector<const Stroke*> clustered_strokes;
  size_t sid = 0;
  for (const auto& cluster : final_clusters) {
    clustered_strokes.clear();
    clustered_strokes.reserve(cluster.size());
    // std::cout << "Cluster: " << std::endl;
    for (const auto si : cluster) {
      clustered_strokes.emplace_back(&in_strokes[si]);
      clustered_strokes.back()->ensure_arclengths();
      // std::cout << si << std::endl;
    }
    Stroke fit;
    fit_width_cluster(clustered_strokes, /*accuracy=*/1, /*cut_strokes=*/false, fit,
                      /*use_default_sampling=*/false);
    sid++;
    if (fit.size() == 0) {
      strokes.emplace_back(std::move(temp_strokes[sid - 1]));
    } else {
      fit.ensure_arclengths();

      auto& temp_fit = temp_strokes[sid - 1];
      // Check for width outliers which indicate fitting failure.
      Float diff_ratio = 3;
      Float sampling_diff_ratio = 3;
      Float length_diff_ratio = 5;
      Float fit_width = fit.pen_width();
      Float hausdorff, width;
      get_hausdorff_width(clustered_strokes, fit, hausdorff, width);
      hausdorff = std::max(fit_width, hausdorff);

      Float hausdorff2, width2;
      get_hausdorff_width(fit, clustered_strokes, hausdorff2, width2);
      hausdorff = std::max(hausdorff2, hausdorff);

      Float max_sample_dist = -1;
      for (size_t i = 0; i + 1 < fit.size(); ++i) {
        max_sample_dist = std::max(max_sample_dist, (fit.xy(i) - fit.xy(i + 1)).norm());
      }

      Float temp_fit_width = temp_fit.pen_width();
      if (hausdorff > diff_ratio * temp_fit_width ||
          max_sample_dist > sampling_diff_ratio * temp_fit_width ||
          std::abs(fit.length() - temp_fit.length()) >
            length_diff_ratio * temp_fit_width) {
        fit = std::move(temp_fit);
      }

      strokes.emplace_back(std::move(fit));
    }
  }

  // FIXME: Should not really destroy time information, but we don't need it for now.
  for (auto& stroke : strokes) {
    stroke.time_ = nullptr;
  }
#else // HAS_GUROBI
  static auto warned = false;
  if (!warned) {
    SPDLOG_WARN(
      "Gurobi not linked; falling back to consolidate_with_chaining with blend fitting");
    warned = true;
  }
  consolidate_with_chaining(in_strokes, strokes);
#endif // HAS_GUROBI
}

void consolidate_with_chaining(const span<const Stroke> in_strokes,
                               std::vector<Stroke>& strokes) {
  std::vector<std::unordered_set<size_t>> final_clusters;
  consolidate_with_chaining(in_strokes, strokes, final_clusters);
}

void consolidate_with_chaining(const span<const Stroke> in_strokes,
                               std::vector<Stroke>& strokes,
                               std::vector<std::unordered_set<size_t>>& final_clusters) {
  assert(!(strokes.data() <= in_strokes.data() &&
           in_strokes.data() <= strokes.data() + strokes.size()));
  strokes.clear();
  strokes.reserve(in_strokes.size());
  for (size_t i = 0; i < in_strokes.size(); ++i) {
    strokes.emplace_back(in_strokes[i].clone());
  }
  std::vector<int> stroke_clusters;
  for (size_t i = 0; i < strokes.size(); ++i) {
    stroke_clusters.emplace_back(i);
  }
  auto color_cluster = [&stroke_clusters](const int from, const int to) {
    for (auto& c : stroke_clusters)
      if (c == from)
        c = to;
  };
  for (size_t i = 1; i < strokes.size();) {
    auto found_chain = false;
    for (size_t j = 0; j < i; ++j) {
      if (strokes[i].size() > 0 && strokes[j].size() > 0) {
        found_chain = oversketch_deform(strokes[i], strokes[j]);
        if (found_chain) {
          // Successfully merged stroke j into stroke i.
          remove_duplicate_vertices(strokes[i]);
          strokes[j].size_ = 0;
          color_cluster(stroke_clusters[j], stroke_clusters[i]);
          break;
        }
      }
    }
    if (!found_chain) {
      i++; // Move on to next.
    } // Otherwise, stroke i changed, so we should re-assess stroke i with the others.
  }
  strokes.erase(std::remove_if(strokes.begin(), strokes.end(),
                               [](const Stroke& s) { return s.size() == 0; }),
                strokes.end());
  for (size_t i = 0; i < stroke_clusters.size(); ++i) {
    std::unordered_set<size_t> cluster;
    for (size_t j = 0; j < stroke_clusters.size(); ++j) {
      if (stroke_clusters[j] == i)
        cluster.emplace(j);
    }
    if (!cluster.empty())
      final_clusters.emplace_back(cluster);
  }

  std::sort(final_clusters.begin(), final_clusters.end(),
            [](const auto& a, const auto& b) {
              return *std::max_element(a.begin(), a.end()) <
                     *std::max_element(b.begin(), b.end());
            });
}

} // namespace sketching
