#include "incremental_obj.h"

#include "../force_assert.h"
#include "../stroke_graph_extra.h"
#include "incremental_param.h"
#include "incremental_region_util.h"
#include "incremental_util.h"

#include <fstream>
#include <iomanip>

#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(disable : 4267) // conversion, possible loss of data
#endif

namespace sketching {
Float max_pos_weight = 0.2;
Float outside_term_ratio = 1;
Float largest_positive_region = std::numeric_limits<Float>::infinity();

namespace {
struct JunctionWeight {
  Float w;
  std::string junc_str;
  std::string info;
  Float orig_dist;
};
} // namespace

Float compute_region_gaussian(Float ratio, Float left, Float right, Float left_w,
                              Float right_w) {
  if (ratio <= left)
    return left_w;
  if (ratio >= right)
    return right_w;
  Float sigma = (right - left) / 3;
  Float gauss = std::exp(-0.5 * ((ratio - right) / sigma) * ((ratio - right) / sigma));
  gauss = (right_w - left_w) * gauss + left_w;

  return gauss;
}

Float region_term(Float radius, Float junc_dist) {
  auto region_func = [](Float radius, Float junc_dist) -> Float {
    // Hard cut
    // Setting 0
    /*Float local_hard_region_junc_ratio = 1.;
    Float zero_region_junc_ratio = 1.5;
    Float zero_region_junc_ratio2 = 1.8;
    Float high_region_junc_ratio = 2.;*/

    // Setting 1
    Float local_hard_region_junc_ratio, zero_region_junc_ratio, zero_region_junc_ratio2,
      high_region_junc_ratio, neg_weight, pos_weight;
    /*{
      local_hard_region_junc_ratio = 0.5;
      zero_region_junc_ratio = 1.0;
      zero_region_junc_ratio2 = 2.0;
      high_region_junc_ratio = 4.0;

      neg_weight = -0.3;
      pos_weight = 0.3;
    }*/

    // Setting 2
    /*{
      local_hard_region_junc_ratio = 1.;
      zero_region_junc_ratio = 1.5;
      zero_region_junc_ratio2 = 3.0;
      high_region_junc_ratio = 5.0;

      neg_weight = -0.4;
      pos_weight = 0.2;
    }*/

    // Setting 3: Statistics 1
    /*{
      local_hard_region_junc_ratio = 0.4;
      zero_region_junc_ratio = 1.4;
      zero_region_junc_ratio2 = 8.0;
      high_region_junc_ratio = 42.0;

      neg_weight = -2.0;
      pos_weight = 1.0;
    }*/

    // Setting 4: Statistics 2
    /*{
      local_hard_region_junc_ratio = 0.4;
      zero_region_junc_ratio = 1.4;
      zero_region_junc_ratio2 = 3.2;
      high_region_junc_ratio = 42.0;

      neg_weight = -2.0;
      pos_weight = 1.0;
    }*/

    // Setting 5: Statistics 3
    /*{
      local_hard_region_junc_ratio = 0.4;
      zero_region_junc_ratio = 1.4;
      zero_region_junc_ratio2 = 3.2;
      high_region_junc_ratio = 24.0;

      neg_weight = -2.0;
      pos_weight = 1.0;
    }*/

    // Setting 6: Statistics 4
    {
      local_hard_region_junc_ratio = 0.4;
      zero_region_junc_ratio = 1.4;
      zero_region_junc_ratio2 = 3.2;
      high_region_junc_ratio = 8.0;

      neg_weight = -2.0;
      pos_weight = 1.0;
    }

    Float ratio = radius / junc_dist;
    if (ratio <= local_hard_region_junc_ratio)
      return -100;
    if (ratio >= high_region_junc_ratio || junc_dist == 0.0)
      return pos_weight;

    if (ratio > local_hard_region_junc_ratio && ratio < zero_region_junc_ratio)
      return compute_region_gaussian(ratio, local_hard_region_junc_ratio,
                                     zero_region_junc_ratio, neg_weight, 0);
    if (ratio > zero_region_junc_ratio2 && ratio < high_region_junc_ratio)
      return compute_region_gaussian(ratio, zero_region_junc_ratio2,
                                     high_region_junc_ratio, 0, pos_weight);
    return 0;
  };

  return region_term_ratio * region_func(radius, junc_dist);
}

Float get_face_size(const StrokeGraph& graph,
                    std::unordered_map<std::string, Float>& region_size_cache, size_t fi,
                    const Float check_junc_dist) {
  if (fi == graph.boundary_face_) {
    return std::numeric_limits<Float>::infinity();
  }
  auto f_id = face_id(graph.face(fi));
  if (f_id.empty() || !region_size_cache.count(f_id)) {
    // Compute maximum inscribing circle radius of the input face
    Float radius = face_maximum_inscribing_circle_radius(graph, fi);

    Float max_width = 0;
    for (const auto& s : graph.orig_strokes_) {
      max_width = std::max(max_width, s.pen_width());
    }

    if (radius < accurate_radius_check_ratio * max_width) {
      Eigen::Vector2d center;
      Float r = face_maximum_inscribing_circle_radius_clipping(graph, fi, center);
      SPDLOG_DEBUG("get_face_size: {} => {}", radius, r);

      if (r > radius) {
        radius = r;
      }
    }

    region_size_cache[f_id] = radius;
  }
  /*else {
    Float radius = face_maximum_inscribing_circle_radius(graph, fi);
    assert(std::abs(region_size_cache[f_id] - radius) <
       std::numeric_limits<Float>::epsilon());
  }*/
  return region_size_cache[f_id];
}

Float get_face_size_const(const StrokeGraph& graph,
                          const std::unordered_map<std::string, Float>& region_size_cache,
                          size_t fi) {
  if (fi == graph.boundary_face_) {
    return std::numeric_limits<Float>::infinity();
  }
  auto f_id = face_id(graph.face(fi));
  Float radius;
  if (f_id.empty() || !region_size_cache.count(f_id)) {
    // Compute maximum inscribing circle radius of the input face
    Float radius = face_maximum_inscribing_circle_radius(graph, fi);

    Float max_width = 0;
    for (const auto& s : graph.orig_strokes_) {
      max_width = std::max(max_width, s.pen_width());
    }

    if (radius < accurate_radius_check_ratio * max_width) {
      Eigen::Vector2d center;
      Float r = face_maximum_inscribing_circle_radius_clipping(graph, fi, center);
      SPDLOG_DEBUG("get_face_size: {} => {}", radius, r);

      if (r > radius) {
        radius = r;
      }
    }
    return radius;
  } else {
    radius = region_size_cache.at(f_id);
  }

  return radius;
}

Float get_junc_prob(const std::vector<Junction>& juncs, const std::string& junc_str) {
  for (const auto& j : juncs) {
    if (j.repr == junc_str)
      return j.probability;
  }

  return -1;
}

void update_high_valence_probabilities(const StrokeGraph& graph,
                                       const StrokeGraph& final_graph,
                                       std::vector<Junction>& candidates) {
  std::unordered_map<size_t, std::vector<std::string>> v2juncs;
  for (size_t i = 0; i < graph.vertices_.size(); ++i) {
    if (!graph.vertex(i).is_active())
      continue;
    if (!v2juncs.count(i))
      v2juncs[i] = std::vector<std::string>();
    for (const auto& vid : graph.vertex(i).vertex_ids()) {
      auto vid_str = vid.repr();
      // Only keep actual junctions
      if (vid_str[0] == 'j')
        v2juncs[i].emplace_back(vid_str);
    }
  }

  // Update probability at high-valence junctions
  StrokeGraph varying_graph = final_graph.clone();
  for (const auto& [vid, juncs] : v2juncs) {
    if (juncs.size() < 2)
      continue;
    for (const auto& junc_str : juncs) {
      int i = 0;
      for (; i < candidates.size(); ++i) {
        if (candidates[i].repr == junc_str) {
          break;
        }
      }
      force_assert(i < candidates.size());

      // Mark other strokes then recompute the predictions
      std::unordered_set<int> junc_sid{candidates[i].points[0].first,
                                       candidates[i].points[1].first};
      const auto he = graph.vertex(vid).hedge();
      auto it = he;
      do {
        auto he_sid = it.stroke_idx();
        StrokeTime end((int)he_sid, (it.forward()) ? 0 : 1);
        const auto ok = convert_strokes2orig(graph, end);
        force_assert(ok && "couldn't map from strokes to orig");

        // Find the chain in the init graph
        StrokeTime end_init = end;
        const auto ok2 = convert_orig2strokes(varying_graph, end_init);
        force_assert(ok2 && "couldn't map from orig to strokes");

        std::vector<std::pair<size_t, bool>> orig_indices;
        size_t he_init = end_init.first * 2;
        if (end_init.second == 0.0 || end_init.second == 1.0) {
          if (end_init.second == 1.0)
            he_init++;
          get_forward_chain_original_indices(varying_graph, he_init, orig_indices);
        } else {
          get_forward_chain_original_indices(varying_graph, he_init, orig_indices);
          get_forward_chain_original_indices(varying_graph, he_init + 1, orig_indices);
        }

        for (const auto& chain_orig_sid : orig_indices)
          if (!junc_sid.count(chain_orig_sid.first)) {
            varying_graph.orig_bvh_->masked_nodes.emplace_back(chain_orig_sid.first);
          }

        it = it.twin().next();
      } while (it != he);

      // Predict on the same graph used for the initial predictions. Only difference is
      // the masked original strokes.
      std::vector<Junction> junc{candidates[i]};
      update_disconnected_junction_predictions(varying_graph, junc,
                                               prediction_feature_type);
      candidates[i].fea.ee_fea_ = junc.front().fea.ee_fea_;
      candidates[i].fea.es_fea_ = junc.front().fea.es_fea_;
      candidates[i].probability = junc.front().probability;

      // Reset mask
      varying_graph.orig_bvh_->masked_nodes.clear();
    }
  }
}

void update_corner_probabilities(const StrokeGraph& graph, const StrokeGraph& final_graph,
                                 std::vector<Junction>& candidates) {
  std::unordered_map<size_t, std::vector<std::string>> v2juncs;
  for (size_t i = 0; i < graph.vertices_.size(); ++i) {
    if (!graph.vertex(i).is_active())
      continue;
    if (!v2juncs.count(i))
      v2juncs[i] = std::vector<std::string>();
    for (const auto& vid : graph.vertex(i).vertex_ids()) {
      auto vid_str = vid.repr();
      // Only keep actual junctions
      if (vid_str[0] == 'j')
        v2juncs[i].emplace_back(vid_str);
    }
  }

  // Update probability at high-valence junctions
  StrokeGraph varying_graph = final_graph.clone();
  for (const auto& [vid, juncs] : v2juncs) {
    if (juncs.size() < 2)
      continue;

    std::vector<std::string> lookup_juncs = juncs;

    // 1. Gather the circulation
    struct PredictionEnd {
      StrokeTime dangling_end;
      StrokeTime point_to_end;
      size_t he_idx;
      std::vector<size_t> adjacent_he_indices;
      std::vector<StrokeTime> adjacent_ends;
      std::vector<StrokeTime> adjacent_to_ends;
      std::vector<size_t> adjacent_junctions;
    };
    std::vector<PredictionEnd> prediction_ends;
    const auto he = graph.vertex(vid).hedge();
    auto it = he;
    do {
      auto he_sid = it.stroke_idx();
      StrokeTime end((int)he_sid, (it.forward()) ? 0 : 1);
      const auto ok = convert_strokes2orig(graph, end);
      force_assert(ok && "couldn't map from strokes to orig");
      StrokeTime to_end((int)he_sid, (it.forward()) ? 1 : 0);
      const auto ok2 = convert_strokes2orig(graph, to_end);
      force_assert(ok2 && "couldn't map from strokes to orig");

      PredictionEnd pred_end;
      pred_end.dangling_end = end;
      pred_end.he_idx = it.index_;
      pred_end.point_to_end = to_end;
      prediction_ends.emplace_back(pred_end);

      it = it.twin().next();
    } while (it != he);

    for (size_t i = 0; i < prediction_ends.size(); ++i) {
      size_t prev = (i + prediction_ends.size() - 1) % prediction_ends.size();
      size_t next = (i + 1) % prediction_ends.size();
      prediction_ends[i].adjacent_he_indices.emplace_back(prediction_ends[prev].he_idx);
      prediction_ends[i].adjacent_he_indices.emplace_back(prediction_ends[next].he_idx);
      prediction_ends[i].adjacent_ends.emplace_back(prediction_ends[prev].dangling_end);
      prediction_ends[i].adjacent_ends.emplace_back(prediction_ends[next].dangling_end);
      prediction_ends[i].adjacent_to_ends.emplace_back(
        prediction_ends[prev].point_to_end);
      prediction_ends[i].adjacent_to_ends.emplace_back(
        prediction_ends[next].point_to_end);

      auto find_junc = [&lookup_juncs, &candidates](
                         const StrokeTime& end0, const StrokeTime& end1,
                         const StrokeTime& to_end0, const StrokeTime& to_end1) -> int {
        for (const auto& junc_str : lookup_juncs) {
          int i = 0;
          for (; i < candidates.size(); ++i) {
            if (candidates[i].repr == junc_str) {
              break;
            }
          }

          if ((candidates[i].points[0] == end0 && candidates[i].points[1] == end1) ||
              (candidates[i].points[0] == end1 && candidates[i].points[1] == end0)) {
            if (!candidates[i].point_tos.empty()) {
              if ((candidates[i].points[0] == end0 &&
                   candidates[i].point_tos.front() == to_end0) ||
                  (candidates[i].points[0] == end1 &&
                   candidates[i].point_tos.front() == to_end1))
                return i;
            } else
              return i;
          }
        }

        return -1;
      };
      auto junc_prev = find_junc(
        prediction_ends[i].dangling_end, prediction_ends[i].adjacent_ends[0],
        prediction_ends[i].point_to_end, prediction_ends[i].adjacent_to_ends[0]);
      auto junc_next = find_junc(
        prediction_ends[i].dangling_end, prediction_ends[i].adjacent_ends[1],
        prediction_ends[i].point_to_end, prediction_ends[i].adjacent_to_ends[1]);
      prediction_ends[i].adjacent_junctions.emplace_back(junc_prev);
      prediction_ends[i].adjacent_junctions.emplace_back(junc_next);

      // Reset probabilities
      if (junc_prev >= 0)
        candidates[junc_prev].probability = 0.5;
      if (junc_next >= 0)
        candidates[junc_next].probability = 0.5;
    }

    // Make predictions
    for (size_t i = 0; i < prediction_ends.size(); ++i) {
      // If both sides are snapped
      if (prediction_ends[i].adjacent_junctions[0] < 0 &&
          prediction_ends[i].adjacent_junctions[1] < 0)
        continue;
      Float scale_factor = (prediction_ends[i].adjacent_junctions[0] >= 0 &&
                            prediction_ends[i].adjacent_junctions[1] >= 0)
                             ? 0.5
                             : 1;
      if (prediction_ends[i].adjacent_junctions[0] >= 0) {
        std::vector<std::pair<size_t, bool>> orig_indices;
        get_forward_chain_original_indices(varying_graph, prediction_ends[i].he_idx,
                                           orig_indices);
        get_forward_chain_original_indices(
          varying_graph, prediction_ends[i].adjacent_he_indices[0], orig_indices);

        for (const auto& chain_orig_sid : orig_indices)
          varying_graph.orig_bvh_->masked_nodes.emplace_back(chain_orig_sid.first);

        // Predict on the same graph used for the initial predictions. Only difference is
        // the masked original strokes.
        size_t prev = prediction_ends[i].adjacent_junctions[0];
        std::vector<Junction> junc{candidates[prev]};
        junc.front().type = JunctionType::T;
        junc.front().fea.type_ = FeatureVector::EndStroke;
        std::vector<size_t> combine_he_idx{prediction_ends[i].he_idx,
                                           prediction_ends[i].adjacent_he_indices[0]};
        varying_graph.orig_bvh_->combine_he_indices = combine_he_idx;
        update_disconnected_junction_predictions(varying_graph, junc,
                                                 prediction_feature_type, false, true);
        candidates[prev].fea = junc.front().fea;
        candidates[prev].probability = scale_factor * (junc.front().probability) + 0.5;

        // Reset mask
        varying_graph.orig_bvh_->masked_nodes.clear();
        varying_graph.orig_bvh_->combine_he_indices.clear();
      }
      if (prediction_ends[i].adjacent_junctions[1] >= 0) {
        std::vector<std::pair<size_t, bool>> orig_indices;
        get_forward_chain_original_indices(varying_graph, prediction_ends[i].he_idx,
                                           orig_indices);
        get_forward_chain_original_indices(
          varying_graph, prediction_ends[i].adjacent_he_indices[1], orig_indices);

        for (const auto& chain_orig_sid : orig_indices)
          varying_graph.orig_bvh_->masked_nodes.emplace_back(chain_orig_sid.first);

        // Predict on the same graph used for the initial predictions. Only difference is
        // the masked original strokes.
        size_t next = prediction_ends[i].adjacent_junctions[1];
        std::vector<Junction> junc{candidates[next]};
        junc.front().type = JunctionType::T;
        junc.front().fea.type_ = FeatureVector::EndStroke;
        std::vector<size_t> combine_he_idx{prediction_ends[i].he_idx,
                                           prediction_ends[i].adjacent_he_indices[1]};
        varying_graph.orig_bvh_->combine_he_indices = combine_he_idx;
        update_disconnected_junction_predictions(varying_graph, junc,
                                                 prediction_feature_type, false, true);
        candidates[next].fea = junc.front().fea;
        candidates[next].probability = scale_factor * (junc.front().probability) + 0.5;

        // Reset mask
        varying_graph.orig_bvh_->masked_nodes.clear();
        varying_graph.orig_bvh_->combine_he_indices.clear();
      }
    }
  }
}

bool is_sol_graph_valid(
  const StrokeGraph& graph,
  const std::unordered_map<std::string, Float>& junc_distance_map,
  const std::vector<Junction>& candidates, const GraphState& in_state,
  const std::unordered_map<std::string, Float>& in_region_size_cache,
  Float certain_prob) {
  std::unordered_map<std::string, Float> region_size_cache = in_region_size_cache;
  std::unordered_set<std::string> seen_junctions;
  for (size_t fi = 1; fi < graph.faces_.size(); ++fi) {
    seen_junctions.clear();
    for (const auto hi : graph.faces_[fi].cycles_) {
      const auto he = graph.hedge(hi);
      auto it = he;
      do {
        // Check if it's still valid
        for (const auto& [t, vid] : graph.strokes2vid_[it.stroke_idx()]) {
          auto vid_str = vid.repr();
          if (junc_distance_map.count(vid_str) ||
              !in_state.junc_distance_map_.count(vid_str))
            continue;

          if (seen_junctions.count(vid_str))
            continue;
          seen_junctions.emplace(vid_str);

          Float junc_dist = in_state.junc_distance_map_.at(vid_str);
          Float prob = get_junc_prob(in_state.candidates_, vid_str);
          // candidates_ and junc_distance_map_ need to be consistent
          assert(prob >= 0);
          if (prob < certain_prob &&
              !hard_region_condition_check(graph, region_size_cache, fi, junc_dist)) {
            return false;
          }
        }

        // Check if the current multiway junction is a connection and belongs to this face
        for (const auto& vid : it.origin().vertex_ids()) {
          auto vid_str = vid.repr();
          bool no_records =
            (vid.connection_type_ == StrokeGraph::VertexID::Initialization) ||
            (!in_state.junc_distance_map_.count(vid_str) &&
             !junc_distance_map.count(vid_str));
          if (!no_records) {
            size_t i;
            for (i = 0; i < candidates.size() && candidates[i].repr != vid_str; ++i)
              ;
            if (i >= candidates.size())
              no_records = true;
          }
          bool in_region = !(no_records ||
                             (in_state.junc_distance_map_.count(vid_str) &&
                              !is_high_valence_junction_in_region_cycle(
                                graph, in_state.candidates_, fi, it.origin(), vid_str)) ||
                             (junc_distance_map.count(vid_str) &&
                              !is_high_valence_junction_in_region_cycle(
                                graph, candidates, fi, it.origin(), vid_str)));

          if (in_region && !seen_junctions.count(vid_str)) {
            assert(junc_distance_map.count(vid_str) ||
                   in_state.junc_distance_map_.count(vid_str));
            seen_junctions.emplace(vid_str);

            Float junc_dist = (in_state.junc_distance_map_.count(vid_str))
                                ? in_state.junc_distance_map_.at(vid_str)
                                : junc_distance_map.at(vid_str);
            Float radius = get_face_size(graph, region_size_cache, fi, junc_dist);
            Float prob = (in_state.junc_distance_map_.count(vid_str))
                           ? get_junc_prob(in_state.candidates_, vid_str)
                           : get_junc_prob(candidates, vid_str);
            // candidates_ and junc_distance_map_ need to be consistent
            assert(prob >= 0);
            if (prob < certain_prob &&
                !hard_region_condition_check(graph, region_size_cache, fi, junc_dist)) {
              return false;
            }
          }
        }

        it = it.next();
      } while (it != he);
    }
  }

  return true;
}

Float obj_function(const GraphState& state, Float& region_obj,
                   const std::string& txt_filename) {
  RegionSolution sol;
  sol.candidates_ = state.candidates_;
  sol.connectivity_ = state.connectivity_;
  sol.graph_ = state.graph_.clone();
  sol.junc_distance_map_ = state.junc_distance_map_;
  sol.region_size_cache_ = state.region_size_cache_;

  Float obj = obj_function(sol, region_obj, txt_filename);

  return obj;
}

Float obj_function_per_junction(const RegionSolution& state, Float& region_obj,
                                const std::string& txt_filename) {
  Float obj = 0;

  // Region term. Skip the outside face.
  std::map<std::string, std::vector<std::string>> junc_region_sizes;
  region_obj = 0;
  std::unordered_set<std::string> active_junctions, adj_outside_junctions;

  auto is_init_junc_region_boundary = [](const StrokeGraph::VertexView& v,
                                         size_t fi) -> bool {
    size_t boundary_edge = 0;
    const auto he = v.hedge();
    auto it = he;
    do {
      if ((it.face_idx() == fi && it.twin().face_idx() != fi) ||
          (it.face_idx() != fi && it.twin().face_idx() == fi))
        boundary_edge++;

      it = it.twin().next();
    } while (it != he);
    return boundary_edge == 2;
  };

  std::set<std::string> corner_juncs;
  // Check if it's the special case of a intersection corner candidate that is a
  // T-junction (only assigned as a T-junction for book-keeping)
  // Assume we don't have multiple junctions of this type connected to the same
  // vertex
  {
    for (size_t fi = 0; fi < state.graph_.faces_.size(); ++fi) {
      for (const auto hi : state.graph_.faces_[fi].cycles_) {
        const auto he = state.graph_.hedge(hi);
        auto it = he;
        do {
          int v_idx = it.origin().index_;

          bool seen_t = false;
          size_t v_valence = it.origin().valence();
          size_t num_junc = 0;
          std::string junc_str = "";
          for (const auto& vid : it.origin().vertex_ids()) {
            auto vid_str = vid.repr();
            bool is_active = false;
            if (vid.connection_type_ == StrokeGraph::VertexID::Type::Junction) {
              for (size_t i = 0; i < state.candidates_.size(); ++i) {
                if (state.candidates_[i].repr == vid_str) {
                  if (is_junction_on_cycle(it.origin(), state.candidates_[i])) {
                    active_junctions.emplace(vid_str);
                    is_active = true;
                    num_junc++;
                    seen_t = (state.candidates_[i].type == JunctionType::T);
                    junc_str = vid_str;
                  }
                  break;
                }
              }
            }
          }

          if (seen_t && v_valence > 3 && num_junc == 1)
            corner_juncs.emplace(junc_str);
          it = it.next();
        } while (it != he);
      }
    }
  }

  std::map<std::pair<size_t, size_t>, std::vector<JunctionWeight>> vf_weights;
  std::map<std::string, std::set<size_t>> junc2fi;
  for (size_t fi = 0; fi < state.graph_.faces_.size(); ++fi) {
    for (const auto hi : state.graph_.faces_[fi].cycles_) {
      const auto he = state.graph_.hedge(hi);
      auto it = he;
      do {
        int v_idx = it.origin().index_;
        if (!vf_weights.count(std::make_pair(v_idx, fi)))
          vf_weights[std::make_pair(v_idx, fi)] = std::vector<JunctionWeight>();

        for (const auto& vid : it.origin().vertex_ids()) {
          auto vid_str = vid.repr();

          // Check if this junction connection is still active wrt the high-valence
          // junction cycle
          bool is_active = false;
          if (vid.connection_type_ == StrokeGraph::VertexID::Type::Junction) {
            for (size_t i = 0; i < state.candidates_.size(); ++i) {
              if (state.candidates_[i].repr == vid_str) {
                if (is_junction_on_cycle(it.origin(), state.candidates_[i])) {
                  active_junctions.emplace(vid_str);
                  is_active = true;
                }
                break;
              }
            }
          }

          // Init junctions
          std::string junc_str = vid_str;
          int junc_i = -1;
          for (size_t i = 0; i < state.candidates_.size(); ++i) {
            if (state.candidates_[i].repr == vid_str) {
              std::stringstream ss;
              ss << std::setprecision(2) << "(" << state.candidates_[i].points[0].first
                 << ", " << state.candidates_[i].points[0].second << " - "
                 << state.candidates_[i].points[1].first << ", "
                 << state.candidates_[i].points[1].second << ")";
              junc_str += ": " + ss.str();
              junc_i = i;
              break;
            }
          }
          bool on_boundary = false;
          if (junc_i >= 0)
            on_boundary =
              (corner_juncs.count(vid_str))
                ? is_high_valence_junction_in_region_cycle(
                    state.graph_, state.candidates_, fi, it.origin(), vid_str)
                : is_high_valence_junction_on_boundary_cycle(
                    state.graph_, state.candidates_, fi, it.origin(), vid_str);
          if (is_active && state.junc_distance_map_.count(vid_str) && on_boundary) {
            if (fi > 0) {
              Float radius =
                get_face_size_const(state.graph_, state.region_size_cache_, fi);

              Float w = region_term(radius, state.junc_distance_map_.at(vid_str));

              std::string junc_info =
                "f" + std::to_string(fi) + ": " + std::to_string(radius) + "(" +
                std::to_string(radius / state.junc_distance_map_.at(vid_str)) + ")" +
                " = " + std::to_string(w) +
                "; p = " + std::to_string(state.candidates_[junc_i].probability);

              bool seen = false;
              for (const auto& w_info : vf_weights[std::make_pair(v_idx, fi)]) {
                if (w_info.junc_str == vid_str) {
                  seen = true;
                  break;
                }
              }

              if (!seen) {
                vf_weights[std::make_pair(v_idx, fi)].emplace_back(JunctionWeight{
                  w, vid_str, junc_info, state.junc_distance_map_.at(vid_str)});

                if (!junc2fi.count(vid_str))
                  junc2fi[vid_str] = std::set<size_t>();
                junc2fi[vid_str].emplace(fi);
              }
            } else if (fi == 0) {
              Float w = outside_term_ratio * region_term(1, 0);
              std::string junc_info =
                "f" + std::to_string(fi) + " = " + std::to_string(w) +
                "; p = " + std::to_string(state.candidates_[junc_i].probability);

              bool seen = false;
              for (const auto& w_info : vf_weights[std::make_pair(v_idx, fi)]) {
                if (w_info.junc_str == vid_str) {
                  seen = true;
                  break;
                }
              }

              if (!seen) {
                vf_weights[std::make_pair(v_idx, fi)].emplace_back(JunctionWeight{
                  w, vid_str, junc_info, state.junc_distance_map_.at(vid_str)});

                adj_outside_junctions.emplace(vid_str);
                if (!junc2fi.count(vid_str))
                  junc2fi[vid_str] = std::set<size_t>();
                junc2fi[vid_str].emplace(fi);
              }
            }
          }
        }

        it = it.next();
      } while (it != he);
    }
  }

  {
    std::map<std::pair<size_t, size_t>, std::vector<JunctionWeight>> tmp_vf_weights;
    for (const auto& vf_w : vf_weights) {
      if (vf_w.second.empty())
        continue;
      tmp_vf_weights.emplace(vf_w);
    }
    vf_weights = std::move(tmp_vf_weights);
  }

  // If a vertex show up twice or more times, it means it's a high-valence vertex with
  // multiple adjacent triangles belonging to this face. This is counted as non-region
  // forming and thus is discarded.
  for (const auto& vf_w : vf_weights) {
    if (vf_w.second.empty() || vf_w.second.size() > 1)
      continue;
    region_obj += vf_w.second.front().w;
    if (!vf_w.second.front().info.empty())
      junc_region_sizes[vf_w.second.front().junc_str].emplace_back(
        vf_w.second.front().info);
  }

  // Junction
  // Boundary
  region_obj = 0;
  for (size_t i = 0; i < state.candidates_.size(); ++i) {
    if (state.connectivity_[i] && active_junctions.count(state.candidates_[i].repr)) {
      assert(state.candidates_[i].probability >= 0);
      Float junc_scale = std::max(junc2fi[state.candidates_[i].repr].size(), (size_t)1);
      /*if (region_term_ratio == 0)
        junc_scale = 1;*/
      obj += junc_scale * (state.candidates_[i].probability - junction_cutoff);
    }
  }

  // SPDLOG_INFO("Junc: {}; Region: {} => {}", obj, region_obj, obj + region_obj);

  obj += region_obj;

  return obj;
}

Float obj_function(const RegionSolution& state, Float& region_obj,
                   const std::string& txt_filename) {
  Float obj = 0;

  Float snap_junc_weight = 0.;
  Float snap_region_weight = 0.;

  // Region term. Skip the outside face.
  std::map<std::string, std::vector<std::string>> junc_region_sizes;
  region_obj = 0;
  std::unordered_set<std::string> active_junctions, adj_outside_junctions;

  auto is_init_junc_region_boundary = [](const StrokeGraph::VertexView& v,
                                         size_t fi) -> bool {
    size_t boundary_edge = 0;
    const auto he = v.hedge();
    auto it = he;
    do {
      if ((it.face_idx() == fi && it.twin().face_idx() != fi) ||
          (it.face_idx() != fi && it.twin().face_idx() == fi))
        boundary_edge++;

      it = it.twin().next();
    } while (it != he);
    return boundary_edge == 2;
  };

  std::set<std::string> corner_juncs;
  // Check if it's the special case of a intersection corner candidate that is a
  // T-junction (only assigned as a T-junction for book-keeping)
  // Assume we don't have multiple junctions of this type connected to the same
  // vertex
  {
    for (size_t fi = 0; fi < state.graph_.faces_.size(); ++fi) {
      for (const auto hi : state.graph_.faces_[fi].cycles_) {
        const auto he = state.graph_.hedge(hi);
        auto it = he;
        do {
          int v_idx = it.origin().index_;

          bool seen_t = false;
          size_t v_valence = it.origin().valence();
          size_t num_junc = 0;
          std::string junc_str = "";
          for (const auto& vid : it.origin().vertex_ids()) {
            auto vid_str = vid.repr();
            bool is_active = false;
            if (vid.connection_type_ == StrokeGraph::VertexID::Type::Junction) {
              for (size_t i = 0; i < state.candidates_.size(); ++i) {
                if (state.candidates_[i].repr == vid_str) {
                  if (is_junction_on_cycle(it.origin(), state.candidates_[i])) {
                    active_junctions.emplace(vid_str);
                    is_active = true;
                    num_junc++;
                    seen_t = (state.candidates_[i].type == JunctionType::T);
                    junc_str = vid_str;
                  }
                  break;
                }
              }
            }
          }

          if (seen_t && v_valence > 3 && num_junc == 1)
            corner_juncs.emplace(junc_str);
          it = it.next();
        } while (it != he);
      }
    }
  }

  std::map<std::pair<size_t, size_t>, std::vector<JunctionWeight>> vf_weights;
  std::map<std::string, std::set<size_t>> junc2fi;
  for (size_t fi = 0; fi < state.graph_.faces_.size(); ++fi) {
    for (const auto hi : state.graph_.faces_[fi].cycles_) {
      const auto he = state.graph_.hedge(hi);
      auto it = he;
      do {
        int v_idx = it.origin().index_;
        if (!vf_weights.count(std::make_pair(v_idx, fi)))
          vf_weights[std::make_pair(v_idx, fi)] = std::vector<JunctionWeight>();

        for (const auto& vid : it.origin().vertex_ids()) {
          auto vid_str = vid.repr();

          // Check if this junction connection is still active wrt the high-valence
          // junction cycle
          bool is_active = false;
          if (vid.connection_type_ == StrokeGraph::VertexID::Type::Junction) {
            for (size_t i = 0; i < state.candidates_.size(); ++i) {
              if (state.candidates_[i].repr == vid_str) {
                if (is_junction_on_cycle(it.origin(), state.candidates_[i])) {
                  active_junctions.emplace(vid_str);
                  is_active = true;
                }
                break;
              }
            }
          }

          // Init junctions
          std::string junc_str = vid_str;
          int junc_i = -1;
          for (size_t i = 0; i < state.candidates_.size(); ++i) {
            if (state.candidates_[i].repr == vid_str) {
              std::stringstream ss;
              ss << std::setprecision(2) << "(" << state.candidates_[i].points[0].first
                 << ", " << state.candidates_[i].points[0].second << " - "
                 << state.candidates_[i].points[1].first << ", "
                 << state.candidates_[i].points[1].second << ")";
              junc_str += ": " + ss.str();
              junc_i = i;
              break;
            }
          }
          bool on_boundary = false;
          if (junc_i >= 0)
            on_boundary =
              (corner_juncs.count(vid_str))
                ? is_high_valence_junction_in_region_cycle(
                    state.graph_, state.candidates_, fi, it.origin(), vid_str)
                : is_high_valence_junction_on_boundary_cycle(
                    state.graph_, state.candidates_, fi, it.origin(), vid_str);
          if (is_active && state.junc_distance_map_.count(vid_str) && on_boundary) {
            if (fi > 0) {
              Float radius =
                get_face_size_const(state.graph_, state.region_size_cache_, fi);

              Float w = region_term(radius, state.junc_distance_map_.at(vid_str));

              std::string junc_info =
                "f" + std::to_string(fi) + ": " + std::to_string(radius) + "(" +
                std::to_string(radius / state.junc_distance_map_.at(vid_str)) + ")" +
                " = " + std::to_string(w) +
                "; p = " + std::to_string(state.candidates_[junc_i].probability);

              bool seen = false;
              for (const auto& w_info : vf_weights[std::make_pair(v_idx, fi)]) {
                if (w_info.junc_str == vid_str) {
                  seen = true;
                  break;
                }
              }

              if (!seen) {
                vf_weights[std::make_pair(v_idx, fi)].emplace_back(JunctionWeight{
                  w, vid_str, junc_info, state.junc_distance_map_.at(vid_str)});

                if (!junc2fi.count(vid_str))
                  junc2fi[vid_str] = std::set<size_t>();
                junc2fi[vid_str].emplace(fi);
              }
            } else if (fi == 0) {
              Float w = outside_term_ratio * region_term(1, 0);
              std::string junc_info =
                "f" + std::to_string(fi) + " = " + std::to_string(w) +
                "; p = " + std::to_string(state.candidates_[junc_i].probability);

              bool seen = false;
              for (const auto& w_info : vf_weights[std::make_pair(v_idx, fi)]) {
                if (w_info.junc_str == vid_str) {
                  seen = true;
                  break;
                }
              }

              if (!seen) {
                vf_weights[std::make_pair(v_idx, fi)].emplace_back(JunctionWeight{
                  w, vid_str, junc_info, state.junc_distance_map_.at(vid_str)});

                adj_outside_junctions.emplace(vid_str);
                if (!junc2fi.count(vid_str))
                  junc2fi[vid_str] = std::set<size_t>();
                junc2fi[vid_str].emplace(fi);
              }
            }
          }
        }

        it = it.next();
      } while (it != he);
    }
  }

  {
    std::map<std::pair<size_t, size_t>, std::vector<JunctionWeight>> tmp_vf_weights;
    for (const auto& vf_w : vf_weights) {
      if (vf_w.second.empty())
        continue;
      tmp_vf_weights.emplace(vf_w);
    }
    vf_weights = std::move(tmp_vf_weights);
  }

  // If a vertex show up twice or more times, it means it's a high-valence vertex with
  // multiple adjacent triangles belonging to this face. This is counted as non-region
  // forming and thus is discarded.
  std::map<size_t, std::vector<JunctionWeight>> region_largest_gap_weight;
  for (const auto& vf_w : vf_weights) {
    if (vf_w.second.empty() || vf_w.first.second == state.graph_.boundary_face_)
      continue;

    region_largest_gap_weight[vf_w.first.second].emplace_back(vf_w.second.front());
  }

  for (auto& [fi, juncs] : region_largest_gap_weight) {
    std::sort(
      juncs.begin(), juncs.end(),
      [](const JunctionWeight& a, const JunctionWeight& b) -> bool { return a.w < b.w; });
    region_obj += std::min(largest_positive_region, juncs.front().w);
    if (!juncs.front().info.empty())
      junc_region_sizes[juncs.front().junc_str].emplace_back(juncs.front().info);
  }
  region_obj = 0;

  // Junction
  // Boundary
  for (size_t i = 0; i < state.candidates_.size(); ++i) {
    if (state.connectivity_[i] && active_junctions.count(state.candidates_[i].repr)) {
      assert(state.candidates_[i].probability >= 0);
      Float junc_scale = std::max(junc2fi[state.candidates_[i].repr].size(), (size_t)1);
      /*if (region_term_ratio == 0)
        junc_scale = 1;*/
      junc_scale = 1;
      obj += junc_scale * (state.candidates_[i].probability - junction_cutoff);
    }
  }

  // SPDLOG_INFO("Junc: {}; Region: {} => {}", obj, region_obj, obj + region_obj);

  obj += region_obj;

  return obj;
}

bool violation_check(const GraphState& state) {
  RegionSolution sol;
  sol.candidates_ = state.candidates_;
  sol.connectivity_ = state.connectivity_;
  sol.graph_ = state.graph_.clone();
  sol.junc_distance_map_ = state.junc_distance_map_;
  sol.region_size_cache_ = state.region_size_cache_;

  return violation_check(sol);
}

bool violation_check(const RegionSolution& state) {
  Float obj = 0;

  Float snap_junc_weight = 0.;
  Float snap_region_weight = 0.;

  bool seen_violation = false;

  // Region term. Skip the outside face.
  std::map<std::string, std::vector<std::string>> junc_region_sizes;
  std::unordered_set<std::string> active_junctions, adj_outside_junctions;

  auto is_init_junc_region_boundary = [](const StrokeGraph::VertexView& v,
                                         size_t fi) -> bool {
    size_t boundary_edge = 0;
    const auto he = v.hedge();
    auto it = he;
    do {
      if ((it.face_idx() == fi && it.twin().face_idx() != fi) ||
          (it.face_idx() != fi && it.twin().face_idx() == fi))
        boundary_edge++;

      it = it.twin().next();
    } while (it != he);
    return boundary_edge == 2;
  };

  std::set<std::string> corner_juncs;
  // Check if it's the special case of a intersection corner candidate that is a
  // T-junction (only assigned as a T-junction for book-keeping)
  // Assume we don't have multiple junctions of this type connected to the same
  // vertex
  {
    for (size_t fi = 0; fi < state.graph_.faces_.size(); ++fi) {
      for (const auto hi : state.graph_.faces_[fi].cycles_) {
        const auto he = state.graph_.hedge(hi);
        auto it = he;
        do {
          int v_idx = it.origin().index_;

          bool seen_t = false;
          size_t v_valence = it.origin().valence();
          size_t num_junc = 0;
          std::string junc_str = "";
          for (const auto& vid : it.origin().vertex_ids()) {
            auto vid_str = vid.repr();
            bool is_active = false;
            if (vid.connection_type_ == StrokeGraph::VertexID::Type::Junction) {
              for (size_t i = 0; i < state.candidates_.size(); ++i) {
                if (state.candidates_[i].repr == vid_str) {
                  if (is_junction_on_cycle(it.origin(), state.candidates_[i])) {
                    active_junctions.emplace(vid_str);
                    is_active = true;
                    num_junc++;
                    seen_t = (state.candidates_[i].type == JunctionType::T);
                    junc_str = vid_str;
                  }
                  break;
                }
              }
            }
          }

          if (seen_t && v_valence > 3 && num_junc == 1)
            corner_juncs.emplace(junc_str);
          it = it.next();
        } while (it != he);
      }
    }
  }

  std::map<std::pair<size_t, size_t>, std::vector<JunctionWeight>> vf_weights;
  std::map<std::string, std::set<size_t>> junc2fi;
  for (size_t fi = 0; fi < state.graph_.faces_.size(); ++fi) {
    for (const auto hi : state.graph_.faces_[fi].cycles_) {
      const auto he = state.graph_.hedge(hi);
      auto it = he;
      do {
        int v_idx = it.origin().index_;
        if (!vf_weights.count(std::make_pair(v_idx, fi)))
          vf_weights[std::make_pair(v_idx, fi)] = std::vector<JunctionWeight>();

        for (const auto& vid : it.origin().vertex_ids()) {
          auto vid_str = vid.repr();

          // Check if this junction connection is still active wrt the high-valence
          // junction cycle
          bool is_active = false;
          if (vid.connection_type_ == StrokeGraph::VertexID::Type::Junction) {
            for (size_t i = 0; i < state.candidates_.size(); ++i) {
              if (state.candidates_[i].repr == vid_str) {
                if (is_junction_on_cycle(it.origin(), state.candidates_[i])) {
                  active_junctions.emplace(vid_str);
                  is_active = true;
                }
                break;
              }
            }
          }

          // Init junctions
          std::string junc_str = vid_str;
          int junc_i = -1;
          for (size_t i = 0; i < state.candidates_.size(); ++i) {
            if (state.candidates_[i].repr == vid_str) {
              std::stringstream ss;
              ss << std::setprecision(2) << "(" << state.candidates_[i].points[0].first
                 << ", " << state.candidates_[i].points[0].second << " - "
                 << state.candidates_[i].points[1].first << ", "
                 << state.candidates_[i].points[1].second << ")";
              junc_str += ": " + ss.str();
              junc_i = i;
              break;
            }
          }
          bool on_boundary = false;
          if (junc_i >= 0)
            on_boundary =
              (corner_juncs.count(vid_str))
                ? is_high_valence_junction_in_region_cycle(
                    state.graph_, state.candidates_, fi, it.origin(), vid_str)
                : is_high_valence_junction_on_boundary_cycle(
                    state.graph_, state.candidates_, fi, it.origin(), vid_str);
          if (is_active && state.junc_distance_map_.count(vid_str) && on_boundary) {
            if (fi > 0) {
              Float radius =
                get_face_size_const(state.graph_, state.region_size_cache_, fi);

              Float w = region_term(radius, state.junc_distance_map_.at(vid_str));

              {
                if (!hard_region_condition_check_const(
                      state.graph_, state.region_size_cache_, fi,
                      state.junc_distance_map_.at(vid_str))) {
                  return false;
                }
              }

              std::string junc_info =
                "f" + std::to_string(fi) + ": " + std::to_string(radius) + "(" +
                std::to_string(radius / state.junc_distance_map_.at(vid_str)) + ")" +
                " = " + std::to_string(w) +
                "; p = " + std::to_string(state.candidates_[junc_i].probability);

              bool seen = false;
              for (const auto& w_info : vf_weights[std::make_pair(v_idx, fi)]) {
                if (w_info.junc_str == vid_str) {
                  seen = true;
                  break;
                }
              }

              if (!seen) {
                vf_weights[std::make_pair(v_idx, fi)].emplace_back(JunctionWeight{
                  w, vid_str, junc_info, state.junc_distance_map_.at(vid_str)});

                if (!junc2fi.count(vid_str))
                  junc2fi[vid_str] = std::set<size_t>();
                junc2fi[vid_str].emplace(fi);
              }
            } else if (fi == 0) {
              Float w = outside_term_ratio * region_term(1, 0);
              std::string junc_info =
                "f" + std::to_string(fi) + " = " + std::to_string(w) +
                "; p = " + std::to_string(state.candidates_[junc_i].probability);

              bool seen = false;
              for (const auto& w_info : vf_weights[std::make_pair(v_idx, fi)]) {
                if (w_info.junc_str == vid_str) {
                  seen = true;
                  break;
                }
              }

              if (!seen) {
                vf_weights[std::make_pair(v_idx, fi)].emplace_back(JunctionWeight{
                  w, vid_str, junc_info, state.junc_distance_map_.at(vid_str)});

                adj_outside_junctions.emplace(vid_str);
                if (!junc2fi.count(vid_str))
                  junc2fi[vid_str] = std::set<size_t>();
                junc2fi[vid_str].emplace(fi);
              }
            }
          }
        }

        it = it.next();
      } while (it != he);
    }
  }

  return true;
}

void find_junctions_in_face(const StrokeGraph& graph,
                            const std::vector<Junction>& candidates, const size_t fi,
                            std::vector<Junction>& in_face_junctions,
                            bool include_interior_junctions) {
  auto get_func = [&candidates](const std::string& vid_str, Junction& junc) -> bool {
    for (size_t i = 0; i < candidates.size(); ++i) {
      if (candidates[i].repr == vid_str) {
        junc = candidates[i];
        return true;
      }
    }

    return false;
  };

  in_face_junctions.reserve(candidates.size());
  std::unordered_set<std::string> seen_junctions;
  for (const auto hi : graph.faces_[fi].cycles_) {
    const auto he = graph.hedge(hi);
    auto it = he;
    do {
      // Dissolved junctions on edge
      for (const auto& [t, vid] : graph.strokes2vid_[it.stroke_idx()]) {
        auto vid_str = vid.repr();

        if (seen_junctions.count(vid_str))
          continue;
        seen_junctions.emplace(vid_str);

        Junction junc = candidates.front();
        bool found = get_func(vid_str, junc);
        assert(found);
        in_face_junctions.emplace_back(junc);
      }

      for (const auto& vid : it.origin().vertex_ids()) {
        auto vid_str = vid.repr();

        // Check if this junction connection is still active wrt the high-valence
        // junction cycle
        bool is_active = false;
        if (vid.connection_type_ == StrokeGraph::VertexID::Type::Junction) {
          for (size_t i = 0; i < candidates.size(); ++i) {
            if (candidates[i].repr == vid_str) {
              if (is_junction_on_cycle(it.origin(), candidates[i])) {
                is_active = true;
              }
              break;
            }
          }
        }

        // Init junctions
        if (is_active && !seen_junctions.count(vid_str) &&
            ((include_interior_junctions)
               ? is_high_valence_junction_in_region_cycle(graph, candidates, fi,
                                                          it.origin(), vid_str)
               : is_high_valence_junction_on_boundary_cycle(graph, candidates, fi,
                                                            it.origin(), vid_str))) {
          Junction junc = candidates.front();
          bool found = get_func(vid_str, junc);
          assert(found);
          in_face_junctions.emplace_back(junc);
        }
        seen_junctions.emplace(vid_str);
      }

      it = it.next();
    } while (it != he);
  }
}

} // namespace sketching
