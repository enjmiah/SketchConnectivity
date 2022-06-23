#pragma once

#include "../types.h"

#include <unordered_set>
#include <vector>

namespace sketching {

struct Stroke;
struct StrokeGraph;

void fit_width_cluster(const std::vector<const Stroke*>& strokes, Float accuracy,
                       bool cut_strokes, Stroke& strokefit,
                       bool use_default_sampling = false);

void consolidate_with_chaining(span<const Stroke> strokes,
                               std::vector<Stroke>& out_consolidated_strokes);

void consolidate_with_chaining_improved(span<const Stroke> strokes,
                                        std::vector<Stroke>& out_consolidated_strokes);

void consolidate_with_chaining(span<const Stroke> strokes,
                               std::vector<Stroke>& out_consolidated_strokes,
                               std::vector<std::unordered_set<size_t>>& final_clusters);
} // namespace sketching
