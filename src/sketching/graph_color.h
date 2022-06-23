#pragma once

#include "types.h"

#include <unordered_set>

namespace sketching {

struct StrokeGraph;

/**
 * Assign indices (e.g. for a colour palette) to faces whilst trying to avoid assigning
 * the same index to adjacent faces.  The boundary face will be assigned an arbitrary
 * index.
 *
 * This algorithm is best effort.  It is recommended that you apply a small random offset
 * to the colour of each face in order to be able to distinguish adjacent or close regions
 * that are assigned the same swatch.
 *
 * This function needs O(max_colors) space.
 *
 * Returns true iff the colouring succeeded.
 */
bool map_color(const StrokeGraph& graph, int max_colors, span<int> out_coloring);

template <typename T>
struct Image {
  T* data_ = nullptr;
  Index width_ = 0;
  Index height_ = 0;
};

/**
 * Assign indices (e.g. for a colour palette) to labels whilst trying to avoid assigning
 * the same index to adjacent pixels with different labels.
 *
 * This function needs O(max_colors) space.
 *
 * @param label_img 2D array where the pixels in each region are assigned a unique number
 *                  in [0, n_regions).
 * @param max_colors Maximum number of colours to use.
 * @param out_coloring Size must correspond to the number of different regions in
 *                     `label_img`.
 * @return True iff the colouring succeeded.
 */
bool map_color_raster(Image<const std::int32_t> label_img, int max_colors,
                      span<int> out_coloring);

/**
 * Create a coloring of `new_graph` which is similar to the `ref_coloring` of `ref_graph`.
 * The boundary face will be assigned an arbitrary index.
 *
 * `max_colors` must be greater than all entries in ref_coloring except for the entry
 * corresponding to ref_graph.boundary_face_, which is ignored.
 *
 * This function needs O(max_colors) space.
 *
 * Returns true iff the coloring succeeded.  If the coloring fails, there may be adjacent
 * faces with the same color.
 */
bool color_by_reference(int max_colors, //
                        const StrokeGraph& ref_graph, span<const int> ref_coloring,
                        const StrokeGraph& new_graph, span<int> out_coloring);

bool color_by_reference_raster( //
  int max_colors, //
  const Image<const std::int32_t>& ref_label_img, span<const int> ref_coloring, //
  const Image<const std::int32_t>& new_label_img, span<int> out_coloring);

void compute_correspondence(const StrokeGraph& ref_graph, const size_t n_ref_faces,
                            const StrokeGraph& new_graph, const size_t n_new_faces,
                            std::vector<int>& new2ref, std::vector<int>& ref2new);
void compute_correspondence_raster(const Image<const std::int32_t>& ref_label_img,
                                   const size_t n_ref_faces,
                                   const Image<const std::int32_t>& new_label_img,
                                   const size_t n_new_faces, std::vector<int>& new2ref,
                                   std::vector<int>& ref2new);
int label_area(const Image<const std::int32_t>& label_img, const size_t fi,
               const size_t n_faces);

std::vector<std::unordered_set<size_t>> compute_connectivity(const StrokeGraph& graph);
std::vector<std::unordered_set<size_t>>
compute_connectivity(const Image<const int32_t>& label_img, const int n_labels);

} // namespace sketching
