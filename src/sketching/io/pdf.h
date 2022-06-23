#pragma once

#include "../bounding_box.h"
#include "../render.h"
#include "../types.h"

namespace sketching {

struct ClassifierPrediction;
struct StrokeGraph;

struct PlotParams {
  PlotParams();

  Col3 envelope_fill; ///< Colour from 0x000000 to 0xFFFFFF.

  Col3 centerline_color; ///< Colour from 0x000000 to 0xFFFFFF.

  Col3 disconnection_color; ///< Colour from 0x000000 to 0xFFFFFF.

  BoundingBox media_box;

  /// Whether to compress streams where possible.  Turn off for maximum plotting speed.
  bool compress;

  /// Whether to draw centerlines and vertices when plotting a drawing
  bool viz_centerlines = true;

  /// Whether to draw stroke envelopes when plotting a drawing
  bool viz_envelopes = true;

  /// Whether to color in faces when plotting a drawing.
  bool viz_faces = true;

  /// Whether to draw faces at full opacity.
  bool opaque_faces = false;

  /// Whether to highlight endpoints in predictions (flagged with negative probability)
  /// when plotting a drawing.
  bool viz_ends = false;

  /// Whether to highlight dangling endpoints when plotting a drawing.
  bool viz_dangling = false;

  /// Whether prediction labels show as e.g. "42 (50%)" or just "50%".
  bool show_prediction_index = true;

  /// Width of prediction/connection lines.
  Float connection_width = 200;
};

/**
 * No entry in face_colors may be greater than or equal to the value of
 * `get_color_palette().size()` except for the entry corresponding to the boundary face,
 * which is ignored.
 */
void save_pdf(const StrokeGraph& graph, const std::string& path, const PlotParams& params,
              span<const ClassifierPrediction> predictions = {nullptr, 0},
              span<const int> face_colors = {nullptr, 0},
              span<const size_t> highlighted_faces = {nullptr, 0});

void save_pdf(span<const Stroke> strokes, const std::string& path,
              const PlotParams& params,
              span<const ClassifierPrediction> predictions = {nullptr, 0});

void save_pdf(const Stroke& stroke, const std::string& path, const PlotParams& params,
              span<const ClassifierPrediction> predictions = {nullptr, 0});

} // namespace sketching
