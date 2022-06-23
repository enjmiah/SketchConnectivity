#include "pdf.h"

#include "../bounding_box.h"
#include "../classifier.h"
#include "../detail/alloca.h"
#include "../detail/suppress_warning.h"
#include "../force_assert.h"
#include "../graph_color.h"
#include "../stroke_graph_extra.h"
#include "error.h"
#include "file_guard.h"

#include <cstring>
#include <stb_sprintf.h>
#include <zlib.h>

namespace sketching {

namespace {

constexpr auto max_coord = int(1e5);
constexpr auto media_size = 1000;

const char* header =
  // clang-format off
  "%%PDF-1.4\x0A"
  "%%\xB7\xBE\xAD\xAA\x0A" // Signal binary content.
  "1 0 obj<</Type /Catalog /Pages 2 0 R>>endobj\x0A"
  "2 0 obj<</Type /Pages /Kids [3 0 R] /Count 1>>endobj\x0A"

  "3 0 obj<<"
  "/Type /Page "
  "/MediaBox [0 0 %d %.1f] " // Must fill this in.
  "/Contents 5 0 R "
  "/Resources <<"
    "/ExtGState<<"
      "/E1<</CA 0.4 /ca 0.4>> "
      "/E3<</CA 0.66 /ca 0.66>> "
    ">> "
    "/Pattern<</P1 4 0 R>> "
    "/Font <<"
      "/F1<<"
        "/Type /Font /Subtype /Type1 /Name /F1 "
        "/BaseFont /Helvetica /Encoding /MacRomanEncoding"
      ">> "
    ">> "
  ">> " // End of Resources
  "/Parent 2 0 R>>endobj\x0A"

  "4 0 obj<<\x0A"
    "/PatternType 1 /PaintType 1 /TilingType 2 "
    "/BBox [0 0 20 20] /XStep 20 /YStep 20 "
    "/Length 47 "
    "/Resources <<"
      "/Font <<"
        "/F1<<"
          "/Type /Font /Subtype /Type1 /Name /F1 "
          "/BaseFont /Helvetica /Encoding /MacRomanEncoding"
        ">> "
      ">> "
    ">>\x0A"
  ">>\x0A"
  "stream\x0A"
  "1 0 0 rg\x0A"
  "BT\x0A"
  "/F1 17 Tf\x0A"
  "(+) Tj\x0A"
  "10 10 TD\x0A"
  "(+) Tj\x0A"
  "ET\x0A"
  "endstream\x0A"
  "endobj\x0A" // End of 4 0 obj

  "5 0 obj<</Length 6 0 R %s>>\x0A" // Must add "/Filter [/FlateDecode]" if compressed.
  "stream\x0A"
  // clang-format on
  ;
constexpr uint8_t footer1[] = "endstream\x0A"
                              "endobj\x0A"
                              "6 0 obj ";
// Minus one to remove the null terminator.
constexpr auto footer1_len = sizeof(footer1) - 1;
constexpr uint8_t footer2[] = " endobj\x0A"
                              "xref\x0A"
                              "0 7\x0A"
                              "0000000000 65535 f\x0A"
                              "%010zu 00000 n\x0A"
                              "%010zu 00000 n\x0A"
                              "%010zu 00000 n\x0A"
                              "%010zu 00000 n\x0A"
                              "%010zu 00000 n\x0A"
                              "%010zu 00000 n\x0A"
                              "trailer<</Root 1 0 R /Size 6>>\x0A"
                              "startxref\x0A%zu\x0A"
                              "%%EOF";
constexpr auto footer2_len = sizeof(footer2) - 1;

/**
 * Usage: APPEND_PRINTF("format string %d %f %s", 1, 2.f, "foo");
 * Needs output_buffer and bytes_written to be in scope.
 * output_buffer needs `size` and `resize` methods and bytes_written should be a `size_t`.
 */
#define APPEND_PRINTF(...)                                                               \
  while (true) {                                                                         \
    const auto snpr_res =                                                                \
      stbsp_snprintf((char*)&output_buffer[bytes_written],                               \
                     int(output_buffer.size() - bytes_written), __VA_ARGS__);            \
    if (snpr_res < 0) {                                                                  \
      throw io_error("encoding error");                                                  \
    } else if (snpr_res >= int(output_buffer.size() - bytes_written)) {                  \
      output_buffer.resize(2 * output_buffer.size());                                    \
      /* Try again. */                                                                   \
    } else {                                                                             \
      bytes_written += snpr_res;                                                         \
      break;                                                                             \
    }                                                                                    \
  }

void plot_predictions(std::vector<uint8_t>& output_buffer, size_t& bytes_written,
                      const BoundingBox& bb,
                      const span<const ClassifierPrediction> in_predictions,
                      const PlotParams& params) {
  const auto bb_width = std::max(bb.width(), 1.0);

  using IndexAndClassifierPrediction = std::pair<int, ClassifierPrediction>;
  auto predictions = std::vector<IndexAndClassifierPrediction>(in_predictions.size());
  for (size_t i = 0; i < in_predictions.size(); ++i) {
    predictions[i] = {(int)i, in_predictions[i]};
  }
  std::sort(
    predictions.begin(), predictions.end(),
    [](const IndexAndClassifierPrediction& a, const IndexAndClassifierPrediction& b) {
      // Put end-stroke junctions first.
      if (a.second.key.type < b.second.key.type) {
        return true;
      } else if (b.second.key.type < a.second.key.type) {
        return false;
      }

      // Sort by location so that we can deduplicate.
      if (a.second.p_a < b.second.p_a) {
        return true;
      } else if (b.second.p_a < a.second.p_a) {
        return false;
      }
      if (a.second.p_b < b.second.p_b) {
        return true;
      } else if (b.second.p_b < a.second.p_b) {
        return false;
      }

      return a.second < b.second;
    });

  APPEND_PRINTF("1 J\x0A" // Round caps.
                "%.1f w\x0A", // Diameter.
                params.connection_width);

  const auto disconnected_color =
    params.disconnection_color; //  Col3::from_hex_rgb(0x981ceb);
  constexpr auto end_stroke_color = Col3::from_hex_rgb(0x25f1f5);
  constexpr auto end_end_color = Col3::from_hex_rgb(0xf54514);

  APPEND_PRINTF("%.3f %.3f %.3f RG\x0A", // Stroke color.
                disconnected_color.r, disconnected_color.g, disconnected_color.b);

  for (size_t i = 0; i < predictions.size(); ++i) {
    const auto& pred = predictions[i].second;
    if (pred.connected) {
      continue;
    }
    const auto x1 = (int)std::round((pred.p_a.x_ - bb.xMin_) * max_coord / bb_width);
    const auto y1 = (int)std::round((bb.yMax_ - pred.p_a.y_) * max_coord / bb_width);
    const auto x2 = (int)std::round((pred.p_b.x_ - bb.xMin_) * max_coord / bb_width);
    const auto y2 = (int)std::round((bb.yMax_ - pred.p_b.y_) * max_coord / bb_width);
    APPEND_PRINTF("%d %d m\x0A%d %d l\x0A", x1, y1, x2, y2);
  }

  APPEND_PRINTF("S\x0A" // Stroke command
                "%.3f %.3f %.3f RG\x0A", // Stroke color.
                end_stroke_color.r, end_stroke_color.g, end_stroke_color.b);

  for (size_t i = 0; i < predictions.size(); ++i) {
    const auto& pred = predictions[i].second;
    if (pred.key.type == JunctionType::R || !pred.connected) {
      continue;
    }
    const auto x1 = (int)std::round((pred.p_a.x_ - bb.xMin_) * max_coord / bb_width);
    const auto y1 = (int)std::round((bb.yMax_ - pred.p_a.y_) * max_coord / bb_width);
    const auto x2 = (int)std::round((pred.p_b.x_ - bb.xMin_) * max_coord / bb_width);
    const auto y2 = (int)std::round((bb.yMax_ - pred.p_b.y_) * max_coord / bb_width);
    APPEND_PRINTF("%d %d m\x0A%d %d l\x0A", x1, y1, x2, y2);
  }
  APPEND_PRINTF("S\x0A" // Stroke command
                "%.3f %.3f %.3f RG\x0A", // Stroke color.
                end_end_color.r, end_end_color.g, end_end_color.b);

  for (size_t i = 0; i < predictions.size(); ++i) {
    const auto& pred = predictions[i].second;
    if (pred.key.type == JunctionType::T || !pred.connected) {
      continue;
    }
    const auto x1 = (int)std::round((pred.p_a.x_ - bb.xMin_) * max_coord / bb_width);
    const auto y1 = (int)std::round((bb.yMax_ - pred.p_a.y_) * max_coord / bb_width);
    const auto x2 = (int)std::round((pred.p_b.x_ - bb.xMin_) * max_coord / bb_width);
    const auto y2 = (int)std::round((bb.yMax_ - pred.p_b.y_) * max_coord / bb_width);
    APPEND_PRINTF("%d %d m\x0A%d %d l\x0A", x1, y1, x2, y2);
  }

  constexpr auto font_size = int(0.004 * max_coord);
  APPEND_PRINTF("S\x0A" // Stroke command.

                // Draw text labels.
                "0 0 0 rg\x0A" // Text color.
                "BT /F1 %d Tf\x0A" // Font.
                "%d Tc\x0A", // Letter spacing.
                font_size, int(-0.05 * font_size));

  size_t duplicate_range_start = 0;
  for (size_t i = 0; i < predictions.size(); ++i) {
    if (i + 1 < predictions.size() &&
        predictions[i].second.p_a.isApprox(predictions[i + 1].second.p_a) &&
        predictions[i].second.p_b.isApprox(predictions[i + 1].second.p_b)) {
      continue; // Avoid putting two labels directly on top of each other.
    }

    const auto& pred = predictions[i].second;
    if (pred.prob >= 0) {
      const auto xy = 0.5 * (pred.p_a + pred.p_b);
      auto label_ss = std::stringstream();
      Float seen_prob = -1;
      for (size_t j = duplicate_range_start; j <= i; ++j) {
        if (seen_prob == predictions[j].second.prob)
          continue;
        seen_prob = predictions[j].second.prob;
        if (j != duplicate_range_start)
          label_ss << ';';

        std::string alt_prob_str = "";
        if (predictions[j].second.alt_prob >= 0)
          alt_prob_str =
            "," + std::to_string((int)std::round(100 * predictions[j].second.alt_prob)) +
            "%";
        if (params.show_prediction_index) {
          label_ss << predictions[j].first << " ("
                   << (int)std::round(100 * predictions[j].second.prob) << "%"
                   << alt_prob_str << ")";
        } else {
          label_ss << (int)std::round(100 * predictions[j].second.prob) << "%"
                   << alt_prob_str;
        }
      }
      // TODO: Compute this dynamically based on the font metrics.
      auto label_width =
        (params.show_prediction_index ? int(3.4 * font_size) : int(1.9 * font_size));
      label_width *= int(i - duplicate_range_start) + 1;
      const auto label_height = font_size;
      const auto x =
        (int)std::round((xy.x_ - bb.xMin_) * max_coord / bb_width - 0.5 * label_width);
      const auto y =
        (int)std::round((bb.yMax_ - xy.y_) * max_coord / bb_width - 0.3 * label_height);
      APPEND_PRINTF("1 0 0 1 %d %d Tm (%s) Tj\x0A", //
                    x, y, label_ss.str().c_str());
    }

    duplicate_range_start = i + 1;
  }
  APPEND_PRINTF("ET\x0A");
}

void plot_envelopes(std::vector<uint8_t>& output_buffer, size_t& bytes_written,
                    const BoundingBox& bb, const span<const Stroke> strokes,
                    const PlotParams& params) {
  const auto bb_width = std::max(bb.width(), 1.0);

  // Write the envelope fill colour.
  APPEND_PRINTF("%f %f %f rg\x0A", //
                std::clamp(params.envelope_fill.r, 0.0, 1.0),
                std::clamp(params.envelope_fill.g, 0.0, 1.0),
                std::clamp(params.envelope_fill.b, 0.0, 1.0));

  auto coords = std::vector<CoordMat>();
  for (const auto& stroke : strokes) {
    outline_to_polygons(stroke, coords);
  }
  for (auto& mat : coords) {
    mat.col(0).array() -= bb.xMin_;
    mat.col(0).array() *= max_coord / bb_width;
    mat.col(1).array() *= -1.0;
    mat.col(1).array() += bb.yMax_;
    mat.col(1).array() *= max_coord / bb_width;
    const auto rows = mat.rows();
    for (Index i = 0; i < rows; ++i) {
      APPEND_PRINTF("%d %d %c\x0A", (int)std::round(mat(i, 0)),
                    (int)std::round(mat(i, 1)), i == 0 ? 'm' : 'l');
    }
  }

  APPEND_PRINTF("f\x0A");
}

/// Returns the length of the compressed stream in bytes.
std::vector<uint8_t> compress_stream(const span<const uint8_t> input) {
  auto stream = z_stream();
  stream.zalloc = Z_NULL;
  stream.zfree = Z_NULL;
  stream.opaque = Z_NULL;
  stream.avail_in = (uInt)input.size();
  stream.next_in = (Bytef*)&input[0]; // Const cast is safe; zlib doesn't modify input.
  stream.avail_out = 0;
  stream.next_out = nullptr;
  if (deflateInit(&stream, 4) != Z_OK) {
    std::abort(); // Failed to initialize zlib stream.
  }
  const auto bound = deflateBound(&stream, (uLong)input.size());
  auto output = std::vector<uint8_t>();
  output.resize(bound);
  stream.avail_out = bound;
  stream.next_out = &output[0];
  if (deflate(&stream, Z_FINISH) != Z_STREAM_END) {
    std::abort(); // Failed to compress buffer.
  }
  output.resize(stream.total_out);
  if (deflateEnd(&stream) != Z_OK) {
    std::abort(); // Failed to free zlib data structures.
  }
  return output;
}

} // namespace

PlotParams::PlotParams()
  : envelope_fill(Col3::from_hex_rgb(0x444444))
  , compress(true)
  , disconnection_color(Col3::from_hex_rgb(0xf6d55c)) {}

void save_pdf(const StrokeGraph& graph, const std::string& path, const PlotParams& params,
              const span<const ClassifierPrediction> predictions,
              span<const int> face_colors, const span<const size_t> highlighted_faces) {
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) {
    throw io_error("could not open file for writing");
  }
  auto guard = FileGuard(f);

  auto output_buffer = std::vector<uint8_t>();
  // Reserve enough space to safely write the PDF boilerplate.  We will reserve more as
  // needed when writing the actual polygons.
  output_buffer.resize(262144); // 0.25 MiB
  auto bytes_written = size_t(0);

  auto bb = params.media_box;
  if (bb.isEmpty()) {
    bb = visual_bounds(graph.bvh_.strokes());
  }
  const auto bb_width = std::max(bb.width(), 1.0);
  APPEND_PRINTF(header, media_size,
                (bb_width > 0 ? media_size * bb.height() / bb_width : 3),
                params.compress ? "/Filter [/FlateDecode]" : "");

  const auto stream_start = bytes_written;

  // Write a transform (uniform scale).
  // This lets us avoid writing decimal points by writing integer coordinates only.
  APPEND_PRINTF("%f 0 0 %f 0 0 cm\x0A", //
                Float(media_size) / max_coord, Float(media_size) / max_coord);

  if (params.viz_envelopes) {
    plot_envelopes(output_buffer, bytes_written, bb, graph.strokes_, params);
  }

  // Write the faces.
  if (params.viz_faces) {
    const auto palette = get_color_palette();

    // Set the graphics state to use partially transparent fills.
    APPEND_PRINTF("q\x0A"); // Save graphics state.
    if (!params.opaque_faces) {
      APPEND_PRINTF("/E1 gs\x0A"); // Partially transparent.
    }

    if (face_colors.empty()) {
      const auto face_colors_buf = ALLOCA_SPAN(int, graph.faces_.size());
      const auto ok = map_color(graph, (int)palette.size(), face_colors_buf);
      if (!ok) {
        SPDLOG_WARN("Map colouring failed. Some adjacent regions will be assigned the "
                    "same colour.");
      }
      face_colors = face_colors_buf; // span<int> -> span<const int>
    } else {
      force_assert(face_colors.size() == graph.faces_.size());
    }
    auto vertices = std::vector<Vec2>();
    for (size_t fi = 1; fi < graph.faces_.size(); ++fi) {
      for (const auto& hi : graph.faces_[fi].cycles_) {
        const auto he = graph.hedge(hi);
        cycle_positions(he, vertices);
        const auto n = vertices.size();
        for (size_t i = 0; i < n; ++i) {
          const auto x =
            (int)std::round((vertices[i].x_ - bb.xMin_) * max_coord / bb_width);
          const auto y =
            (int)std::round((bb.yMax_ - vertices[i].y_) * max_coord / bb_width);
          APPEND_PRINTF("%d %d %c\x0A", x, y, i == 0 ? 'm' : 'l');
        }
        vertices.clear();
      }

      assert(face_colors[fi] < (int)palette.size());
      const auto& color = palette[face_colors[fi]];
      APPEND_PRINTF("%.2f %.2f %.2f rg\x0A" // Set fill color.
                    "f\x0A", // Fill command.
                    color.r, color.g, color.b);
    }

    APPEND_PRINTF("Q\x0A"); // Restore graphics state.
  }

  if (!highlighted_faces.empty()) {
    APPEND_PRINTF("q\x0A" // Save graphics state.
                  "/Pattern cs\x0A" // Switch to pattern color space.
                  "/P1 scn\x0A"); // Switch to pattern fill.
    auto vertices = std::vector<Vec2>();
    for (const auto fi : highlighted_faces) {
      for (const auto hi : graph.faces_[fi].cycles_) {
        const auto he = graph.hedge(hi);
        cycle_positions(he, vertices);
        const auto n = vertices.size();
        for (size_t i = 0; i < n; ++i) {
          const auto x =
            (int)std::round((vertices[i].x_ - bb.xMin_) * max_coord / bb_width);
          const auto y =
            (int)std::round((bb.yMax_ - vertices[i].y_) * max_coord / bb_width);
          APPEND_PRINTF("%d %d %c\x0A", x, y, i == 0 ? 'm' : 'l');
        }
        vertices.clear();
      }
    }
    APPEND_PRINTF("f*\x0A" // Fill using even-odd rule.
                  "Q\x0A"); // Restore graphics state.
  } // if (!highlighted_faces.empty())

  // Draw the endpoints defined by predictions
  if (params.viz_ends) {
    // We draw circles as a length 0 line segment with round caps.
    APPEND_PRINTF("q\x0A" // Save graphics state.
                  "/E3 gs\x0A" // Transparent graphics state.
                  "1 J\x0A" // Round caps.
                  "0.49 0.92 0.03 RG\x0A" // Stroke color.
                  "1000 w\x0A" // Diameter.
    );

    // Write the endpoint.
    std::vector<ClassifierPrediction> real_predictions;
    real_predictions.reserve(predictions.size());
    for (size_t i = 0; i < predictions.size(); ++i) {
      if (predictions[i].prob >= 0) {
        real_predictions.emplace_back(predictions[i]);
        continue;
      }
      for (const auto& [p, is_end] :
           {std::make_pair(predictions[i].p_a, predictions[i].key.cand1),
            std::make_pair(predictions[i].p_b, predictions[i].key.cand2)}) {
        if (is_end < 0)
          continue;
        const auto x = (int)std::round((p.x_ - bb.xMin_) * max_coord / bb_width);
        const auto y = (int)std::round((bb.yMax_ - p.y_) * max_coord / bb_width);
        APPEND_PRINTF("%d %d m\x0A%d %d l\x0A", x, y, x, y);
      }
    }

    APPEND_PRINTF("S\x0A" // Write the stroke command.
                  "Q\x0A" // Restore graphics state.
    );

    plot_predictions(output_buffer, bytes_written, bb, real_predictions, params);
  } else {
    plot_predictions(output_buffer, bytes_written, bb, predictions, params);
  }

  if (params.viz_dangling) {
    // We draw circles as a length 0 line segment with round caps.
    APPEND_PRINTF("q\x0A" // Save graphics state.
                  "/E3 gs\x0A" // Transparent graphics state.
                  "1 J\x0A" // Round caps.
                  "0.49 0.92 0.03 RG\x0A" // Stroke color.
                  "3000 w\x0A" // Diameter.
    );

    // Write the endpoint.
    for (size_t vi = 0; vi < graph.vertices_.size(); ++vi) {
      const auto v = graph.vertex(vi);
      if (v.is_active() && v.is_dangling()) {
        const auto& p = graph.vertices_[vi].p_;
        const auto x = (int)std::round((p.x_ - bb.xMin_) * max_coord / bb_width);
        const auto y = (int)std::round((bb.yMax_ - p.y_) * max_coord / bb_width);
        APPEND_PRINTF("%d %d m\x0A%d %d l\x0A", x, y, x, y);
      }
    }

    APPEND_PRINTF("S\x0A" // Write the stroke command.
                  "Q\x0A" // Restore graphics state.
    );
  }

  // Write the centerlines.
  if (params.viz_centerlines) {
    APPEND_PRINTF("1 j\x0A" // Round joins.
                  "40 w\x0A" // Line width.
    );

    for (size_t si = 0; si < graph.strokes_.size(); ++si) {
      const auto& stroke = graph.strokes_[si];
      const auto n = stroke.size();
      if (!(graph.hedge(2 * si).flags() & StrokeGraph::HedgeRecord::Bridge)) {
        for (Index i = 0; i < n; ++i) {
          const auto x = (int)std::round((stroke.x(i) - bb.xMin_) * max_coord / bb_width);
          const auto y = (int)std::round((bb.yMax_ - stroke.y(i)) * max_coord / bb_width);
          APPEND_PRINTF("%d %d %c\x0A", x, y, i == 0 ? 'm' : 'l');
        }
      }
    }
    APPEND_PRINTF("%.2f %.2f %.2f RG\x0A" // Set centerline color.
                  "S\x0A", // Finish the centerlines by writing the stroke command.
                  params.centerline_color.r, params.centerline_color.g,
                  params.centerline_color.b);

    // Write the bridges.
    for (size_t si = 0; si < graph.strokes_.size(); ++si) {
      const auto& stroke = graph.strokes_[si];
      const auto n = stroke.size();
      if (graph.hedge(2 * si).flags() & StrokeGraph::HedgeRecord::Bridge) {
        for (Index i = 0; i < n; ++i) {
          const auto x = (int)std::round((stroke.x(i) - bb.xMin_) * max_coord / bb_width);
          const auto y = (int)std::round((bb.yMax_ - stroke.y(i)) * max_coord / bb_width);
          APPEND_PRINTF("%d %d %c\x0A", x, y, i == 0 ? 'm' : 'l');
        }
      }
    }
    const auto bridge_color = Col3::from_hex_rgb(0xff00dd);
    APPEND_PRINTF("%.2f %.2f %.2f RG\x0A" // Set centerline color.
                  "S\x0A", // Finish the centerlines by writing the stroke command.
                  bridge_color.r, bridge_color.g, bridge_color.b);

    // Set up vertex drawing.
    // We draw vertices as a length 0 line segment with round caps.
    APPEND_PRINTF("1 J\x0A" // Round caps.
                  "1 0.8 0 RG\x0A" // Stroke color.
                  "80 w\x0A" // Diameter.
    );

    // Write the vertices.
    for (size_t vi = 0; vi < graph.vertices_.size(); ++vi) {
      const auto v = graph.vertex(vi);
      if (v.is_active() && !v.is_dangling()) {
        const auto& p = graph.vertices_[vi].p_;
        const auto x = (int)std::round((p.x_ - bb.xMin_) * max_coord / bb_width);
        const auto y = (int)std::round((bb.yMax_ - p.y_) * max_coord / bb_width);
        APPEND_PRINTF("%d %d m\x0A%d %d l\x0A", x, y, x, y);
      }
    }

    // Finish the vertices by writing the stroke command.
    APPEND_PRINTF("S\x0A");
  }

  auto stream_length = bytes_written - stream_start;

  const auto footer_margin = size_t(256);

  if (params.compress) {
    const auto compressed =
      compress_stream({&output_buffer[stream_start], stream_length});
    if (output_buffer.size() - stream_start < compressed.size()) {
      output_buffer.resize(output_buffer.size() + compressed.size() + footer_margin);
    }
    memcpy(&output_buffer[stream_start], &compressed[0], compressed.size());
    stream_length = compressed.size();
    bytes_written = stream_start + stream_length;
  }

  // Ensure we have enough space to finish the file.
  if (output_buffer.size() - bytes_written < footer_margin) {
    output_buffer.resize(output_buffer.size() + footer_margin);
  }

  const auto xref5_offset =
    bytes_written + (strstr((const char*)footer1, "6 0 obj") - (const char*)footer1);
  memcpy(&output_buffer[bytes_written], footer1, footer1_len);
  bytes_written += footer1_len;

  APPEND_PRINTF("%zu", stream_length);

  const auto startxref_offset =
    bytes_written + ((uint8_t*)strstr((const char*)footer2, "xref") - footer2);
  APPEND_PRINTF(
    (const char*)footer2, //
    (uint8_t*)strstr((const char*)&output_buffer[0], "1 0 obj") - &output_buffer[0],
    (uint8_t*)strstr((const char*)&output_buffer[0], "2 0 obj") - &output_buffer[0],
    (uint8_t*)strstr((const char*)&output_buffer[0], "3 0 obj") - &output_buffer[0],
    (uint8_t*)strstr((const char*)&output_buffer[0], "4 0 obj") - &output_buffer[0],
    (uint8_t*)strstr((const char*)&output_buffer[0], "5 0 obj") - &output_buffer[0],
    xref5_offset, startxref_offset);

  const auto actual_bytes_written =
    fwrite(output_buffer.data(), sizeof(uint8_t), bytes_written, f);
  if (actual_bytes_written != bytes_written) {
    throw io_error("failed to write file");
  }
}

void save_pdf(const span<const Stroke> strokes, const std::string& path,
              const PlotParams& params,
              const span<const ClassifierPrediction> predictions) {
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) {
    throw io_error("could not open file for writing");
  }
  auto guard = FileGuard(f);

  auto output_buffer = std::vector<uint8_t>();
  // Reserve enough space to safely write the PDF boilerplate.  We will reserve more as
  // needed when writing the actual polygons.
  output_buffer.resize(262144); // 0.25 MiB
  auto bytes_written = size_t(0);

  auto bb = params.media_box;
  if (bb.isEmpty()) {
    bb = visual_bounds(strokes);
  }
  const auto bb_width = std::max(bb.width(), 1.0);
  APPEND_PRINTF(header, media_size,
                (bb_width > 0 ? media_size * bb.height() / bb_width : 3),
                params.compress ? "/Filter [/FlateDecode]" : "");

  const auto stream_start = bytes_written;

  // Write a transform (uniform scale).
  // This lets us avoid writing decimal points by writing integer coordinates only.
  APPEND_PRINTF("%f 0 0 %f 0 0 cm\x0A", //
                Float(media_size) / max_coord, Float(media_size) / max_coord);

  if (params.viz_envelopes) {
    plot_envelopes(output_buffer, bytes_written, bb, strokes, params);
  }

  if (params.viz_ends) {
    // We draw circles as a length 0 line segment with round caps.
    APPEND_PRINTF("q\x0A" // Save graphics state.
                  "/E3 gs\x0A" // Transparent graphics state.
                  "1 J\x0A" // Round caps.
                  "0.49 0.92 0.03 RG\x0A" // Stroke color.
                  "1000 w\x0A" // Diameter.
    );

    // Write the endpoint.
    for (const auto& stroke : strokes) {
      for (const auto p : {stroke.xy(0), stroke.xy(Back)}) {
        const auto x = (int)std::round((p.x_ - bb.xMin_) * max_coord / bb_width);
        const auto y = (int)std::round((bb.yMax_ - p.y_) * max_coord / bb_width);
        APPEND_PRINTF("%d %d m\x0A%d %d l\x0A", x, y, x, y);
      }
    }

    APPEND_PRINTF("S\x0A" // Write the stroke command.
                  "Q\x0A" // Restore graphics state.
    );
  }

  APPEND_PRINTF("1 j\x0A" // Round joins.
                "40 w\x0A" // Line width.
  );

  if (params.viz_centerlines) {
    // Write the centerlines.
    for (const auto& stroke : strokes) {
      const auto n = stroke.size();
      for (Index i = 0; i < n; ++i) {
        const auto x = (int)std::round((stroke.x(i) - bb.xMin_) * max_coord / bb_width);
        const auto y = (int)std::round((bb.yMax_ - stroke.y(i)) * max_coord / bb_width);
        APPEND_PRINTF("%d %d %c\x0A", x, y, i == 0 ? 'm' : 'l');
      }
    }
    // Finish the centerlines by writing the stroke command.
    APPEND_PRINTF("%.2f %.2f %.2f RG\x0A" // Set centerline color.
                  "S\x0A", // Finish the centerlines by writing the stroke command.
                  params.centerline_color.r, params.centerline_color.g,
                  params.centerline_color.b);
  }

  plot_predictions(output_buffer, bytes_written, bb, predictions, params);

  auto stream_length = bytes_written - stream_start;
  const auto footer_margin = size_t(256);

  if (params.compress) {
    const auto compressed =
      compress_stream({&output_buffer[stream_start], stream_length});
    if (output_buffer.size() - stream_start < compressed.size()) {
      output_buffer.resize(output_buffer.size() + compressed.size() + footer_margin);
    }
    memcpy(&output_buffer[stream_start], &compressed[0], compressed.size());
    stream_length = compressed.size();
    bytes_written = stream_start + stream_length;
  }

  // Ensure we have enough space to finish the file.
  if (output_buffer.size() - bytes_written < footer_margin) {
    output_buffer.resize(output_buffer.size() + footer_margin);
  }

  const auto xref5_offset =
    bytes_written + (strstr((const char*)footer1, "6 0 obj") - (const char*)footer1);
  memcpy(&output_buffer[bytes_written], footer1, footer1_len);
  bytes_written += footer1_len;

  APPEND_PRINTF("%zu", stream_length);

  const auto startxref_offset =
    bytes_written + ((uint8_t*)strstr((const char*)footer2, "xref") - footer2);
  APPEND_PRINTF(
    (const char*)footer2, //
    (uint8_t*)strstr((const char*)&output_buffer[0], "1 0 obj") - &output_buffer[0],
    (uint8_t*)strstr((const char*)&output_buffer[0], "2 0 obj") - &output_buffer[0],
    (uint8_t*)strstr((const char*)&output_buffer[0], "3 0 obj") - &output_buffer[0],
    (uint8_t*)strstr((const char*)&output_buffer[0], "4 0 obj") - &output_buffer[0],
    (uint8_t*)strstr((const char*)&output_buffer[0], "5 0 obj") - &output_buffer[0],
    xref5_offset, startxref_offset);

  const auto actual_bytes_written =
    fwrite(output_buffer.data(), sizeof(uint8_t), bytes_written, f);
  if (actual_bytes_written != bytes_written) {
    throw io_error("failed to write file");
  }
}

void save_pdf(const Stroke& stroke, const std::string& path, const PlotParams& params,
              const span<const ClassifierPrediction> predictions) {
  save_pdf({&stroke, 1}, path, params, predictions);
}

} // namespace sketching
