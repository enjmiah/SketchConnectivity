#include "svg.h"

#include "../bounding_box.h"
#include "../busyness.h"
#include "../sketching.h"
#include "error.h"
#include "file_guard.h"

#include <stb_sprintf.h>

namespace sketching {

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

static const char* const header = //
  "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
  "<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\" "
  "\"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n"
  "<svg viewBox=\"%.3f %.3f %.3f %.3f\" xmlns=\"http://www.w3.org/2000/svg\" "
  "version=\"1.1\">\n";

void save_svg(const span<const Stroke> strokes, const std::string& path) {
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) {
    throw io_error("could not open file for writing");
  }
  auto guard = FileGuard(f);

  auto output_buffer = std::vector<uint8_t>();
  output_buffer.resize(262144); // 0.25 MiB
  auto bytes_written = size_t(0);

  const auto media_box = visual_bounds(strokes);

  APPEND_PRINTF(header, media_box.xMin_, media_box.yMin_, media_box.width(),
                media_box.height());

  for (const auto& stroke : strokes) {
    APPEND_PRINTF("<path d=\"");
    for (Index i = 0; i < stroke.size(); ++i) {
      APPEND_PRINTF("%s%.5f %.5f", i == 0 ? "M" : " L", stroke.x(i), stroke.y(i));
    }
    APPEND_PRINTF("\" data-width=\"");
    for (Index i = 0; i < stroke.size(); ++i) {
      APPEND_PRINTF("%s%.5f", i == 0 ? "" : " ", stroke.width(i));
    }
    APPEND_PRINTF("\" fill=\"none\" stroke=\"black\" stroke-width=\"%.3f\" "
                  "stroke-linecap=\"round\" stroke-linejoin=\"round\" />\n",
                  average_width(stroke));
  }

  APPEND_PRINTF("</svg>\n");

  const auto actual_bytes_written =
    fwrite(output_buffer.data(), sizeof(uint8_t), bytes_written, f);
  if (actual_bytes_written != bytes_written) {
    throw io_error("failed to write file");
  }
}

} // namespace sketching
