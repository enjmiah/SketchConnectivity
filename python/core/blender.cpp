#include "drawing.h"

#include <sketching/types.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;
using namespace sketching;

namespace {

constexpr auto invalid_time = 9999.0;

/**
 * The `strokes` argument is NOT validated; please validate it from the Python side first.
 */
void drawing_add_gpencil(Drawing& drawing, py::object& strokes,
                         const Eigen::Matrix<Float, 3, 4>& transform,
                         const Float pixel_factor,
                         const std::string& stroke_thickness_space) {
  const auto worldspace = (stroke_thickness_space == "WORLDSPACE");
  for (auto s : strokes) {
    auto points = s.attr("points");
    const auto npoints = py::len(points);
    if (npoints > 1) {
      const auto has_time = py::hasattr(s, "init_time_s_hi");
      auto& stroke = drawing.mut_strokes().emplace_back(npoints, has_time);

      const auto pen_width = 2 * /* radius */ s.attr("line_width").cast<Float>();
      auto init_time = 0.0;
      if (has_time) {
        const auto seconds =
          static_cast<double>((s.attr("init_time_s_hi").cast<std::uint64_t>() << 32) +
                              s.attr("init_time_s_lo").cast<std::uint64_t>());
        init_time = seconds + 0.000000001 * s.attr("init_time_ns").cast<std::uint32_t>();
      }

      auto i = 0u;
      for (auto point : points) {
        const auto& co = point.attr("co");
        const auto world_coords =
          Eigen::Vector4d(co.attr("x").cast<Float>(), co.attr("y").cast<Float>(),
                          co.attr("z").cast<Float>(), 1.0);
        const auto xyw = transform * world_coords;
        auto width = point.attr("pressure").cast<Float>() * pen_width * pixel_factor;
        if (worldspace)
          width /= xyw.z();
        stroke.x(i) = xyw.x() / xyw.z();
        stroke.y(i) = xyw.y() / xyw.z();
        stroke.width(i) = width;
        if (has_time) {
          auto time = point.attr("time").cast<Float>();
          if (time != invalid_time) { // NOLINT(clang-diagnostic-float-equal)
            time += init_time;
          }
          stroke.time(i) = time;
        }
        ++i;
      }
    }
  }
}

void fill_in_empty_time_values(py::EigenDRef<Eigen::VectorXd>& time) {
  const auto npoints = time.rows();
  Eigen::Index range_start = -1;
  Eigen::Index range_end = -1;
  for (auto i = 0; i < npoints + 1; ++i) {
    if (i < npoints && time(i) == invalid_time) { // NOLINT(clang-diagnostic-float-equal)
      if (range_start == -1) {
        range_start = i;
        if (range_start == 0) {
          throw std::logic_error("First time point must be valid");
        }
      }
    } else {
      if (range_start != -1) {
        range_end = i;

        // Do the fill in from [range_start, range_end).
        if (range_end == npoints) {
          // Trickier case where the invalid values stretch to the end.
          if (range_start == 1) {
            // Only one point?  Can't extrapolate from that.
            for (auto j = range_start; j < range_end; ++j) {
              time(j) = time(range_start - 1);
            }
          } else {
            // Move `time(range_start - 1)` to the end of the empty range, and interpolate
            // that.
            range_end = npoints - 1;
            time(range_end) = time(range_start - 1);
            range_start--;
            const double a = time(range_start - 1);
            const double b = time(range_end);
            // Linearly interpolate from a to b.
            const auto denom = static_cast<double>(range_end - range_start + 1);
            const auto len = range_end - range_start;
            for (auto j = 0; j < len; ++j) {
              time(j + range_start) = ((len - j) * a + (j + 1) * b) / denom;
            }
          }
        } else {
          const double a = time(range_start - 1);
          const double b = time(range_end);
          // Linearly interpolate from a to b.
          const double denom = static_cast<double>(range_end - range_start + 1);
          const auto len = range_end - range_start;
          for (auto j = 0; j < len; ++j) {
            time(j + range_start) = ((len - j) * a + (j + 1) * b) / denom;
          }
        }

        range_start = range_end = -1; // reset
      }
    }
  }
}

} // namespace

void init_blender(py::module& m) {
  m.def("drawing_add_gpencil", &drawing_add_gpencil, "drawing"_a, "gpencil"_a,
        "transform"_a, "pixel_factor"_a, "stroke_thickness_space"_a);
  m.def("fill_in_empty_time_values", &fill_in_empty_time_values, "time_points"_a);
}
