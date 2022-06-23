// clang-format off
/**
 * Define automatic conversions between C++ types and Python types.
 *
 * Include this file if you get an error like
 *
 *     TypeError: Unable to convert function return value to a Python type! The signature was
 *         (...) -> sketching::Vec2
 */
// clang-format on

#pragma once

#include <sketching/types.h>

#include <pybind11/pybind11.h>

// "Polluting" the global namespace should be OK since this header only gets included by
// files that would include such typedefs anyway.
namespace py = pybind11;
using namespace pybind11::literals;

namespace pybind11::detail {

template <>
struct type_caster<sketching::Vec2> {
public:
  PYBIND11_TYPE_CASTER(sketching::Vec2, _("Vec2"));

  // Python -> C++
  bool load(py::handle src, bool convert) {
    if (!convert && !py::array_t<sketching::Float>::check_(src))
      return false;

    auto buf =
      py::array_t<sketching::Float, py::array::c_style | py::array::forcecast>::ensure(
        src);
    if (!buf || buf.ndim() != 1 || buf.size() != 2)
      return false;

    value = sketching::Vec2(buf.data()[0], buf.data()[1]);

    return true;
  }

  // C++ -> Python
  static py::handle cast(const sketching::Vec2& src, py::return_value_policy /*policy*/,
                         py::handle /*parent*/) {
    py::array a({2}, {sizeof(sketching::Float)}, &src.x_);
    return a.release();
  }
};

template <>
struct type_caster<sketching::span<sketching::Float>> {
public:
  PYBIND11_TYPE_CASTER(sketching::span<sketching::Float>, _("FloatSpan"));

  // Python -> C++
  bool load(py::handle src, bool convert) {
    if (!convert && !py::array_t<sketching::Float>::check_(src))
      return false;

    auto buf =
      py::array_t<sketching::Float, py::array::c_style | py::array::forcecast>::ensure(
        src);
    if (!buf || buf.ndim() != 1 || buf.strides()[0] != sizeof(sketching::Float))
      return false;

    value = sketching::span<sketching::Float>{
      (sketching::Float*)buf.data(),
      (sketching::span<sketching::Float>::size_type)buf.size()};

    return true;
  }

  // C++ -> Python
  static py::handle cast(const sketching::span<sketching::Float>& src,
                         py::return_value_policy /*policy*/, py::handle parent) {
    auto a = py::array_t<sketching::Float>({src.size()}, {sizeof(sketching::Float)},
                                           src.data(), parent);
    return a.release();
  }
};

template <>
struct type_caster<sketching::span<const sketching::Float>> {
public:
  PYBIND11_TYPE_CASTER(sketching::span<const sketching::Float>, _("ConstFloatSpan"));

  // Python -> C++
  bool load(py::handle src, bool convert) {
    if (!convert && !py::array_t<sketching::Float>::check_(src))
      return false;

    auto buf =
      py::array_t<sketching::Float, py::array::c_style | py::array::forcecast>::ensure(
        src);
    if (!buf || buf.ndim() != 1 || buf.strides()[0] != sizeof(sketching::Float))
      return false;

    value = sketching::span<const sketching::Float>{
      (const sketching::Float*)buf.data(),
      (sketching::span<const sketching::Float>::size_type)buf.size()};

    return true;
  }

  // C++ -> Python
  static py::handle cast(const sketching::span<const sketching::Float>& src,
                         py::return_value_policy /*policy*/, py::handle parent) {
    auto a = py::array_t<sketching::Float>({src.size()}, {sizeof(sketching::Float)},
                                           src.data(), parent);
    array_proxy(a.ptr())->flags &= ~detail::npy_api::NPY_ARRAY_WRITEABLE_;
    return a.release();
  }
};

} // namespace pybind11::detail
