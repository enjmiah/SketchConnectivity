#pragma once

#include <sketching/sketching.h>

#include <cstdint>
#include <vector>

namespace sketching {

struct Drawing {
  Drawing() = default;
  explicit Drawing(const span<const Stroke> ss)
    : view_(ss) {}

  const Stroke& at(std::int64_t i) const {
    if (i < 0) {
      i += size();
    }
    if (i >= 0 && i < (std::int64_t)size()) {
      return strokes()[i];
    }
    throw std::out_of_range("");
  }

  void add(const Stroke& s) { mut_strokes().push_back(s.clone()); }

  inline static const auto avg_sample_dist_doc =
    "Compute the mean distance between adjacent vertices in the stroke.";
  Float avg_sample_dist() const;

  size_t size() const {
    if (view_.data()) {
      return view_.size();
    }
    return strokes_.size();
  }

  span<const Stroke> strokes() const {
    if (view_.data()) {
      return view_;
    }
    return strokes_;
  }

  operator span<const Stroke>() const { return strokes(); }

  std::vector<Stroke>& mut_strokes() {
    if (view_.data()) {
      throw std::domain_error("cannot modify a read-only view");
    }
    return strokes_;
  }

  Drawing(Drawing&& mE) noexcept = default;
  Drawing& operator=(Drawing&& mE) noexcept = default;

  Drawing(Drawing& mE) noexcept = delete;
  Drawing& operator=(Drawing& mE) noexcept = delete;

  std::string path_;
  std::vector<Stroke> strokes_;
  span<const Stroke> view_ = {nullptr, 0};
};

} // namespace sketching
