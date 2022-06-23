#include "drawing.h"

namespace sketching {

Float Drawing::avg_sample_dist() const {
  auto acc = Float(0);
  auto denom = Index(0);
  for (const auto& s : strokes()) {
    acc += s.length();
    denom += s.size() - 1; // number of segments
  }
  assert(denom != 0);
  return acc / Float(denom);
}

} // namespace sketching
