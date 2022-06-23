#pragma once

#include <vector>

namespace sketching {

struct StrokeGraph;

void changed_faces(const StrokeGraph& graph1, const StrokeGraph& graph2,
                   std::vector<size_t>& out_changed1, std::vector<size_t>& out_changed2);

} // namespace sketching
