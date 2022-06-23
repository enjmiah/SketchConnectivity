#pragma once

#include "../types.h"

namespace sketching {

struct StrokeGraph;

void load_json(const std::string& path, StrokeGraph& out_graph);

void save_json(const StrokeGraph&, const std::string& path);

} // namespace sketching
