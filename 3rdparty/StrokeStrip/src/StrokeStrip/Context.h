#pragma once

#include <gurobi_c++.h>

struct Context {
	Context(bool grb_env, bool grb_log = false);
	void optimize_model(GRBModel* model) const;

	std::unique_ptr<GRBEnv> grb_ptr;
	bool debug_viz = false;
	bool cut = false;
	bool taper_widths = false;
};
