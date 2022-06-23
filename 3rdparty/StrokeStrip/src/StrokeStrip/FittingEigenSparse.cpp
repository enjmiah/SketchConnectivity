#define _USE_MATH_DEFINES
#include <cmath>

#include "FittingEigenSparse.h"
#include "Parameterization.h"
#include "Utils.h"
#include "SvgUtils.h"

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/OrderingMethods>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

FittingEigenSparse::FittingEigenSparse(const Context &context)
    : context(context) {}

void FittingEigenSparse::fit_svg(std::ostream& os, const Input& input, const std::map<int, std::vector<glm::dvec2>>& fits) {
	double padding = input.thickness;
	double w = input.width * input.thickness + 2 * padding;
	double h = input.height * input.thickness + 2 * padding;
	double x = -w / 2;
	double y = -h / 2;

	SVG::begin(os, x, y, w, h);
	for (auto& kv : fits) {
		std::vector<glm::dvec2> path = kv.second;
		for (auto& pt : path) {
			pt *= input.thickness;
		}
		SVG::polyline(os, path, input.thickness);
	}
	SVG::end(os);
}

std::map<int, std::unique_ptr<std::vector<glm::dvec2>[]>> FittingEigenSparse::fit(Input* input, bool fix_single_samples, double sample_rate,
	double k_weight, const std::vector<glm::dvec2>& endpoints) {
	return map_clusters<std::unique_ptr<std::vector<glm::dvec2>[]>>(*input, [&](Cluster& c) { 
		return fit_cluster(c, fix_single_samples, sample_rate, k_weight, endpoints); 
	});
}

std::unique_ptr<std::vector<glm::dvec2>[]> FittingEigenSparse::fit_cluster(Cluster cluster, bool fix_single_samples, double sample_rate, 
	double k_weight, const std::vector<glm::dvec2>& endpoints) {
	// 1. Get xsec samples more densely sampled near endpoints
	{
		Parameterization parameterization(context);
		double rate = cluster.periodic ? 0.025 : 0.1;
		if (sample_rate >= 0) {
			rate = sample_rate;
			if (cluster.periodic)
				rate /= 4;
		}
		cluster.xsecs = parameterization.xsecs_from_params(cluster, rate, !cluster.periodic);
	}

	auto mapping = std::make_unique<std::vector<glm::dvec2>[]>(cluster.strokes.size() + 1);

	// 2. Make one fitting sample per xsec
	std::vector<FittingEigenSparse::Sample> samples;
	for (size_t i = 0; i < cluster.xsecs.size(); ++i) {
		auto res = samples_from_xsec(cluster, i);
		const auto u = cluster.xsecs[i].u;
		for (const auto& p : cluster.xsecs[i].points) {
			mapping[p.stroke_idx].emplace_back(p.i, u);
		}
		samples.insert(samples.end(), res.begin(), res.end());
	}

	// 3. Solve for tangents
	auto tangents = fit_tangents(samples, cluster.periodic, k_weight);

	// 3.5 Weight position terms
	double endpoint_weight = std::sqrt(2.);
	const double sqrt_gamma = std::sqrt(POS_WEIGHT);
	for (size_t i = 0; i < samples.size(); ++i) {
		samples[i].pos_weight = sqrt_gamma;
		// Only fix the two endpoints
		if (!cluster.periodic && (i == 0 || i + 1 == samples.size())) {
			samples[i].pos_weight = (endpoints.empty()) ? endpoint_weight : END_POS_WEIGHT_SQRT;
		}
	}
	if (!cluster.periodic && fix_single_samples)
		fix_positions(samples);

	// 4. Solve for positions
	if (!samples.empty() && !tangents.empty())
		mapping[cluster.strokes.size()] = fit_positions(samples, tangents, cluster.periodic, endpoints);
	return mapping;
}

std::vector<double> FittingEigenSparse::fit_widths(const Cluster &cluster) {
	assert(context.grb_ptr);
	GRBModel model(*context.grb_ptr);

	bool periodic = cluster.periodic;
	std::vector<FittingEigenSparse::Sample> samples;
	for (size_t i = 0; i < cluster.xsecs.size(); ++i) {
		auto res = samples_from_xsec(cluster, i);
		const auto u = cluster.xsecs[i].u;
		samples.insert(samples.end(), res.begin(), res.end());
	}

	const unsigned int N = samples.size();
	//std::cout << "model finish 1" << std::endl;
	// Create width variables
	std::vector<GRBVar> vars;
	vars.reserve(N);
	for (size_t i = 0; i < N; ++i) {
		GRBVar w = model.addVar(-GRB_INFINITY, GRB_INFINITY, 1.0, GRB_CONTINUOUS);
		vars.push_back(w);
	}
	//std::cout << "model finish 2" << std::endl;
	std::vector<double> dists;
	for (size_t i = 1; i < N; ++i) {
		dists.push_back(glm::distance(samples[i].point, samples[i-1].point));
		dists.back() = std::max(1e-3, dists.back());
	}

	// Distances used in Laplacian matrix
	std::vector<double> laplacian_dists;
	{
		double total = 0.0;
		laplacian_dists.push_back(dists.front());
		total += dists.front();

		for (auto d : dists) {
			laplacian_dists.push_back(d);
			total += d;
		}

		laplacian_dists.push_back(dists.back());
		total += dists.back();

		for (auto& d : laplacian_dists) {
			d /= total;
		}
	}

	// Weight each distance matching term by the distance to the next point
	std::vector<double> dist_weights;
	{
		double total = 0.0;
		for (auto d : dists) {
			dist_weights.push_back(d);
			total += d;
		}

		dist_weights.push_back(dists.back());
		total += dists.back();

		for (auto& d : dist_weights) {
			d /= total;
		}
	}

	// Laplacian matrix
	std::vector<std::vector<double>> L;
	for (size_t i = 0; i < N; ++i) {
		std::vector<double> col(N, 0.0);
		L.push_back(col);
	}
	for (size_t i = 0; i < N; ++i) {
		for (size_t j = 0; j < N; ++j) {
			L[i][j] = 0.0;
		}
	}
	for (size_t i = 0; i < N; ++i) {
		if (context.taper_widths) {
			// Zero Dirichlet conditions
			L[i][i] = 1.0/laplacian_dists[i] + 1.0/laplacian_dists[i + 1];
			if (i > 0) {
				L[i][i-1] = -1.0/laplacian_dists[i];
			}
			if (i+1 < N) {
				L[i][i+1] = -1.0/laplacian_dists[i+1];
			}
		} else {
			// Zero Neumann conditions
			if (i > 0) {
				L[i][i-1] = -1.0/laplacian_dists[i-1];
			}
			if (i > 0 && i+1 < N) {
				// Middle
				L[i][i] = 1.0/laplacian_dists[i-1] + 1.0/laplacian_dists[i];
			} else if (i == 0) {
				// First
				L[i][i] = 1.0/laplacian_dists[i];
			} else {
				// Last
				L[i][i] = 1.0/laplacian_dists[i-1];
			}
			if (i+1 < N) {
				L[i][i+1] = -1.0/laplacian_dists[i];
			}
		}
	}

	// Laplacian objective: w^T * L * w;
	GRBQuadExpr wT_L_w;
	//// std::cout << "model finish 3" << std::endl;
	for (size_t i = 0; i < N; ++i) {
		GRBLinExpr L_w;
		for (size_t j = 0; j < N; ++j) {
			L_w += L[i][j] * vars[j];
		}
		wT_L_w += vars[i] * L_w;
	}
	// std::cout << "model finish 4" << std::endl;
	// Width matching objectives
	std::vector<GRBLinExpr> width_matches;
	width_matches.reserve(N);
	for (size_t i = 0; i < N; ++i) {
		double extra_w = samples[i].single_sample ? 1e2 : 1;
		width_matches.push_back(extra_w * dist_weights[i] * (vars[i] - samples[i].width));
	}
	// std::cout << "model finish 5" << std::endl;
	double matching_weight = 1e5;// Default: 200;
	model.setObjective(wT_L_w + matching_weight * l2_norm_sq(&model, width_matches));

	// Add constraints
	for (size_t i = 0; i < N; ++i) {
		if (samples[i].single_sample) {
			model.addConstr(vars[i] >= 0.9 * samples[i].width);
		} else {
			model.addConstr(vars[i] >= 0.45 * samples[i].width);
		}
	}
	for (size_t i : { size_t(0), size_t(N-1) }) {
		if (samples[i].single_sample)
			model.addConstr(vars[i] <= 1.1 * samples[i].width);
		else
			model.addConstr(vars[i] <= 1.5 * samples[i].width);
	}

	context.optimize_model(&model);

	std::vector<double> opt_widths;
	opt_widths.reserve(vars.size());
	for (auto& var : vars) {
		opt_widths.push_back(var.get(GRB_DoubleAttr_X));
	}

	if (periodic) {
		opt_widths.push_back(opt_widths.front());
	}

	return opt_widths;
}

double compute_region_gaussian(double value, double left, double right, double left_w,
	double right_w) {
	if (value <= left)
		return left_w;
	if (value >= right)
		return right_w;
	double sigma = (right - left) / 4;
	double gauss = std::exp(-0.5 * ((value - right) / sigma) * ((value - right) / sigma));
	gauss = (right_w - left_w) * gauss + left_w;

	return gauss;
}

void FittingEigenSparse::fix_positions(std::vector<Sample>& samples) {
	if (samples.size() <= 1) {
		if (!samples.empty())
			samples.front().pos_weight = END_POS_WEIGHT_SQRT;
		return;
	}

	const double sqrt_gamma = std::sqrt(POS_WEIGHT);
	// Position fixed range
	double fixed_range = 0.5;

	// Head
	std::pair<int, int> end_range(0, 1);
	int i;
	for (i = 1; i < samples.size() && samples[i].single_sample; ++i);
	end_range.second = (i == samples.size()) ? i : i + 1;
	std::pair<double, double> end_u_range(samples[end_range.first].u, samples[end_range.second - 1].u);
	for (i = end_range.first; i < end_range.second; ++i) {
		samples[i].pos_weight = (i == end_range.first) ? END_POS_WEIGHT_SQRT : 
			compute_region_gaussian(samples[i].u, fixed_range * (end_u_range.second - end_u_range.first) + end_u_range.first,
				end_u_range.second, END_POS_WEIGHT_SQRT, sqrt_gamma);
	}

	// Tail
	end_range = std::make_pair(samples.size() - 1, samples.size() - 2);
	for (i = samples.size() - 1; i >= 0 && samples[i].single_sample; --i);
	end_range.second = (i >= 0) ? i - 1 : i;
	end_u_range = std::make_pair(samples[end_range.first].u, samples[end_range.second + 1].u);
	for (i = end_range.first; i > end_range.second; --i) {
		samples[i].pos_weight = (i == end_range.first) ? END_POS_WEIGHT_SQRT :
			compute_region_gaussian(end_u_range.first - samples[i].u, fixed_range * (end_u_range.first - end_u_range.second),
				end_u_range.first - end_u_range.second, END_POS_WEIGHT_SQRT, sqrt_gamma);
	}
}

std::vector<glm::dvec2>
FittingEigenSparse::fit_positions(const std::vector<Sample>& samples, const std::vector<glm::dvec2>& tangents, bool periodic, 
		const std::vector<glm::dvec2>& endpoints) {
  size_t rows = (periodic) ? 2 * samples.size() : 2 * samples.size() - 1;
  Eigen::SparseMatrix<double> A(rows, samples.size());

  // A
  // Forward difference matrix D
  std::vector<double> W_diag;
  W_diag.reserve(samples.size());
  for (size_t j = 0; j < samples.size(); j++) {
    if (j + 1 < samples.size())
      W_diag.emplace_back(
          std::max(MIN_DIST, glm::dot(samples[j + 1].point - samples[j].point,
                                      tangents[j])));
    else if (periodic)
      W_diag.emplace_back(
          std::max(MIN_DIST,
                   glm::dot(samples[0].point - samples[j].point, tangents[j])));
  }

  // A
  // Forward difference matrix D
  for (size_t i = 0; i < tangents.size(); i++) {
    if (i + 1 != samples.size()) {
      A.coeffRef(i, i) = -1 / W_diag[i];
      A.coeffRef(i, i + 1) = 1 / W_diag[i];
    } else if (periodic) {
      A.coeffRef(i, i) = -1 / W_diag[i];
      A.coeffRef(i, 0) = 1 / W_diag[i];
    }
  }

  // Position term
  for (size_t i = 0; i < samples.size(); i++) {
    A.coeffRef(tangents.size() + i, i) = samples[i].pos_weight;
  }
  A.makeCompressed();

  std::vector<Eigen::VectorXd> b;
  b.resize(tangents.front().length(), Eigen::VectorXd(rows));

  // b
  // tangent
  for (size_t i = 0; i < tangents.size(); i++) {
    for (size_t j = 0; j < tangents.front().length(); j++) {
      b[j][i] = tangents[i][j];
    }
  }

  // Initial sample positions
  for (size_t i = 0; i < samples.size(); i++) {
    for (size_t j = 0; j < tangents.front().length(); j++) {
        b[j][tangents.size() + i] = samples[i].pos_weight * samples[i].point[j];
        if (!periodic && (i == 0 || i == samples.size() - 1)) {
          if (!endpoints.empty()) {
            double fixed_v = endpoints[(i + 1) == samples.size()][j];
            b[j][tangents.size() + i] = fixed_v;
          }
      }
    }
  }

  // Solve for the normal equation
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Lower,
                        Eigen::COLAMDOrdering<int>>
      solver;
  solver.compute(A.transpose() * A);
  if (solver.info() != Eigen::Success) {
    abort();
  }

  std::vector<Eigen::VectorXd> x;
  x.resize(samples.front().tangent.length());
  for (size_t j = 0; j < samples.front().tangent.length(); j++) {
    x[j] = solver.solve(A.transpose() * b[j]);
    // std::cout << "Diff: " << A.transpose() * A * x[j] - A.transpose() * b[j] << std::endl;
    if (solver.info() != Eigen::Success) {
      abort();
    }
  }

  std::vector<glm::dvec2> results;
  results.resize(samples.size());

  for (size_t i = 0; i < samples.size(); i++) {
    for (size_t j = 0; j < tangents.front().length(); j++) {
      results[i][j] = x[j][i];
    }
  }

  if (periodic) {
    results.push_back(results.front());
  }

  return results;
}

std::vector<glm::dvec2>
FittingEigenSparse::fit_tangents(const std::vector<Sample>& samples, bool periodic, double k_weight_in) {
  size_t rows = (periodic) ? 2 * samples.size() : 2 * samples.size() - 1;
  Eigen::SparseMatrix<double> A(rows, samples.size());

  // A
  // Forward difference matrix D
  double fit_k_weight = K_WEIGHT;
  if (k_weight_in >= 0)
      fit_k_weight = k_weight_in;
  double k_weight = std::sqrt(fit_k_weight);
  for (size_t i = 0; i < samples.size(); i++) {
    double scale = samples[i].no_k ? 0.1 : 1.;
    scale *= k_weight;
    if (i + 1 != samples.size()) {
      A.coeffRef(i, i) = -1 * scale;
      A.coeffRef(i, i + 1) = 1 * scale;
    } else if (periodic) {
      A.coeffRef(i, i) = -1 * scale;
      A.coeffRef(i, 0) = 1 * scale;
    }
  }
  // Positional term (matching the input tangents)
  double endpoint_weight = std::sqrt(2.);
  for (size_t i = 0; i < samples.size(); i++) {
    size_t j = (periodic) ? samples.size() + i : samples.size() - 1 + i;
    A.coeffRef(j, i) += (!periodic && (i == 0 || i == samples.size() - 1))
                            ? endpoint_weight
                            : 1;
  }
  A.makeCompressed();

  std::vector<Eigen::VectorXd> b;
  b.resize(samples.front().tangent.length(), Eigen::VectorXd(rows));

  // b
  // Curvature
  for (size_t i = 0; i < samples.size(); i++) {
    for (size_t k = 0; k < samples.front().tangent.length(); k++) {
      size_t pos_i = (periodic) ? samples.size() + i : samples.size() - 1 + i;
      b[k][pos_i] = samples[i].tangent[k];
    }

    if (periodic || i < samples.size() - 1) {
      double scale = samples[i].no_k ? 0.1 : 1.;
      scale *= k_weight;
      int j = (i + 1) % samples.size();
      glm::dvec2 avg_tangent =
          glm::normalize(samples[i].tangent + samples[j].tangent);
      glm::dvec2 ortho = normal(avg_tangent);

      for (size_t k = 0; k < samples.front().tangent.length(); k++) {
        b[k][i] = scale * samples[i].k * ortho[k];
      }
    }
  }

  // Solve for the normal equation
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Lower,
                        Eigen::COLAMDOrdering<int>>
      solver;
  auto AtA = A.transpose() * A;
  solver.compute(AtA);
  if (solver.info() != Eigen::Success) {
    abort();
  }

  std::vector<Eigen::VectorXd> x;
  x.resize(samples.front().tangent.length());
  for (size_t j = 0; j < samples.front().tangent.length(); j++) {
    x[j] = solver.solve(A.transpose() * b[j]);
    if (solver.info() != Eigen::Success) {
      abort();
    }
  }

  std::vector<glm::dvec2> tangents;
  tangents.resize(samples.size());

  for (size_t i = 0; i < samples.size(); i++) {
    for (size_t j = 0; j < samples.front().tangent.length(); j++) {
      tangents[i][j] = x[j][i];
    }
    tangents[i] = glm::normalize(tangents[i]);
  }

  if (!periodic)
    tangents.pop_back();

  return tangents;
}

std::vector<FittingEigenSparse::Sample>
FittingEigenSparse::samples_from_xsec(const Cluster &cluster, size_t xsec_idx) {
  std::vector<FittingEigenSparse::Sample> result;
  auto &xsec = cluster.xsecs[xsec_idx];

  if (xsec.connector &&
      glm::distance(xsec.points[0].point, xsec.points[1].point) > 1e-4) {
    glm::dvec2 pt1 = xsec.points[0].point;
    glm::dvec2 pt2 = xsec.points[1].point;
    glm::dvec2 tan1 = xsec.points[0].tangent;
    glm::dvec2 tan2 = xsec.points[1].tangent;

    glm::dvec2 tan = xsec.avg_tangent();
    glm::dvec2 ortho = normal(tan);

    // Find out which point leads into the next
    size_t stroke1 = xsec.points[0].stroke_idx;
    if (xsec_idx > 0) {
      auto &prev = cluster.xsecs[xsec_idx - 1];
      if (!std::any_of(prev.points.begin(), prev.points.end(),
                       [=](const Cluster::XSecPoint &p) {
                         return p.stroke_idx == stroke1;
                       })) {
        std::swap(pt1, pt2);
        std::swap(tan1, tan2);
      }
    } else {
      auto &next = cluster.xsecs[xsec_idx + 1];
      if (std::any_of(next.points.begin(), next.points.end(),
                      [=](const Cluster::XSecPoint &p) {
                        return p.stroke_idx == stroke1;
                      })) {
        std::swap(pt1, pt2);
        std::swap(tan1, tan2);
      }
    }

    double dist = glm::distance(pt1, pt2);
    size_t num_samples = std::max(1., std::ceil(dist / 0.05));
    double sign = glm::dot(tan2 - tan1, ortho) > 0 ? 1 : -1;
    double k = glm::length(tan2 - tan1) / double(num_samples) * sign;
    for (size_t i = 0; i < num_samples; ++i) {
      double t = double(i) / double(num_samples - 1);
      result.push_back({(1. - t) * pt1 + t * pt2, tan, k, false, true, 0.0, xsec.points.size() <= 1, xsec.u});
	  if (xsec.width >= 0)
		  result.back().width = xsec.width;
    }
  } else {
    result.push_back(
        {xsec.avg_point(), xsec.avg_tangent(), 0., false, false, 0.0, xsec.points.size() <= 1, xsec.u});
	if (xsec.width >= 0)
		result.back().width = xsec.width;
    auto &sample = result.back();

    double change_mags = 0.;
    int num_changes = 0;
    for (auto &p : xsec.points) {
      double di = 0.1;

      // Don't add tangent change at endpoints
      if (p.i < di ||
          p.i >= cluster.strokes[p.stroke_idx].points.size() - 1 - di)
        continue;

      glm::dvec2 ortho = normal(p.tangent);

      glm::dvec2 prev_pt(point(cluster.strokes[p.stroke_idx].points, p.i - di));
      glm::dvec2 pt(p.point);
      glm::dvec2 next_pt(point(cluster.strokes[p.stroke_idx].points, p.i + di));

      glm::dvec2 prev_tan(
          tangent(cluster.strokes[p.stroke_idx].points, p.i - di));
      glm::dvec2 tan(p.tangent);
      glm::dvec2 next_tan(
          tangent(cluster.strokes[p.stroke_idx].points, p.i + di));

      double k1 = glm::dot(ortho, tan - prev_tan) /
                  std::max(1e-4, glm::distance(pt, prev_pt));
      double k2 = glm::dot(ortho, next_tan - tan) /
                  std::max(1e-4, glm::distance(pt, next_pt));
      change_mags += (k1 + k2) / 2.;
      ++num_changes;
    }

    if (num_changes > 0) {
      change_mags /= double(num_changes);
    } else {
      sample.no_k = true;
    }
    double dist;
    if (xsec_idx < cluster.xsecs.size() - 1) {
      dist = glm::dot(normal(sample.tangent),
                      cluster.xsecs[xsec_idx + 1].avg_point() - sample.point);
    } else {
      dist = glm::dot(normal(sample.tangent),
                      sample.point - cluster.xsecs[xsec_idx - 1].avg_point());
    }
    sample.k = std::max(MIN_DIST, dist) * change_mags;

    if (xsec.width < 0 && xsec.points.size() > 0) {
      sample.width = std::abs(
          glm::dot(normal(sample.tangent),
                   xsec.points.front().point - xsec.points.back().point));
    }
  }

  return result;
}

