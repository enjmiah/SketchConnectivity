#pragma once

#include <map>

#include "Cluster.h"
#include "Context.h"

struct FittingEigenSparse {
  FittingEigenSparse(const Context &context);
  std::map<int, std::unique_ptr<std::vector<glm::dvec2>[]>> fit(Input* input, bool fix_single_samples, double sample_rate = -1, double k_weight = -1,
		const std::vector<glm::dvec2>& endpoints = std::vector<glm::dvec2>());
  std::unique_ptr<std::vector<glm::dvec2>[]> fit_cluster(Cluster cluster, bool fix_single_samples, double sample_rate = -1, double k_weight = -1, 
		const std::vector<glm::dvec2>& endpoints = std::vector<glm::dvec2>());

  std::vector<double> fit_widths(const Cluster &cluster);
  void fit_svg(std::ostream& os, const Input& input, const std::map<int, std::vector<glm::dvec2>>& fits);

private:
  const Context &context;

  double K_WEIGHT = 1e3;
  const double POS_WEIGHT = 5e-5;
  const double MIN_DIST = 1e-4;
  const double END_POS_WEIGHT_SQRT = 500;

  struct Sample {
    glm::dvec2 point;
    glm::dvec2 tangent;
    double k;
    bool no_k;
    bool gap;
    double width;
	bool single_sample;
	double u;

	double pos_weight;
  };
  std::vector<Sample> samples_from_xsec(const Cluster &cluster,
                                        size_t xsec_idx);

  std::vector<glm::dvec2> fit_tangents(const std::vector<Sample>& samples, bool periodic, double k_weight = -1);

  void fix_positions(std::vector<Sample>& samples);
  std::vector<glm::dvec2> fit_positions(const std::vector<Sample>& samples, const std::vector<glm::dvec2>& tangents, bool periodic, 
		const std::vector<glm::dvec2>& endpoints);
};

