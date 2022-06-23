#include "cast.h"

#include <sketching/features/junction_features_impl.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;
using namespace ::sketching;
using namespace ::sketching::features;

void init_junction_features(py::module& m) {
  const auto fm = m.def_submodule("features", "Features for classifying junctions.");

  py::class_<JunctionFeature> jf(m, "JunctionFeature");
  jf.def("description", &JunctionFeature::description);
  jf.def("human_readable", &JunctionFeature::human_readable);
  jf.def("human_readable", [](JunctionFeature& fea, py::EigenDRef<Eigen::VectorXd> v) {
    const auto n = v.rows();
    for (Index i = 0; i < n; ++i) {
      fea.human_readable({&v[i], 1});
    }
  });
  jf.def("init", [](JunctionFeature& f, const PolylineBVH& bvh) { f.init(bvh); });
  jf.def("__call__", [](const JunctionFeature& f, //
                        const Stroke& s1, Float arclen1, size_t idx1, //
                        const Stroke& s2, Float arclen2, size_t idx2) { //
    return f(s1, arclen1, idx1, s2, arclen2, idx2);
  });

  py::enum_<Normalization>(fm, "Normalization")
    .value("BOUNDING_BOX", Normalization::BoundingBox)
    .value("STROKE_LENGTH1", Normalization::StrokeLength1)
    .value("STROKE_LENGTH2", Normalization::StrokeLength2)
    .value("STROKE_LENGTH_PAIRWISE_MIN", Normalization::StrokeLengthPairwiseMin)
    .value("STROKE_LENGTH_PAIRWISE_MAX", Normalization::StrokeLengthPairwiseMax)
    .value("STROKE_LENGTH_PAIRWISE_MEAN", Normalization::StrokeLengthPairwiseMean)
    .value("PEN_WIDTH1", Normalization::PenWidth1)
    .value("PEN_WIDTH2", Normalization::PenWidth2)
    .value("PEN_WIDTH_PAIRWISE_MAX", Normalization::PenWidthPairwiseMax)
    .value("PEN_WIDTH_PAIRWISE_MEAN", Normalization::PenWidthPairwiseMean);

  py::class_<Busyness1, JunctionFeature>(fm, "Busyness1")
    .def(py::init<Float>(), "busyness_falloff"_a);

  py::class_<Busyness2, JunctionFeature>(fm, "Busyness2")
    .def(py::init<Float>(), "busyness_falloff"_a);

  py::class_<BusynessMin, JunctionFeature>(fm, "BusynessMin")
    .def(py::init<Float>(), "busyness_falloff"_a);

  py::class_<BusynessMax, JunctionFeature>(fm, "BusynessMax")
    .def(py::init<Float>(), "busyness_falloff"_a);

  py::class_<EnvelopeDistance, JunctionFeature>(fm, "EnvelopeDistance") //
    .def(py::init<Normalization>(), "scheme"_a);

#define BIND_FEATURE(Feature)                                                            \
  py::class_<Feature, JunctionFeature>(fm, #Feature).def(py::init<>())

#define BIND_FEATURE_4X(FeaturePrefix)                                                   \
  BIND_FEATURE(FeaturePrefix##1);                                                        \
  BIND_FEATURE(FeaturePrefix##2);                                                        \
  BIND_FEATURE(FeaturePrefix##Min);                                                      \
  BIND_FEATURE(FeaturePrefix##Max)

  BIND_FEATURE(EndEndJunctionType);
  BIND_FEATURE(EndStrokeJunctionType);

  BIND_FEATURE_4X(ProjectionOverConnection);

  BIND_FEATURE_4X(ClosestDistanceOnExtension);

  BIND_FEATURE_4X(ClosestAnyOverConnection);

  BIND_FEATURE(ClosestEndpointOverConnection1)
    .def("closest", &ClosestEndpointOverConnection1::closest);

  BIND_FEATURE(ClosestAnyOtherOverConnection1)
    .def(py::init<Float>(), "limit_to_visible"_a)
    .def("closest", &ClosestAnyOtherOverConnection1::closest);
  BIND_FEATURE(ClosestAnyOtherOverConnection2)
    .def(py::init<Float>(), "limit_to_visible"_a);
  BIND_FEATURE(ClosestAnyOtherOverConnectionMin)
    .def(py::init<Float>(), "limit_to_visible"_a);
  BIND_FEATURE(ClosestAnyOtherOverConnectionMax)
    .def(py::init<Float>(), "limit_to_visible"_a);

  BIND_FEATURE(OtherEndpointClosestAnyEnvOverEnvConnection1);
  BIND_FEATURE(OtherEndpointClosestAnyEnvOverEnvConnection2);

  BIND_FEATURE(StepawayOverConnection1).def(py::init<Float>(), "factor"_a);
  BIND_FEATURE(StepawayOverConnection2).def(py::init<Float>(), "factor"_a);
  BIND_FEATURE(StepawayOverConnectionMin).def(py::init<Float>(), "factor"_a);
  BIND_FEATURE(StepawayOverConnectionMax).def(py::init<Float>(), "factor"_a);

  BIND_FEATURE_4X(NearestEndpointOverStepaway);

  BIND_FEATURE_4X(StepawayOverProjection);

  BIND_FEATURE(TangentAngle1);
  BIND_FEATURE(TangentAngle2);

  BIND_FEATURE_4X(StepawayTangentAngle);

  BIND_FEATURE(InteriorTangentAngle);

  BIND_FEATURE(PenWidth1);
  BIND_FEATURE(PenWidth2);
  BIND_FEATURE(PenWidthMax);
  BIND_FEATURE(PenWidthMean);

  BIND_FEATURE(ConnectedDistanceToEndpoint);

  BIND_FEATURE_4X(ProjectionToEndpointRatio);

  BIND_FEATURE(ConnectedLocationOverConnection);
  BIND_FEATURE_4X(ProjectionToEndpointOverConnection);

  BIND_FEATURE(DrawingOrder);
  BIND_FEATURE(DrawingOrderUnsignedDiff);
  BIND_FEATURE(DrawingOrderBucketedUnsignedDiff);

  //

  BIND_FEATURE(AbsEnvelopeDistance);
  BIND_FEATURE(AbsCenterlineDistance);
  BIND_FEATURE(AbsStrokeLength1);
  BIND_FEATURE(AbsStrokeLength2);
  BIND_FEATURE(AbsProjectionDist1);
  BIND_FEATURE(AbsProjectionDist2);
  BIND_FEATURE(AbsProjectionToClosestEndp1);
  BIND_FEATURE(AbsProjectionToClosestEndp2);
  BIND_FEATURE(AbsPenWidth1);
  BIND_FEATURE(AbsPenWidth2);
  BIND_FEATURE(AbsWidth1);
  BIND_FEATURE(AbsWidth2);
  BIND_FEATURE(AbsStepawayDist1);
  BIND_FEATURE(AbsStepawayDist2);
}
