import _sketching as _s


##############################
# Classifier hyperparameters #
##############################

N_ESTIMATORS = 100
MAX_DEPTH_E = 10
MAX_DEPTH_T = 12


#######################
# Classifier features #
#######################

BUSYNESS_FALLOFF = 1.0

def make_features_e():
    return [
        _s.features.EndEndJunctionType(),
        _s.features.EnvelopeDistance(_s.features.Normalization.PEN_WIDTH_PAIRWISE_MEAN),
        _s.features.EnvelopeDistance(_s.features.Normalization.STROKE_LENGTH_PAIRWISE_MIN),
        _s.features.EnvelopeDistance(_s.features.Normalization.STROKE_LENGTH_PAIRWISE_MAX),
        _s.features.StepawayTangentAngleMin(),
        _s.features.StepawayTangentAngleMax(),
        _s.features.ClosestAnyOtherOverConnectionMin(),
        _s.features.ClosestAnyOtherOverConnectionMax(),
        _s.features.ProjectionToEndpointRatioMin(),
        _s.features.ProjectionToEndpointRatioMax(),
        _s.features.StepawayOverConnectionMin(),
        _s.features.StepawayOverConnectionMax(),
        _s.features.ProjectionOverConnectionMin(),
        _s.features.ProjectionOverConnectionMax(),
    ]

def make_features_t():
    return [
        _s.features.EnvelopeDistance(_s.features.Normalization.PEN_WIDTH_PAIRWISE_MEAN),
        _s.features.EnvelopeDistance(_s.features.Normalization.STROKE_LENGTH1),
        _s.features.EnvelopeDistance(_s.features.Normalization.STROKE_LENGTH2),
        _s.features.StepawayTangentAngle1(),
        _s.features.Busyness1(BUSYNESS_FALLOFF),
        _s.features.ClosestAnyOtherOverConnection1(limit_to_visible=True),
        _s.features.ConnectedDistanceToEndpoint(),
        _s.features.StepawayOverConnection1(),
    ]
