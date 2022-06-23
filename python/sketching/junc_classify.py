from __future__ import annotations
import json
import logging
import os
from dataclasses import dataclass
from abc import abstractmethod
from typing import Callable, List, Optional
from numbers import Number

import json5
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

import _sketching as _s
from sketching.util import fair_savgol
from sketching.junction import from_json


NEGATIVE = 0
POSITIVE = 1
UNKNOWN = 2
INCONSISTENT = 3 # Internal use only.

n_closest_e = 8
n_closest_t = 8


BUSYNESS_FALLOFF = 1.0

def make_features_e_all():
    return [
        _s.features.EndEndJunctionType(),
        _s.features.EnvelopeDistance(_s.features.Normalization.PEN_WIDTH_PAIRWISE_MEAN),
        _s.features.EnvelopeDistance(_s.features.Normalization.STROKE_LENGTH_PAIRWISE_MAX),
        _s.features.EnvelopeDistance(_s.features.Normalization.STROKE_LENGTH_PAIRWISE_MIN),
        _s.features.StepawayTangentAngleMin(),
        _s.features.StepawayTangentAngleMax(),
        _s.features.Busyness1(BUSYNESS_FALLOFF),
        _s.features.Busyness2(BUSYNESS_FALLOFF),
        _s.features.BusynessMin(BUSYNESS_FALLOFF),
        _s.features.BusynessMax(BUSYNESS_FALLOFF),
        _s.features.ClosestDistanceOnExtensionMin(),
        _s.features.ClosestDistanceOnExtensionMax(),
        _s.features.ClosestAnyOverConnectionMin(),
        _s.features.ClosestAnyOverConnectionMax(),
        _s.features.ClosestAnyOtherOverConnectionMin(),
        _s.features.ClosestAnyOtherOverConnectionMax(),
        _s.features.ProjectionToEndpointRatioMin(),
        _s.features.ProjectionToEndpointRatioMax(),
        _s.features.ProjectionToEndpointOverConnectionMin(),
        _s.features.ProjectionToEndpointOverConnectionMax(),
        _s.features.StepawayOverConnectionMin(),
        _s.features.StepawayOverConnectionMax(),
        _s.features.ProjectionOverConnectionMin(),
        _s.features.ProjectionOverConnectionMax(),
        _s.features.StepawayOverProjectionMin(),
        _s.features.StepawayOverProjectionMax(),

        _s.features.AbsCenterlineDistance(),
        _s.features.AbsEnvelopeDistance(),
        _s.features.AbsStrokeLength1(),
        _s.features.AbsStrokeLength2(),
        _s.features.AbsWidth1(),
        _s.features.AbsWidth2(),
        _s.features.AbsPenWidth1(),
        _s.features.AbsPenWidth2(),
        _s.features.AbsProjectionDist1(),
        _s.features.AbsProjectionDist2(),
        _s.features.AbsProjectionToClosestEndp1(),
        _s.features.AbsProjectionToClosestEndp2(),
        _s.features.AbsStepawayDist1(),
        _s.features.AbsStepawayDist2(),
    ]

def make_features_t_all():
    return [
        _s.features.EnvelopeDistance(_s.features.Normalization.PEN_WIDTH_PAIRWISE_MEAN),
        _s.features.EnvelopeDistance(_s.features.Normalization.STROKE_LENGTH1),
        _s.features.EnvelopeDistance(_s.features.Normalization.STROKE_LENGTH2),
        _s.features.EnvelopeDistance(_s.features.Normalization.STROKE_LENGTH_PAIRWISE_MAX),
        _s.features.EnvelopeDistance(_s.features.Normalization.STROKE_LENGTH_PAIRWISE_MIN),
        _s.features.StepawayTangentAngle1(),
        _s.features.ClosestAnyOverConnection1(),
        _s.features.ClosestAnyOtherOverConnection1(limit_to_visible=True),
        _s.features.OtherEndpointClosestAnyEnvOverEnvConnection1(),
        _s.features.ConnectedDistanceToEndpoint(),
        _s.features.ConnectedLocationOverConnection(),
        _s.features.StepawayOverConnection1(),
        _s.features.Busyness1(BUSYNESS_FALLOFF),
        _s.features.Busyness2(BUSYNESS_FALLOFF),
        _s.features.BusynessMin(BUSYNESS_FALLOFF),
        _s.features.BusynessMax(BUSYNESS_FALLOFF),
        _s.features.ClosestDistanceOnExtension1(),
        _s.features.EndStrokeJunctionType(),

        _s.features.AbsCenterlineDistance(),
        _s.features.AbsEnvelopeDistance(),
        _s.features.AbsStrokeLength1(),
        _s.features.AbsStrokeLength2(),
        _s.features.AbsWidth1(),
        _s.features.AbsWidth2(),
        _s.features.AbsPenWidth1(),
        _s.features.AbsPenWidth2(),
        _s.features.AbsProjectionDist1(),
        _s.features.AbsProjectionToClosestEndp1(),
        _s.features.AbsStepawayDist1(),
    ]


def fair_drawing(original: _s.Drawing) -> None:
    for stroke in original:
        fair_savgol(stroke)
    original.remove_duplicate_vertices()


def matchups_full(nstrokes: int, gt_junctions: List[_s.Junction]) -> tuple:
    """
    Returns (e_matchups_head, e_matchups_tail, t_matchups_head, t_matchups_tail)
    for a complete set of junctions.

    e_matchups_head[i, j] = 1
        <=> head of stroke i is connected to stroke j // 2, head if j % 2 == 0
    e_matchups_tail[i, j] = 1
        <=> tail of stroke i is connected to stroke j // 2, head if j % 2 == 0
    t_matchups_head[i, j] = 1 <=> head of stroke i is connected to stroke j
    t_matchups_tail[i, j] = 1 <=> tail of stroke i is connected to stroke j

    Values are NEGATIVE, POSITIVE, and UNKNOWN (for weak connections).
    """
    e_matchups_head = np.full((nstrokes, 2 * nstrokes), NEGATIVE, dtype=np.uint8)
    e_matchups_tail = np.full((nstrokes, 2 * nstrokes), NEGATIVE, dtype=np.uint8)
    t_matchups_head = np.full((nstrokes, nstrokes), NEGATIVE, dtype=np.uint8)
    t_matchups_tail = np.full((nstrokes, nstrokes), NEGATIVE, dtype=np.uint8)
    for junction in gt_junctions:
        for (i, (si, arc)) in enumerate(junction):
            if arc <= 0.0 or arc >= 1.0:
                e_m = e_matchups_head if arc <= 0.0 else e_matchups_tail
                t_m = t_matchups_head if arc <= 0.0 else t_matchups_tail
                for (j, (sj, other_arc)) in enumerate(junction):
                    if i != j:
                        if other_arc < 0.01 and e_m[si, 2 * sj] != POSITIVE:
                            e_m[si, 2 * sj] = (UNKNOWN if junction.is_weak else POSITIVE)
                        elif other_arc > 0.99 and e_m[si, 2 * sj + 1] != POSITIVE:
                            e_m[si, 2 * sj + 1] = (UNKNOWN if junction.is_weak else POSITIVE)
                        if t_m[si, sj] != POSITIVE:
                            if junction.type == _s.Junction.T:
                                t_m[si, sj] = (UNKNOWN if junction.is_weak else POSITIVE)
                            elif junction.type == _s.Junction.R:
                                t_m[si, sj] = UNKNOWN
    return (e_matchups_head, e_matchups_tail, t_matchups_head, t_matchups_tail)


def matchups_partial(nstrokes: int, junctions_pos: List[_s.Junction],
                     junctions_neg: List[_s.Junction]) -> tuple:
    """
    Returns (e_matchups_head, e_matchups_tail, t_matchups_head, t_matchups_tail)
    for a partial set of junctions.  Each junction may only have 2 points.
    """
    e_matchups_head = np.full((nstrokes, 2 * nstrokes), UNKNOWN, dtype=np.uint8)
    e_matchups_tail = np.full((nstrokes, 2 * nstrokes), UNKNOWN, dtype=np.uint8)
    t_matchups_head = np.full((nstrokes, nstrokes), UNKNOWN, dtype=np.uint8)
    t_matchups_tail = np.full((nstrokes, nstrokes), UNKNOWN, dtype=np.uint8)
    for junction in junctions_pos:
        if len(junction) != 2:
            logging.warning('skipping a high valence partial annotation')
            continue
        assert not junction.is_weak, 'weak partial annotations not supported'
        if junction.type == _s.Junction.R:
            for i in range(2): # Symmetric.
                (si, arc1) = junction[i]
                (sj, arc2) = junction[1 - i]
                e_m = e_matchups_head if arc1 <= 0.0 else e_matchups_tail
                if arc2 <= 0.0:
                    e_m[si, 2 * sj] = POSITIVE
                elif arc2 >= 1.0:
                    e_m[si, 2 * sj + 1] = POSITIVE
        elif junction.type == _s.Junction.T:
            (si, arc) = junction[1] # Endpoint.
            assert arc in (0.0, 1.0)
            (sj, _) = junction[0] # Interior (possibly).
            t_m = t_matchups_head if arc <= 0.0 else t_matchups_tail
            t_m[si, sj] = POSITIVE
        else:
            raise NotImplementedError(
                ("Partial annotations from the annotator tool shouldn't have " +
                 f"junctions of type {junction.type}"))
    for junction in junctions_neg:
        if len(junction) != 2:
            logging.warning('skipping a high valence partial annotation')
            continue
        assert not junction.is_weak, 'weak partial junctions not supported'
        if junction.type == _s.Junction.R:
            for i in range(2): # Symmetric.
                (si, arc1) = junction[i]
                (sj, arc2) = junction[1 - i]
                e_m = e_matchups_head if arc1 <= 0.0 else e_matchups_tail
                if arc2 <= 0.0:
                    e_m[si, 2 * sj] = \
                        (INCONSISTENT if e_m[si, 2 * sj] == POSITIVE else NEGATIVE)
                elif arc2 >= 1.0:
                    e_m[si, 2 * sj + 1] = \
                        (INCONSISTENT if e_m[si, 2 * sj + 1] == POSITIVE else NEGATIVE)
        elif junction.type == _s.Junction.T:
            (si, arc) = junction[1] # Endpoint.
            assert arc in (0.0, 1.0)
            (sj, _) = junction[0] # Interior (possibly).
            t_m = t_matchups_head if arc <= 0.0 else t_matchups_tail
            t_m[si, sj] = (INCONSISTENT if t_m[si, sj] == POSITIVE else NEGATIVE)
            if t_m[si, sj] == INCONSISTENT:
                logging.warning(f'{si}, {sj} inconsistent')
        else:
            raise NotImplementedError(
                ("Partial annotations from the annotator tool shouldn't have " +
                 f"junctions of type {junction.type}"))
    # Let's replace inconsistent labels with UNKNOWN.
    e_matchups_head[e_matchups_head == INCONSISTENT] = UNKNOWN
    e_matchups_tail[e_matchups_tail == INCONSISTENT] = UNKNOWN
    t_matchups_head[t_matchups_head == INCONSISTENT] = UNKNOWN
    t_matchups_tail[t_matchups_tail == INCONSISTENT] = UNKNOWN
    return (e_matchups_head, e_matchups_tail, t_matchups_head, t_matchups_tail)


class Fraction:
    """
    Replacement for fractions.Fraction that doesn't automatically reduce.
    """
    def __init__(self, numerator: int, denominator: int):
        self.numerator = numerator
        self.denominator = denominator

    def __str__(self):
        return f'{self.numerator}/{self.denominator}'

    def __float__(self):
        if self.denominator == 0:
            return 1.0
        return float(self.numerator) / float(self.denominator)


@dataclass
class ClassifierMetricsDiff:
    dtp: int
    dtn: int
    dfp: int
    dfn: int


LABELS = np.array([False, True], dtype=bool)
class ClassifierMetrics:
    def __init__(self, tp: int = 0, tn: int = 0, fp: int = 0, fn: int = 0):
        self.tp = int(tp)
        self.tn = int(tn)
        self.fp = int(fp)
        self.fn = int(fn)

    @classmethod
    def from_pred(cls, y_true: np.ndarray, y_pred: np.ndarray) -> ClassifierMetrics:
        self = cls.__new__(cls)
        (self.tn, self.fp, self.fn, self.tp) = \
            confusion_matrix(y_true, y_pred, labels=LABELS).ravel()
        self.tp = int(self.tp)
        self.tn = int(self.tn)
        self.fp = int(self.fp)
        self.fn = int(self.fn)
        return self

    @classmethod
    def from_json(cls, jso) -> ClassifierMetrics:
        self = cls.__new__(cls)
        self.tp = jso['tp']
        self.tn = jso['tn']
        self.fp = jso['fp']
        self.fn = jso['fn']
        return self

    @property
    def balanced_accuracy(self) -> float:
        return 0.5 * (float(self.recall) + float(self.true_negative_rate))

    @property
    def true_negative_rate(self) -> Fraction:
        denom = self.tn + self.fp
        return Fraction(self.tn, denom)

    @property
    def precision(self) -> Fraction:
        denom = self.tp + self.fp
        return Fraction(self.tp, denom)

    @property
    def recall(self) -> Fraction:
        denom = self.tp + self.fn
        return Fraction(self.tp, denom)

    def __iadd__(self, other: ClassifierMetrics):
        self.tp += other.tp
        self.tn += other.tn
        self.fp += other.fp
        self.fn += other.fn
        return self

    def __sub__(self, other: ClassifierMetrics) -> ClassifierMetricsDiff:
        return ClassifierMetricsDiff(dtp=self.tp - other.tp,
                                     dtn=self.tn - other.tn,
                                     dfp=self.fp - other.fp,
                                     dfn=self.fn - other.fn)

    def __bool__(self):
        return bool(self.tp or self.tn or self.fp or self.fn)

    def as_json(self) -> dict:
        return {
            'tp': int(self.tp),
            'tn': int(self.tn),
            'fp': int(self.fp),
            'fn': int(self.fn),
        }


class OverallMetricsDiff:
    def __init__(self, e: ClassifierMetricsDiff, t: ClassifierMetricsDiff):
        self.e = e
        self.t = t

    @property
    def dtp(self) -> int:
        return self.e.dtp + self.t.dtp

    @property
    def dtn(self) -> int:
        return self.e.dtn + self.t.dtn

    @property
    def dfp(self) -> int:
        return self.e.dfp + self.t.dfp

    @property
    def dfn(self) -> int:
        return self.e.dfn + self.t.dfn


class OverallMetrics:
    def __init__(self):
        self.e = ClassifierMetrics()
        self.t = ClassifierMetrics()

    @classmethod
    def from_pred(cls, e_true: np.ndarray, e_pred: np.ndarray,
                 t_true: np.ndarray, t_pred: np.ndarray) -> OverallMetrics:
        self = cls.__new__(cls)
        self.e = ClassifierMetrics.from_pred(e_true, e_pred)
        self.t = ClassifierMetrics.from_pred(t_true, t_pred)
        return self

    @classmethod
    def from_file(cls, fp) -> OverallMetrics:
        return OverallMetrics.from_json(json.load(fp))

    @classmethod
    def from_json(cls, obj: dict) -> OverallMetrics:
        self = cls.__new__(cls)
        self.e = ClassifierMetrics.from_json(obj['e'])
        self.t = ClassifierMetrics.from_json(obj['t'])
        return self

    @property
    def tp(self) -> int:
        return self.e.tp + self.t.tp

    @property
    def tn(self) -> int:
        return self.e.tn + self.t.tn

    @property
    def fp(self) -> int:
        return self.e.fp + self.t.fp

    @property
    def fn(self) -> int:
        return self.e.fn + self.t.fn

    @property
    def overall_balanced_accuracy(self) -> float:
        return 0.5 * (float(self.overall_recall) + float(self.overall_true_negative_rate))

    @property
    def overall_true_negative_rate(self) -> Fraction:
        denom = self.tn + self.fp
        return Fraction(self.tn, denom)

    @property
    def overall_precision(self) -> Fraction:
        denom = self.tp + self.fp
        return Fraction(self.tp, denom)

    @property
    def overall_recall(self) -> Fraction:
        denom = self.tp + self.fn
        return Fraction(self.tp, denom)

    def __iadd__(self, other: OverallMetrics):
        self.e += other.e
        self.t += other.t
        return self

    def __sub__(self, other: OverallMetrics) -> OverallMetricsDiff:
        return OverallMetricsDiff(self.e - other.e, self.t - other.t)

    def __bool__(self):
        return bool(self.e or self.t)

    def as_json(self) -> dict:
        return {
            'e': self.e.as_json(),
            't': self.t.as_json(),
        }


class Dataset:
    class Base:
        def __init__(self):
            self.fea_arrs = []
            self.all_fea_arrs = []
            self.lab_arrs = []
            self.naive_lab_arrs = []
            self.candidates1 = []
            self.candidates2 = []

        def features(self) -> np.ndarray:
            return np.vstack(self.fea_arrs)

        def all_features(self) -> np.ndarray:
            return np.vstack(self.all_fea_arrs)

        def cand1(self) -> np.ndarray:
            return np.concatenate(self.candidates1)

        def cand2(self) -> np.ndarray:
            if len(self.candidates2[0].shape) > 1:
                return np.vstack(self.candidates2)
            else:
                return np.concatenate(self.candidates2)

        def labels(self) -> np.ndarray:
            return np.concatenate(self.lab_arrs)

        def groups(self) -> np.ndarray:
            known_lab_arrs = [arr[arr != -1] for arr in self.lab_arrs]
            groups = np.ones(sum(len(a) for a in known_lab_arrs), dtype=np.int32)
            start = 0
            for (group_index, a) in enumerate(known_lab_arrs):
                groups[start:start+len(a)] = group_index
                start += len(a)
            assert np.all(groups[:-1] <= groups[1:]) # Sorted.
            return groups

        def naive_labels(self) -> np.ndarray:
            if self.naive_lab_arrs:
                return np.concatenate(self.naive_lab_arrs)
            return np.array([], dtype=bool)

        def add_drawing(self, graph, matchups_head, matchups_tail, ignored_strokes):
            """
            Note: for efficiency reasons, the caller must update
            `naive_lab_arrs` themselves.
            """
            (cand1, cand2) = self.gen_candidates(graph, ignored_strokes)
            labels = \
                self.gen_labels(cand1, cand2, matchups_head, matchups_tail)

            self.lab_arrs.append(labels)
            self.candidates1.append(cand1)
            self.candidates2.append(cand2)
            self.fea_arrs.append(self.gen_features(graph, cand1, cand2))
            self.all_fea_arrs.append(self.gen_all_features(graph, cand1, cand2))

        @abstractmethod
        def gen_candidates(self, graph, ignored_strokes):
            raise NotImplementedError

        @abstractmethod
        def gen_labels(self, cand1, cand2, matchups_head, matchups_tail):
            raise NotImplementedError

        @abstractmethod
        def gen_features(self, graph, cand1, cand2):
            raise NotImplementedError

        @abstractmethod
        def gen_all_features(self, graph, cand1, cand2):
            raise NotImplementedError

        def __bool__(self):
            return bool(self.candidates1)

    class Endp(Base):
        def __init__(self, make_features):
            self.make_features = make_features
            super().__init__()

        def gen_candidates(self, graph, ignored_strokes):
            vanilla_candidates = _s.vanilla_candidates(graph, True, n_closest_e)
            junctions = [junc for junc in vanilla_candidates
                         if junc.type == _s.Junction.R]
            # print(f'vanilla_candidates: {len(vanilla_candidates)}; junctions: {len(junctions)}')
            cand1 = np.zeros(len(junctions), dtype=np.int32)
            cand2 = np.zeros(len(junctions), dtype=np.int32)
            for (i, junction) in enumerate(junctions):
                assert len(junction) == 2, 'high-valence not supported!'
                (s1, arclen1) = junction.points[0]
                cand1[i] = (s1 if arclen1 == 0 else ~s1)
                (s2, arclen2) = junction.points[1]
                cand2[i] = (s2 if arclen2 == 0 else ~s2)
            return (cand1, cand2)
            # return _s.endpoint_classifier_candidates(
            #     graph, ignored_strokes, n_closest=n_closest_e)

        def gen_labels(self, cand1, cand2, matchups_head, matchups_tail):
            return _s.endpoint_classifier_labels(cand1, cand2, matchups_head,
                                                 matchups_tail)

        def gen_features(self, graph, cand1, cand2):
            # Note: One must create the features anew every time, since a
            # feature can cache drawing-specific information for efficiency.
            features = self.make_features()
            return _s.endpoint_classifier_features(graph, cand1, cand2, features)

        def gen_all_features(self, graph, cand1, cand2):
            # Note: One must create the features anew every time, since a
            # feature can cache drawing-specific information for efficiency.
            features = make_features_e_all()
            return _s.endpoint_classifier_features(graph, cand1, cand2, features)

    class TJunc(Base):
        def __init__(self, make_features):
            self.make_features = make_features
            super().__init__()

        def gen_candidates(self, graph, ignored_strokes):
            return _s.tjunc_classifier_candidates(
                graph, ignored_strokes, n_closest=n_closest_t)

        def gen_labels(self, cand1, cand2, matchups_head, matchups_tail):
            return _s.tjunc_classifier_labels(cand1, cand2, matchups_head,
                                              matchups_tail)

        def gen_features(self, graph, cand1, cand2):
            # Note: One must create the features anew every time, since a
            # feature can cache drawing-specific information for efficiency.
            features = self.make_features()
            return _s.tjunc_classifier_features(graph, cand1, cand2, features)

        def gen_all_features(self, graph, cand1, cand2):
            # Note: One must create the features anew every time, since a
            # feature can cache drawing-specific information for efficiency.
            features = make_features_t_all()
            return _s.tjunc_classifier_features(graph, cand1, cand2, features)

    def __init__(self, make_features_e: Callable[[], List[_s.JunctionFeature]],
                 make_features_t: Callable[[], List[_s.JunctionFeature]]):
        self.e = Dataset.Endp(make_features_e)
        self.t = Dataset.TJunc(make_features_t)
        self.e_fea: Optional[np.ndarray] = None
        self.t_fea: Optional[np.ndarray] = None
        self.e_lab: Optional[np.ndarray] = None
        self.t_lab: Optional[np.ndarray] = None
        self.name2group_idx = dict()
        self.graphs = []

    def load(self, drawing, needs_fairing,
             labels_positive=None, labels_negative=None, name: str = None):
        if not isinstance(drawing, _s.Drawing):
            if name is None:
                name = os.path.splitext(os.path.basename(drawing))[0]
            drawing = _s.load(drawing)
            drawing.remove_duplicate_vertices()
        if needs_fairing:
            fair_drawing(drawing)
            drawing.remove_duplicate_vertices()

        def read_gt(labels):
            if labels == None:
                return None
            if isinstance(labels, dict):
                return labels
            if os.path.exists(labels):
                with open(labels) as f:
                    return json5.load(f)
            elif self.has_labels():
                raise FileNotFoundError(f'could not find "{labels}" and ' +
                                        'cannot mix annotated and unannotated ' +
                                        'drawings in the same dataset')
            return {'junctions': []}

        # graph = _s.StrokeGraph(drawing)
        # _s.dehook_dangling_edges(graph)
        graph = _s.build_plane_graph(drawing)

        self.add_drawing(graph, read_gt(labels_positive), read_gt(labels_negative))
        if name in self.name2group_idx:
            logging.warning('Drawing names should be unique; lookup may be broken')
        self.name2group_idx[name] = len(self.e.candidates1) - 1
        return {'drawing': drawing, 'graph': graph}

    def add_drawing(self, graph, gt_positive: dict, gt_negative: dict):
        if gt_positive is not None and gt_negative is None:
            pos_junctions = [from_json(j) for j in gt_positive['junctions']]
            # We have a complete set of junctions, and any junctions not present
            # are presumed negative.
            (emh, emt, tmh, tmt) = matchups_full(graph.n_orig_strokes, pos_junctions)
        else:
            # We either have a partial set of junction annotations, or we have
            # no annotations at all.
            if gt_positive is None:
                gt_positive = {'junctions': []}
            if gt_negative is None:
                gt_negative = {'junctions': []}
            pos_junctions = [from_json(j) for j in gt_positive['junctions']]
            neg_junctions = [from_json(j) for j in gt_negative['junctions']]
            (emh, emt, tmh, tmt) = \
                matchups_partial(graph.n_orig_strokes, pos_junctions, neg_junctions)

        # Indices of strokes to ignore (e.g. strokes that fill in areas).
        ignored_strokes = set(gt_positive.get('junctions_ignore', set()))
        self.e.add_drawing(graph, emh, emt, ignored_strokes)
        self.t.add_drawing(graph, tmh, tmt, ignored_strokes)
        self.graphs.append(graph)

    def gen_matrices(self, is_training: bool):
        """Cache a bunch of matrices."""
        self.e_fea = self.e.features()
        self.t_fea = self.t.features()
        self.e_fea_all = self.e.all_features()
        self.t_fea_all = self.t.all_features()
        self.e_lab = self.e.labels()
        self.t_lab = self.t.labels()
        self.e_groups = self.e.groups()
        self.t_groups = self.t.groups()
        self.e_cand1 = self.e.cand1()
        self.t_cand1 = self.t.cand1()
        self.e_cand2 = self.e.cand2()
        self.t_cand2 = self.t.cand2()

        # For training, filter out the junctions with no annotations
        if is_training:
            e_lab_known = self.e_lab != UNKNOWN
            t_lab_known = self.t_lab != UNKNOWN
            self.e_fea = self.e_fea[e_lab_known]
            self.t_fea = self.t_fea[t_lab_known]
            self.e_fea_all = self.e_fea_all[e_lab_known]
            self.t_fea_all = self.t_fea_all[t_lab_known]
            self.e_lab = self.e_lab[e_lab_known]
            self.t_lab = self.t_lab[t_lab_known]
            self.e_groups = self.e_groups[e_lab_known]
            self.t_groups = self.t_groups[t_lab_known]
            self.e_cand1 = self.e_cand1[e_lab_known]
            self.t_cand1 = self.t_cand1[t_lab_known]
            self.e_cand2 = self.e_cand2[e_lab_known]
            self.t_cand2 = self.t_cand2[t_lab_known]

    def has_labels(self) -> bool:
        return bool(self.e.lab_arrs)

    def __bool__(self):
        return bool(self.e)

    def group_index(self, drawing_name: str):
        return self.name2group_idx.get(drawing_name)

    def shuffle(self, random_state):
        """Requires `gen_matrices` be called first."""
        (self.e_fea, self.e_lab, self.e_groups, self.e_cand1, self.e_cand2) = \
            shuffle(self.e_fea, self.e_lab, self.e_groups, self.e_cand1,
                    self.e_cand2, random_state=random_state)
        (self.t_fea, self.t_lab, self.t_groups, self.t_cand1, self.t_cand2) = \
            shuffle(self.t_fea, self.t_lab, self.t_groups, self.t_cand1,
                    self.t_cand2, random_state=random_state)


def junctions_from_endp_candidates_with_duplicates(cand1_arr: np.ndarray,
                                                   cand2_arr: np.ndarray,
                                                   prob_arr: np.ndarray,
                                                   out_junctions: List[_s.Junction]):
    for (cand1, cand2, prob) in zip(cand1_arr, cand2_arr, prob_arr):
        s1 = (cand1 if cand1 >= 0 else ~cand1)
        s2 = (cand2 if cand2 >= 0 else ~cand2)
        entry = dict()
        entry['type'] = 'r'
        entry['points'] = [[s1, 0.0 if cand1 >= 0 else 1.0],
                           [s2, 0.0 if cand2 >= 0 else 1.0]]
        entry['probability'] = prob
        out_junctions.append(from_json(entry))


def _junctions_from_endp_candidates(cand1_arr: np.ndarray,
                                    cand2_arr: np.ndarray,
                                    prob_arr: np.ndarray,
                                    out_junctions: List[_s.Junction]):
    already_added = dict()
    for (cand1, cand2, prob) in zip(cand1_arr, cand2_arr, prob_arr):
        key = ((cand1, cand2) if cand1 < cand2 else (cand2, cand1))
        if key not in already_added:
            s1 = int(cand1 if cand1 >= 0 else ~cand1)
            s2 = int(cand2 if cand2 >= 0 else ~cand2)
            entry = dict()
            entry['type'] = 'r'
            entry['points'] = [[s1, 0.0 if cand1 >= 0 else 1.0],
                               [s2, 0.0 if cand2 >= 0 else 1.0]]
            entry['probability'] = prob
            out_junctions.append(from_json(entry))
            already_added[key] = out_junctions[-1]
        else:
            already_added[key].probability = \
                max(prob, already_added[key].probability)


def _junctions_from_tjunc_candidates(cand1_arr: np.ndarray,
                                     cand2_arr: np.ndarray,
                                     prob_arr: np.ndarray,
                                     out_junctions: List[_s.Junction]):
    already_added = set()
    for (cand1, cand2, prob) in zip(cand1_arr, cand2_arr, prob_arr):
        key = (cand1, int(cand2[0]))
        # NOTE (jerry): I think this check is unnecessary because there
        # shouldn't be duplicates of this type in the first place.
        if key not in already_added:
            already_added.add(key)
            s1 = (cand1 if cand1 >= 0 else ~cand1)
            s2 = int(cand2[0])
            entry = dict()
            entry['type'] = 't'
            entry['points'] = [[s2, cand2[1]],
                               [s1, 0.0 if cand1 >= 0 else 1.0]]
            entry['probability'] = prob
            out_junctions.append(from_json(entry))


class JunctionCandidate:
    cand1: int
    cand2: int | np.ndarray
    type: _s.Junction.Type
    prob: float

    def __init__(self, cand1: int, cand2: int | np.ndarray, prob: float):
        self.cand1 = int(cand1) # Convert from a numpy int type if necessary.
        if isinstance(cand2, Number):
            self.cand2 = int(cand2)
            self.type = _s.Junction.R
        else:
            assert isinstance(cand2, np.ndarray), 'cand2 must be int or ndarray'
            self.cand2 = cand2
            self.type = _s.Junction.T
        self.prob = prob

    def as_junction(self) -> _s.Junction:
        if self.type == _s.Junction.R:
            s1 = (self.cand1 if self.cand1 >= 0 else ~self.cand1)
            s2 = (self.cand2 if self.cand2 >= 0 else ~self.cand2)
            points = [[s1, 0.0 if self.cand1 >= 0 else 1.0],
                      [s2, 0.0 if self.cand2 >= 0 else 1.0]]
            return _s.Junction(points, 'r')
        elif self.type == _s.Junction.T:
            s1 = (self.cand1 if self.cand1 >= 0 else ~self.cand1)
            s2 = int(self.cand2[0])
            points = [[s2, self.cand2[1]],
                      [s1, 0.0 if self.cand1 >= 0 else 1.0]]
            return _s.Junction(points, 't')
        raise ValueError


class JunctionInfo:
    junctions: List[_s.Junction]
    metrics: Optional[OverallMetrics]
    false_positives: Optional[List[_s.Junction]]
    false_negatives: Optional[List[_s.Junction]]
    positives_e: List[_s.Junction]
    positives_t: List[_s.Junction]
    negatives_e: List[_s.Junction]
    negatives_t: List[_s.Junction]
    inconsistencies: List[_s.Junction]
    fp_candidates: List[JunctionCandidate]
    fn_candidates: List[JunctionCandidate]

    e_fea: np.ndarray
    e_prob: np.ndarray
    t_fea: np.ndarray
    t_prob: np.ndarray

    def __init__(self):
        self.junctions = []
        self.metrics = None
        self.false_positives = None
        self.false_negatives = None
        self.positives_e = []
        self.positives_t = []
        self.negatives_e = []
        self.negatives_t = []
        self.inconsistencies = []
        self.fp_candidates = []
        self.fn_candidates = []

        self.e_fea = None
        self.e_prob = None
        self.t_fea = None
        self.t_prob = None


class Models:
    def __init__(self, e_clf, t_clf):
        self.e_clf = e_clf
        self.t_clf = t_clf


def training_junctions(dataset: Dataset, group: int) -> list:
    accepted = []
    rejected = []

    e_filter = dataset.e_groups == group
    e_cand1 = dataset.e_cand1[e_filter]
    e_cand2 = dataset.e_cand2[e_filter]
    e_lab = dataset.e_lab[e_filter]
    e_lab_kno = e_lab != UNKNOWN
    e_lab = e_lab[e_lab_kno]
    e_cand1 = e_cand1[e_lab_kno]
    e_cand2 = e_cand2[e_lab_kno]

    already_added = set()
    for (cand1, cand2, lab) in zip(e_cand1, e_cand2, e_lab):
        key = ((cand1, cand2) if cand1 < cand2 else (cand2, cand1))
        if key not in already_added:
            already_added.add(key)
            arr = (accepted if lab else rejected)
            _junctions_from_endp_candidates((cand1,), (cand2,), (-1.0,), arr)

    t_filter = dataset.t_groups == group
    t_cand1 = dataset.t_cand1[t_filter]
    t_cand2 = dataset.t_cand2[t_filter]
    t_lab = dataset.t_lab[t_filter]
    t_lab_kno = t_lab != UNKNOWN
    t_lab = t_lab[t_lab_kno]
    t_cand1 = t_cand1[t_lab_kno]
    t_cand2 = t_cand2[t_lab_kno]

    for (cand1, cand2, lab) in zip(t_cand1, t_cand2, t_lab):
        arr = (accepted if lab else rejected)
        _junctions_from_tjunc_candidates((cand1,), (cand2,), (-1.0,), arr)

    return (accepted, rejected)


def junctions_from_clf(dataset: Dataset, group: int, models: Models) -> JunctionInfo:
    e_filter = dataset.e_groups == group
    t_filter = dataset.t_groups == group
    if dataset.e_fea[e_filter].shape[0] > 0:
        e_prob = models.e_clf.predict_proba(dataset.e_fea[e_filter])[:, 1]
    else:
        e_prob = np.zeros((0,))
    if dataset.t_fea[t_filter].shape[0] > 0:
        t_prob = models.t_clf.predict_proba(dataset.t_fea[t_filter])[:, 1]
    else:
        t_prob = np.zeros((0,))
    e_pred = e_prob > 0.5
    t_pred = t_prob > 0.5
    e_cand1 = dataset.e_cand1[e_filter]
    t_cand1 = dataset.t_cand1[t_filter]
    e_cand2 = dataset.e_cand2[e_filter]
    t_cand2 = dataset.t_cand2[t_filter]

    out = JunctionInfo()
    _junctions_from_tjunc_candidates(t_cand1[t_pred], t_cand2[t_pred], t_prob[t_pred], out.junctions)
    _junctions_from_endp_candidates(e_cand1[e_pred], e_cand2[e_pred], e_prob[e_pred], out.junctions)
    graph = dataset.graphs[group]

    e_features = dataset.e.make_features()
    def predict_one_e(cand1, cand2) -> bool:
        row = _s.endpoint_classifier_features(graph, np.array([cand1]), np.array([cand2]), e_features)
        return models.e_clf.predict_proba(row)[0, 1]

    if dataset.has_labels():
        out.metrics = OverallMetrics()

        # Endpoint-endpoint.

        true_positives_e_count = 0
        true_negatives_e_count = 0
        false_positives_e = []
        false_negatives_e = []

        # Filter down to known data.
        e_lab = dataset.e_lab[e_filter]
        e_lab_kno = e_lab != UNKNOWN
        e_lab = e_lab[e_lab_kno]
        e_pred = e_pred[e_lab_kno]
        e_prob = e_prob[e_lab_kno]
        e_cand1 = e_cand1[e_lab_kno]
        e_cand2 = e_cand2[e_lab_kno]

        already_added = set()
        for (cand1, cand2, pred, prob, lab) in zip(e_cand1, e_cand2, e_pred, e_prob, e_lab):
            key = ((cand1, cand2) if cand1 < cand2 else (cand2, cand1))
            if key not in already_added:
                already_added.add(key)
                other_dir_prob = predict_one_e(cand2, cand1)
                other_dir_pred = other_dir_prob > 0.5
                prob = 0.5 * (prob + other_dir_prob)
                if other_dir_pred != pred:
                    _junctions_from_endp_candidates((cand1,), (cand2,), (prob,), out.inconsistencies)
                _junctions_from_endp_candidates((cand1,), (cand2,), (prob,),
                                                out.positives_e if lab else out.negatives_e)
                pred = prob > 0.5
                if pred and lab:
                    true_positives_e_count += 1
                elif not pred and not lab:
                    true_negatives_e_count += 1
                elif pred and not lab:
                    _junctions_from_endp_candidates((cand1,), (cand2,), (prob,), false_positives_e)
                elif lab and not pred:
                    _junctions_from_endp_candidates((cand1,), (cand2,), (prob,), false_negatives_e)

        for (cand1, cand2, pred, prob, lab) in zip(e_cand1, e_cand2, e_pred, e_prob, e_lab):
            if pred and not lab:
                out.fp_candidates.append(JunctionCandidate(cand1, cand2, prob))
            elif lab and not pred:
                out.fn_candidates.append(JunctionCandidate(cand1, cand2, prob))

        out.metrics.e.tp = true_positives_e_count
        out.metrics.e.tn = true_negatives_e_count
        out.metrics.e.fp = len(false_positives_e)
        out.metrics.e.fn = len(false_negatives_e)

        # Endpoint-stroke.

        true_positives_t_count = 0
        true_negatives_t_count = 0
        false_positives_t = []
        false_negatives_t = []

        t_lab = dataset.t_lab[t_filter]
        t_lab_kno = t_lab != UNKNOWN
        t_lab = t_lab[t_lab_kno]
        t_pred = t_pred[t_lab_kno]
        t_prob = t_prob[t_lab_kno]
        t_cand1 = t_cand1[t_lab_kno]
        t_cand2 = t_cand2[t_lab_kno]

        for (cand1, cand2, pred, prob, lab) in zip(t_cand1, t_cand2, t_pred, t_prob, t_lab):
            _junctions_from_tjunc_candidates((cand1,), (cand2,), (prob,),
                                             out.positives_t if lab else out.negatives_t)
            if pred and lab:
                true_positives_t_count += 1
            elif not pred and not lab:
                true_negatives_t_count += 1
            elif pred and not lab:
                _junctions_from_tjunc_candidates((cand1,), (cand2,), (prob,), false_positives_t)
            elif lab and not pred:
                _junctions_from_tjunc_candidates((cand1,), (cand2,), (prob,), false_negatives_t)

        for (cand1, cand2, pred, prob, lab) in zip(t_cand1, t_cand2, t_pred, t_prob, t_lab):
            if pred and not lab:
                out.fp_candidates.append(JunctionCandidate(cand1, cand2, prob))
            elif lab and not pred:
                out.fn_candidates.append(JunctionCandidate(cand1, cand2, prob))

        out.metrics.t.tp = true_positives_t_count
        out.metrics.t.tn = true_negatives_t_count
        out.metrics.t.fp = len(false_positives_t)
        out.metrics.t.fn = len(false_negatives_t)

        out.false_positives = false_positives_e + false_positives_t
        out.false_negatives = false_negatives_e + false_negatives_t

        out.e_fea = dataset.e_fea_all[e_filter][e_lab_kno]
        out.t_fea = dataset.t_fea_all[t_filter][t_lab_kno]
        out.e_prob = e_prob
        out.t_prob = t_prob
        out.e_lab = e_lab
        out.t_lab = t_lab
        out.e_cand1 = e_cand1
        out.t_cand1 = t_cand1
        out.e_cand2 = e_cand2
        out.t_cand2 = t_cand2

    return out
