import itertools

import numpy as np

import _sketching as _s


def from_json(junc: dict) -> _s.Junction:
    return _s.Junction(junc['points'], junc['type'],
                       junc.get('is_weak', False),
                       float(junc.get('probability', -1)))


def classifier_prediction_from_json(pred: dict) -> _s.ClassifierPrediction:
    out = _s.ClassifierPrediction()
    out.type = (_s.Junction.T if pred['type'] == 't' else _s.Junction.R)
    out.cand1 = pred['cand1']
    out.cand2 = pred['cand2']
    out.prob = pred['prob']
    out.p_a = np.array(pred['p_a'])
    out.p_b = np.array(pred['p_b'])
    out.orig_a = pred['orig_a']
    out.orig_b = pred['orig_b']
    out.connected = pred['connected']
    out.junction_repr = pred.get('junction_repr', '')
    return out


def _to_json_junction(junction: _s.Junction) -> dict:
    out = {
        '_searchstr': ','.join(map(str, sorted([tup[0] for tup in junction]))),
        'points': list(junction),
        'type': str(junction.type).split('.')[-1].lower(),
    }
    if junction.is_weak:
        out['is_weak'] = junction.is_weak
    if junction.probability:
        out['probability'] = junction.probability
    return out


def _to_json_classifier_prediction(pred: _s.ClassifierPrediction) -> dict:
    out = {
        'type': ('t' if pred.type == _s.Junction.T else 'r'),
        'cand1': pred.cand1,
        'cand2': pred.cand2,
        'prob': pred.prob,
        'p_a': [pred.p_a[0], pred.p_a[1]],
        'p_b': [pred.p_b[0], pred.p_b[1]],
        'orig_a': [pred.orig_a[0], pred.orig_a[1]],
        'orig_b': [pred.orig_b[0], pred.orig_b[1]],
        'connected': pred.connected,
        'junction_repr': pred.junction_repr,
        'alt_prob' : pred.alt_prob,
        # TODO: Also serialize the features.
    }
    return out


def to_json(obj) -> dict:
    if isinstance(obj, _s.Junction):
        return _to_json_junction(obj)
    elif isinstance(obj, _s.ClassifierPrediction):
        return _to_json_classifier_prediction(obj)
    raise TypeError(f"don't know how to convert {obj} to JSON")


def split_junction(junc: _s.Junction) -> list:
    if len(junc) == 2:
        return [junc.clone()]
    elif junc.type == _s.Junction.R:
        out = []
        for (p1, p2) in itertools.combinations(junc, 2):
            s1 = p1[0]
            s2 = p2[0]
            out.append(_s.Junction([p1, p2] if s1 < s2 else [p2, p1], 'r'))
        return out
    elif junc.type == _s.Junction.T:
        out = []
        occluding = junc[0]
        s1 = occluding[0]
        for other in junc.points[1:]:
            s2 = other[0]
            out.append(_s.Junction([occluding, other], 't'))
        for (p1, p2) in itertools.combinations(junc.points[1:], 2):
            s1 = p1[0]
            s2 = p2[0]
            out.append(_s.Junction([p1, p2] if s1 < s2 else [p2, p1], 'r'))
        return out
    else:
        raise NotImplementedError(f"unknown junction type {junc.type}")


def _endpoint_id(point: tuple) -> int:
    s = int(point[0])
    return s if point[1] < 0.5 else ~s


def merge_junctions(junctions: list, drawing: _s.Drawing) -> set:
    """
    The "inverse" of `split_junction`.

    Junctions must be pre-split into junctions with only two points each (using
    e.g. `split_junction` from this module).
    """
    # TODO: This function still needs more testing, especially for cases with
    #       complex transitivity.
    id2junction = dict()
    for junc in junctions:
        junc = junc.normalized(drawing) # Copy, NOT equivalent to `junc.normalize()`.
        to_merge = set()
        for p in junc:
            if p in id2junction:
                to_merge.add(id2junction[p])
            id2junction[p] = junc
        for other in to_merge:
            junc.merge(other)
        for p in junc: # Re-traverse `junc` to get transitivity.
            id2junction[p] = junc
    deduplicated = set(id2junction.values())
    for junc in deduplicated:
        junc.normalize(drawing)
    # The creation of the set will automatically deduplicate based on memory
    # address, which is what we want.
    return deduplicated


def _key(junc: _s.Junction):
    junc.sort_entries()
    if junc.type == _s.Junction.R:
        return tuple(_endpoint_id(p) for p in junc)
    elif junc.type == _s.Junction.T:
        return (junc[0][0], *(_endpoint_id(p) for p in junc.points[1:]), 't')
    else:
        raise NotImplementedError(f"unknown junction type {junc.type}")


def deduplicate(junctions: list, key=_key) -> list:
    """
    Return a list with duplicate junctions removed.

    For best results, junctions should be pre-split into junctions with only two
    points each (using e.g. `split_junction` from this module).

    Args:
        junctions (list[_sketching.Junction])
        key (function)
    """
    already_added = set()
    out = []
    for junc in junctions:
        k = key(junc)
        # TODO: Non-weak junctions should override weak junctions
        if k not in already_added:
            already_added.add(k)
            out.append(junc)
    return out


def diff(new_junctions: list, old_junctions: list, key=_key) -> (list, list):
    """
    Junctions must be pre-split into junctions with only two points each (using
    e.g. `split_junction` from this module).

    Args:
        new_junctions (list[_sketching.Junction])
        old_junctions (list[_sketching.Junction])
        key (function)

    Returns:
        (added_junctions: list[_sketching.Junction],
         removed_junctions: list[_sketching.Junction])
    """
    new_dict = dict()
    old_dict = dict()
    for junc in new_junctions:
        assert len(junc) == 2
        new_dict[key(junc)] = junc
    for junc in old_junctions:
        assert len(junc) == 2
        old_dict[key(junc)] = junc
    added_keys = new_dict.keys() - old_dict.keys()
    removed_keys = old_dict.keys() - new_dict.keys()
    return ([new_dict[k] for k in added_keys], [old_dict[k] for k in removed_keys])


def intersection(new_junctions: list, old_junctions: list, key=_key) -> list:
    """
    Junctions must be pre-split into junctions with only two points each (using
    e.g. `split_junction` from this module).

    Args:
        new_junctions (list[_sketching.Junction])
        old_junctions (list[_sketching.Junction])
        key (function)
    """
    new_dict = dict()
    old_dict = dict()
    for junc in new_junctions:
        assert len(junc) == 2
        new_dict[key(junc)] = junc
    for junc in old_junctions:
        assert len(junc) == 2
        old_dict[key(junc)] = junc
    intersect = set.intersection(set(new_dict.keys()), set(old_dict.keys()))
    return [new_dict[k] for k in intersect]
