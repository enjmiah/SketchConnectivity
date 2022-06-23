#!/usr/bin/env python

"""
Train the classifiers for identifying endpoint-endpoint junctions and
T-junctions.
"""

import argparse
import logging
import os
import pickle
import subprocess
import sys

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.abspath(os.path.split(os.path.abspath(__file__))[0]+'/../python/'))
import _sketching as _s
import sketching.junc_classify as classify
from sketching.util import Timing

from config import N_ESTIMATORS, MAX_DEPTH_E, MAX_DEPTH_T

import config

TEMPLATE_DIR = os.path.dirname(__file__)
CONVERT_PY = os.path.abspath(os.path.split(os.path.abspath(__file__))[0]+'/forest_to_c.py')

ARGS = None

make_features_e = None
make_features_t = None


def main():
    global ARGS, make_features_e, make_features_t
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('index',
                        help='path to YAML index file specifying drawings to train on')
    parser.add_argument('--pickle', default='classify_junction_models.pickle',
                        help='path to output the pickled scikit-learn models')
    parser.add_argument('--cpp', default='forest.cpp',
                        help='path to output the generated C++ code for the model')
    parser.add_argument('--debug', action='store_true',
                        help='place a breakpoint before execution')
    ARGS = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if ARGS.debug:
        # Give time to attach a debugger.
        breakpoint()

    # Set up feature definitions
    make_features_e = config.make_features_e
    make_features_t = config.make_features_t

    np.random.seed(123)
    random_state = np.random.RandomState(234)

    with open(ARGS.index) as f:
        index = yaml.safe_load(f)
    root = os.path.dirname(ARGS.index)
    if index.get('cwd'):
        root = os.path.abspath(os.path.join(os.path.dirname(ARGS.index), index['cwd']))

    train_set = classify.Dataset(make_features_e, make_features_t)

    print('Loading training set...', file=sys.stderr)

    for (category, examples) in index.items():
        if category == 'cwd':
            continue
        for example in examples:
            with Timing(f'  Loading {category}/{example["id"]}'):
                path = '/'.join((root, category, example['id'] + '.vec'))
                drawing = _s.load(path)
                drawing.remove_duplicate_vertices()
                ex_dict = {**example}
                ex_dict['id'] += f'_{np.random.randint(0, 9999):04}'
                train_set.load(drawing, needs_fairing=False,
                               labels_positive=os.path.join(root, example['label_accepted']),
                               labels_negative=(os.path.join(root, example['label_rejected'])
                                                if 'label_rejected' in example else None),
                               name=ex_dict['id'])

    np.random.seed(123)
    random_state = np.random.RandomState(234)
    train_set.gen_matrices(is_training=True)

    with Timing('Fitting final classifiers'):
        # Endpoint-endpoint classifier.
        e_clf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH_E,
            class_weight=(None),
            random_state=random_state)
        # Endpoint-stroke classifier.
        t_clf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH_T,
            class_weight=(None),
            random_state=random_state)

        e_clf = e_clf.fit(train_set.e_fea, train_set.e_lab)
        t_clf = t_clf.fit(train_set.t_fea, train_set.t_lab)

    with open(ARGS.pickle, 'wb') as f:
        pickle.dump({'endpoint': e_clf, 'tjunction': t_clf}, f)

    # Convert the trained model into a cpp file that can be used by our method.
    convert_cmd = [sys.executable, CONVERT_PY, ARGS.pickle, ARGS.cpp]
    print('Running', ' '.join(convert_cmd))
    subprocess.check_call(convert_cmd)


if __name__ == '__main__':
    main()
