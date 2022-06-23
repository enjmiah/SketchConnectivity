#!/usr/bin/env python

"""
Convert the sklearn endpoint-endpoint and endpoint-stroke classifiers into C++
code.
"""

import argparse
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_text, _tree


def tree_to_c(tree, function_name: str) -> str:
    tree_ = tree.tree_

    string_builder = []
    string_builder.append(f"static Float {function_name}(const Float* feature_vec) {{")

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            threshold = tree_.threshold[node]
            feature = tree_.feature[node]
            string_builder.append(f"{indent}if (feature_vec[{feature}] <= {threshold}) {{")
            recurse(tree_.children_left[node], depth + 1)
            string_builder.append(f"{indent}}} else {{")
            recurse(tree_.children_right[node], depth + 1)
            string_builder.append(f"{indent}}}")
        else:
            val = tree_.value[node, 0, 1] / tree_.value[node, 0].sum()
            string_builder.append(f"{indent}return {val};")

    recurse(0, 1)

    string_builder.append('}\n')

    return '\n'.join(string_builder)


def forest_to_c(clf: RandomForestClassifier, function_name: str) -> str:
    string_builder = []
    for i, est in enumerate(clf.estimators_):
        string_builder.append(tree_to_c(est, f'{function_name}_tree_{i}'))
        # string_builder.append(export_text(est))
    if function_name.endswith('endpoint'):
        param_type = 'EndEndFeatures'
    elif function_name.endswith('tjunction'):
        param_type = 'EndStrokeFeatures'
    else:
        raise ValueError(f'unknown parameter type for function "{function_name}"')
    string_builder.append(f'Float {function_name}(const {param_type}& feature_vec) {{')
    string_builder.append(
        f'  static_assert(sizeof(feature_vec.data_) / sizeof(feature_vec.data_[0]) == {clf.n_features_});')
    string_builder.append(f'  const Float* feat = feature_vec.data_;')
    string_builder.append('  Float pred = 0.0;')
    for i in range(len(clf.estimators_)):
        string_builder.append(  f'  pred += {function_name}_tree_{i}(feat);')
    string_builder.append(f'  return pred / {len(clf.estimators_)}.0;\n}}\n')
    return '\n'.join(string_builder)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('classifier', help='path to pickled classifier')
    parser.add_argument('out', help='path to C++ file to output')
    args = parser.parse_args()

    with open(args.classifier, 'rb') as f:
        clf = pickle.load(f)
    e_clf = clf['endpoint']
    t_clf = clf['tjunction']

    string_builder = []
    string_builder.append('// Automatically generated using forest_to_c.py.')
    string_builder.append('// Do not edit.\n')
    string_builder.append('#include <sketching/classifier.h>\n')
    string_builder.append('namespace sketching {\n')
    string_builder.append(forest_to_c(e_clf, 'clf_endpoint'))
    string_builder.append(forest_to_c(t_clf, 'clf_tjunction'))
    string_builder.append('} // namespace sketching')
    with open(args.out, 'w') as f:
        f.write('\n'.join(string_builder))


if __name__ == '__main__':
    main()
