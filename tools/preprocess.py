#!/usr/bin/env python

"""
Example pre-processing script.

The actual processing you need to perform will depend on the characteristics of
your input data.

Format of the input file:

  drawings:
  - path: category/name.vec # Path to drawing relative to input file.
    needs_fairing: True     # Whether to smooth the input strokes.
                            # Defaults to False.
    consolidate: True       # Whether to consolidate. Defaults to True.
    dehook_factor: 1        # How aggressive to be with dehooking.
                            # Defaults to 1.
                            # Set to 0 to disable.
    chain: True             # Whether to merge strokes across valence 2 vertices.
                            # Defaults to True.
    filter_invisible: True  # Whether to filter out invisible and barely visible strokes.
                            # Defaults to True.
  - other_category/foo.vec  # Short form. Uses all the defaults.
"""

import argparse
import os
import sys

import yaml

import _sketching as _s
sys.path.insert(0, os.path.abspath(os.path.split(os.path.abspath(__file__))[0] + '/../python'))
from sketching.util import Timing


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', help='YAML file listing inputs to pre-process')
    parser.add_argument('-o', '--output', default='preprocessed',
                        help='output directory')
    args = parser.parse_args()

    with open(args.input) as f:
        input_ = yaml.safe_load(f)

    os.makedirs(args.output, exist_ok=True)

    root = input_.get('root', os.path.dirname(args.input))
    for path in input_['drawings']:
        needs_fairing = False
        dehook_factor = 1
        consolidate = True
        filter_invisible = True
        chain = True
        if isinstance(path, dict):
            needs_fairing = path.get('needs_fairing', needs_fairing)
            dehook_factor = path.get('dehook_factor', dehook_factor)
            consolidate = path.get('consolidate', consolidate)
            chain = path.get('chain', chain)
            filter_invisible = path.get('filter_invisible', filter_invisible)
            path = path['path']

        out_path = os.path.join('preprocessed', path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with Timing(os.path.splitext(path)[0]):
            # Load the raw strokes.
            drawing = _s.load(os.path.join(root, path))

            # Remove duplicate and very-close together vertices.
            drawing.remove_duplicate_vertices()

            # Smooth the strokes.
            # Only necessary if the strokes contain high-frequency noise.
            if needs_fairing:
                _s.smooth_drawing_box3(drawing, iterations=1)
                drawing.remove_duplicate_vertices()

            # Remove hooks at the ends of strokes using the heuristic described
            # in the supplementary materials.
            if dehook_factor > 0:
                _s.dehook_strokes(drawing, factor=dehook_factor)

            # Consolidate strokes.
            # Only needed if the drawing contains overdrawn strokes.
            if consolidate:
                drawing = _s.cut_at_corners(drawing)
                drawing = _s.consolidate_with_chaining_improved(drawing)
                drawing.remove_duplicate_vertices()

            # Merge strokes if their endpoints join at a valence 2 vertex.
            if chain:
                graph = _s.build_plane_graph(drawing)
                (drawing, _mapping_arr) = _s.chained_drawing(graph)

            # Filter out invisible and barely visible strokes.
            if filter_invisible:
                drawing = _s.remove_strokes_visual(
                    drawing, average_area_threshold_proportion=0.05,
                    stroke_area_threshold_proportion=0.05)

            drawing.save(out_path)


if __name__ == '__main__':
    main()
