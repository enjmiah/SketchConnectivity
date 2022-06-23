#!/usr/bin/env python

"""
Interactive VEC viewer.
"""

import argparse

import matplotlib.pyplot as plt
import matplotlib.collections as col
from matplotlib.patches import Polygon
import numpy as np

import _sketching as _s


def _axes_add_stroke(ax, stroke: _s.Stroke, color):
    pc_array = [plt.Polygon(coords, lw=0, fill=True)
                for coords in _s.outline_to_polygons(stroke)]
    pc = col.PatchCollection(pc_array, facecolors=color)
    ax.add_collection(pc)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('drawing', help='path to a VEC file')
    parser.add_argument('--spines', action='store_true',
                        help='visualize stroke centerlines')
    args = parser.parse_args()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    drawing = _s.load(args.drawing)
    wireframes = []
    for (i, stroke) in enumerate(drawing):
        facecolors = '#aaaaaa' if args.spines else '#000000'
        _axes_add_stroke(ax, stroke, facecolors)

        if args.spines:
            xy = np.vstack((stroke.x, stroke.y)).T
            wireframes.append(Polygon(xy, closed=False, fill=False))
    if args.spines:
        ax.add_collection(col.PatchCollection(wireframes, facecolors='#00000000',
                                              linewidths=0.2, edgecolors='black'))

    bb = drawing.bvh().bb
    ax.set_xlim(bb.xmin, bb.xmax)
    ax.set_ylim(bb.ymin, bb.ymax)
    ax.axis('equal')
    ax.invert_yaxis()
    plt.tight_layout()

    plt.get_current_fig_manager().toolbar.pan()
    plt.show()


if __name__ == '__main__':
    main()
