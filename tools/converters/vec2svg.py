#!/usr/bin/env python

"""
Example of converting the polylines in a VPaint VEC file to SVG.

Note that SVG does not support variable-width strokes, thus strokes will display
in viewers with uniform widths.  The width of each vertex polyline is stored as
space-separated values in a custom `data-width` attribute of each <path>
element.
"""

import argparse
import os
import sys
from glob import glob

import _sketching as _s


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', nargs='+', help='paths to VEC files to convert')
    parser.add_argument('-o', '--output', help='output directory')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    for expr in args.files:
        for path in glob(expr):
            name = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(args.output, name + '.svg')
            drawing = _s.load(path)
            print(f'Saving {out_path}...', end='', file=sys.stderr)
            _s.save_svg(drawing, out_path)
            print(' ✓', file=sys.stderr)


if __name__ == '__main__':
    main()
