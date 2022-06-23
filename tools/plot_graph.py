#!/usr/bin/env python

"""
Visualize the inferred stroke graph for a drawing.
"""

import argparse
import os
import random
import sys
import shutil
import subprocess
from tempfile import TemporaryDirectory

import jinja2
import matplotlib.pyplot as plt
import matplotlib.collections as col
import matplotlib.patches as pat
import numpy as np
import yaml
from bokeh.palettes import Set3

import _sketching as _s
sys.path.insert(0, os.path.abspath(os.path.split(os.path.abspath(__file__))[0]+'/../python/'))
from sketching import plot as pl
from sketching.plot import plot
from sketching.util import Timing, fair_savgol


ARGS = None
INTERACTIVE = False
PALETTE = Set3[12]
TEMPLATE = r'''
\pdfsuppresswarningpagegroup=1

\documentclass[10pt,letterpaper]{article}
\usepackage[paperheight=40in, paperwidth=8in, margin=4mm]{geometry}
\usepackage{graphicx}
\usepackage{listings}
\linespread{0}
\setlength{\parindent}{0em}
\setlength{\parskip}{0em}

\begin{document}

{% for (name, record) in drawings.items() %}
  \begin{centering}\lstinline!{{ name }}!\end{centering} \\*
  {{ tex_graphics(record['original']) }}
  {{ tex_graphics(record['graph']) }} \\
{% endfor %}

\end{document}
'''


def tex_graphics(path: str) -> str:
    path = path.replace('\\', '/')
    return fr'\includegraphics[width=0.49\textwidth]{{{path}}}'


def cluster_color(i: int):
    hexstring = PALETTE[i]
    (r, g, b) = bytes.fromhex(hexstring[1:])
    # This random offset helps distinguish adjacent areas that come from the
    # same swatch but are from different clusters.
    RAND_FAC = 0.1
    rand1 = 2 * random.random() - 1
    rand2 = 2 * random.random() - 1
    rand3 = 2 * random.random() - 1
    return (np.clip(r / 255 + RAND_FAC * rand1, 0.0, 1.0),
            np.clip(g / 255 + RAND_FAC * rand2, 0.0, 1.0),
            np.clip(b / 255 + RAND_FAC * rand3, 0.0, 1.0))


def plot_graph(ax, graph: _s.StrokeGraph, bb, interactive: bool, 
               debug=False, show_sliver=False):
    faces = []
    assert graph.boundary_face == 0
    collapsing_threshold = 0.1
    stroke_width_scale = 0.8
    if graph.n_faces - 1 < len(PALETTE):
        facecolors = list(PALETTE)
    else:
        facecolors = [cluster_color(i) for i in _s.map_color(graph, len(PALETTE))[1:]]

    for fi in range(1, graph.n_faces):
        xy = graph.face_positions(fi)
        faces.append(pat.Polygon(xy, closed=True, fill=True))

        if show_sliver:
            is_small = graph.is_face_collapsible_clipping(fi,
                collapsing_threshold, stroke_width_scale)
            if is_small:
                facecolors[fi-1] = '#ff0000ff'
                coord = np.sum(xy, axis=0) / len(xy)
                ax.text(coord[0], coord[1], f'{fi}',
                        color='b', size=1)
        # ax.text(*np.mean(xy, axis=0), str(fi), color='#0000ff')
    
    # facecolors = ['#ff000000', '#00ff0000', '#0000ff00',
    #               '#77000000', '#00770000', '#00007700',
    #               '#ffff0000', '#00ffffaa', '#ff00ff00',
    #               '#77770000', '#00777700', '#77007700']
    ax.add_collection(col.PatchCollection(faces, alpha=0.5, facecolors=facecolors,
                                          edgecolors=facecolors, linewidths=0))

    wireframes = []
    for (_i, stroke) in enumerate(graph.strokes):
        if len(stroke) > 0:
            xy = np.vstack((stroke.x, stroke.y)).T
            # MPL will remove the final vertex of a closed polygon if closed is false.
            wireframes.append(pat.Polygon(xy, closed=np.array_equal(xy[0], xy[-1]),
                                          fill=False, zorder=3))
            # ax.text(*stroke.pos(0.5 * stroke.length()), str(_i), color='#bb6600')
    ax.add_collection(col.PatchCollection(wireframes, facecolors='#00000000',
                                          linewidths=0.1, edgecolors='black',
                                          capstyle='round'))

    vertex_positions = graph.vertex_positions(min_valence=1)
    ax.scatter(vertex_positions[:, 0], vertex_positions[:, 1],
               (4 if interactive else 0.3), 'black', linewidths=0)

    if debug:
        (coords, labels) = graph.text_labels()
        for ((x, y, offx, offy), label) in zip(coords, labels):
            ax.annotate(label, (x, y),
                        xytext=(22.0 * offx - 15.0, -22.0 * offy - 3.0),
                        textcoords='offset points', fontsize=7.0, family='Arial')


def plot_drawing(ax, drawing: _s.Drawing):
    pc_array = []
    for (_i, stroke) in enumerate(drawing):
        for coords in _s.outline_to_polygons(stroke):
            pc_array.append(plt.Polygon(coords, lw=0, fill=True))
        # ax.text(*stroke.pos(0.5 * stroke.length()), str(_i), color='#bb6600')
    pc = col.PatchCollection(pc_array, facecolors='#22222222')

    ax.add_collection(pc)


def compile_pdf(show_sliver=False):
    with open(ARGS.index) as f:
        index = yaml.safe_load(f)
    root =  os.path.abspath(os.path.dirname(ARGS.index))
    if index.get('cwd') is not None:
        root = os.path.abspath(os.path.join(os.path.dirname(ARGS.index), index['cwd']))

    drawings = dict()
    with Timing('Loading dataset'):
        for (category, examples) in index.items():
            if category == 'cwd':
                continue
            for example in examples:
                path = '/'.join((root, category, example['id'] + '.vec'))
                drawing = _s.load(path)
                drawing.remove_duplicate_vertices()
                if example.get('needs_fairing', False):
                    fair_savgol(drawing)
                    drawing.remove_duplicate_vertices()
                drawings[example['id']] =  {
                    'drawing': drawing,
                    'original': '/'.join((root, category, '_output',
                                          example['id'] + '.orig.pdf'))
                }

    with Timing('Plotting originals'):
        for (name, record) in drawings.items():
            if not os.path.exists(record['original']):
                plot(record['drawing'], record['original'])

    # tmpdir = 'graph-tmp'; os.makedirs(tmpdir, exist_ok=True)
    with TemporaryDirectory() as tmpdir:
        for (name, record) in drawings.items():
            drawing = record['drawing']
            with Timing(f"Processing '{name}'"):
                graph = _s.StrokeGraph(drawing)
                bb = drawing.visual_bounds()
                fig = pl.figure(drawing, bb)
                (ax,) = fig.axes
                plot_drawing(ax, drawing)
                plot_graph(ax, graph, bb, debug=ARGS.debug, interactive=False, show_sliver=show_sliver)
                ax.set_aspect('equal')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.set_xlim((bb.xmin, bb.xmax))
                ax.set_ylim((bb.ymax, bb.ymin))
                out_file = name + '.graph.pdf'
                plt.savefig(os.path.join(tmpdir, out_file))
                plt.close(fig)
            record['graph'] = out_file
        template = jinja2.Template(TEMPLATE)
        with open(os.path.join(tmpdir, 'stroke-graphs.tex'), 'w') as f:
            f.write(template.render(drawings=drawings, tex_graphics=tex_graphics))
        subprocess.check_call(['latexmk', '-quiet', '-pdf', 'stroke-graphs.tex'],
                              cwd=tmpdir)
        shutil.move(os.path.join(tmpdir, 'stroke-graphs.pdf'), 'stroke-graphs.pdf')


def main():
    global ARGS
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # parser.add_argument('drawing', help='path to a drawing')
    parser.add_argument('index', help='path to a METADATA.yml')
    parser.add_argument('--debug', action='store_true', help='debug view')
    parser.add_argument('--sliver', action='store_true', help='visualize the determined sliver face')
    parser.add_argument('-o', '--output', required=False,
                        help='output path for vector graphics complex')
    ARGS = parser.parse_args()

    if ARGS.debug:
        # Give time to attach a debugger.
        breakpoint()

    if ARGS.index.endswith(('.yaml', '.yml')):
        compile_pdf(show_sliver=ARGS.sliver)
    else:
        drawing = _s.load(ARGS.index).duplicate_vertices_removed()
        # new_drawing = _s.Drawing()
        # new_drawing.add(drawing[8])
        # new_drawing.add(drawing[10])
        # drawing = new_drawing
        graph = _s.StrokeGraph(drawing)

        if ARGS.debug:
            print(graph)
        bb = drawing.visual_bounds()
        fig = plt.figure(figsize=(8, 8))
        ax = fig.gca()
        plot_drawing(ax, drawing)
        plot_graph(ax, graph, bb, interactive=True, debug=ARGS.debug, show_sliver=ARGS.sliver)
        ax.axis('equal')
        ax.invert_yaxis()

        if ARGS.output:
            graph.save(ARGS.output)

        plt.tight_layout()
        plt.get_current_fig_manager().toolbar.pan()
        plt.show()


if __name__ == '__main__':
    main()
