import os
import random
import sys

import matplotlib
import matplotlib.collections as col
import matplotlib.patches as pat
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
from adjustText import adjust_text
from bokeh.palettes import Set3

sys.path.insert(0, os.path.abspath(os.path.split(os.path.abspath(__file__))[0]+'/../'))
import _sketching as _s
from sketching.junction import merge_junctions


FACE_PALETTE = Set3[12]
JUNC_COLOR_IDX =  {
    _s.Junction.Type.T: 0,
    _s.Junction.Type.R: 1,
    _s.Junction.Type.X: 2,
}
JUNC_PALETTE = np.array(['#25f1f5', '#f5145f', '#00cf0e'])
JUNC_COLOR = {t: JUNC_PALETTE[i] for (t, i) in JUNC_COLOR_IDX.items()}

# VIZ_SCALE = 2
VIZ_SCALE = 4


def cluster_color(i: int, variance=0.0):
    hexstring = FACE_PALETTE[i]
    (r, g, b) = bytes.fromhex(hexstring[1:])
    # This random offset helps distinguish adjacent areas that come from the
    # same swatch but are from different clusters.
    rand1 = 2 * random.random() - 1
    rand2 = 2 * random.random() - 1
    rand3 = 2 * random.random() - 1
    return (np.clip(r / 255 + variance * rand1, 0.0, 1.0),
            np.clip(g / 255 + variance * rand2, 0.0, 1.0),
            np.clip(b / 255 + variance * rand3, 0.0, 1.0))


def coord_mats_to_mpl_patches(polygons: list, **kwargs):
    pc_array = []
    for shape in polygons:
        pc_array.append(plt.Polygon(shape, fill=True, lw=0))
    return col.PatchCollection(pc_array, **kwargs)


def figure(drawing, bb):
    fig = plt.figure()
    # Use this scale to make font smaller (the larger the scale is, the smaller the font is)
    fig.set_size_inches((VIZ_SCALE * 3, VIZ_SCALE * 3 * bb.height() / bb.width()))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    return fig


def draw_envelopes(ax, drawing, plot_order=False):
    if plot_order:
        cmap = matplotlib.cm.get_cmap('jet')
        for (i, stroke) in enumerate(drawing):
            polygons = _s.outline_to_polygons(stroke)
            if polygons:
                ax.add_collection(coord_mats_to_mpl_patches(
                    polygons, facecolors=cmap(1.0 - i / (len(drawing) - 1))))

        for (i, stroke) in enumerate(drawing):
            ax.text(*drawing[i].pos(0.5 * stroke.length()), str(i),
                    fontsize=2.0, horizontalalignment='center',
                    verticalalignment='center')

        # Highlight start of each stroke.
        starts = []
        for stroke in drawing:
            starts.append(stroke.xy(0))
        starts = np.array(starts)
        ax.scatter(starts[:, 0], starts[:, 1], c='#999999cc', s=0.4)
    else:
        polygons = _s.outline_to_polygons(drawing)
        if polygons:
            ax.add_collection(coord_mats_to_mpl_patches(
                polygons, facecolors='#444444'))


def plot_fitted(drawing, fitted, fname: str, junctions: list):
    bb = drawing.visual_bounds()
    fig = figure(drawing, bb)
    (ax,) = fig.axes

    # Base strokes
    draw_envelopes(ax, drawing)

    # Highlight junctions.
    if junctions:
        coords = np.zeros((len(junctions), 2))
        colors = np.zeros((len(junctions),), dtype=np.int32)
        for (i, junc) in enumerate(junctions):
            coords[i, :] = junc.position(fitted)
            colors[i] = JUNC_COLOR_IDX[junc.type]
        pc = ax.scatter(coords[:, 0], coords[:, 1], c=JUNC_PALETTE[colors],
                        s=(VIZ_SCALE * 3200 * drawing.avg_stroke_width() / bb.width()),
                        edgecolors='none')
        pc.set_alpha(0.5)

    if junctions:
        marker_scale = 1.0
        for junc in sorted(junctions):
            color = JUNC_COLOR[junc.type]
            line_segments = col.LineCollection(junc.star(fitted),
                                               linewidths=VIZ_SCALE * 0.2 * marker_scale,
                                               capstyle='round', color=color,
                                               linestyle='solid')
            ax.add_collection(line_segments)

    # Show probabilities as annotations
    texts = []
    if junctions:
        bbox_props = dict(boxstyle="round", fc="w", ec="0.1", alpha=0.3, lw=0.1)

        for junc in junctions:
            if junc.probability < 0: continue

            alt_prob_str = ''
            if junc.alt_probability >= 0:
                alt_prob_str = ',{prob:.2f}'.format(prob=junc.alt_probability)

            coord = junc.position(fitted)
            t = ax.text(coord[0], coord[1],
                '{prob:.2f}'.format(prob=junc.probability) + alt_prob_str,
                color=JUNC_COLOR[junc.type],
                size=1, bbox=bbox_props)
            #t.set_path_effects([path_effects.Stroke(linewidth=0.05, foreground='black'),
            #           path_effects.Normal()])
            texts.append(t)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_aspect('equal')
    ax.set_xlim((bb.xmin, bb.xmax))
    ax.set_ylim((bb.ymax, bb.ymin))
    if fname:
        save(fig, fname)
        plt.close(fig)
    else:
        plt.get_current_fig_manager().toolbar.pan()
        plt.show()


def plot(drawing, fname: str, junctions=None, normalize_highlight=False,
         plot_envelopes=True, marker_scale=1.0, highlight_scale=1.0, font_size=1.0,
         graph=None, plot_wireframes=True, plot_graph_vertices=False,
         plot_order=False, plot_faces=False):
    """
    Args:
        drawing: Drawing or StrokeGraph or path to VEC drawing.
        fname: File name to save to.  Set to falsy value for interactive
            display.
        junctions: Junctions to plot.
        normalize_highlight (bool): Whether to merge (and normalize) junctions
            for the circle highlight.
        plot_envelopes (bool)
        marker_scale (float): Amount to scale junction connection line widths
            by.
        font_size (>=1): Font size to show the probability numbers.
        highlight_scale: Scaling factor for faint circle that is drawn around
            junctions.
        graph (_sketching.StrokeGraph): Used to display vertices of the graph.
        plot_wireframes (bool)
        plot_graph_vertices (bool)
        plot_order (bool): Colour stroke envelopes according to drawing order.
        plot_faces (bool): Fill graph faces with colour.  Requires graph to be
            specified.
    """
    if isinstance(drawing, str):
        drawing = _s.load(drawing).duplicate_vertices_removed()
    if isinstance(drawing, _s.StrokeGraph):
        drawing = drawing.original_drawing()
    bb = drawing.visual_bounds()
    fig = figure(drawing, bb)
    (ax,) = fig.axes

    # Base strokes
    if plot_envelopes:
        draw_envelopes(ax, graph.as_drawing() if graph else drawing, plot_order=plot_order)

    if graph and plot_graph_vertices:
        coords = graph.vertex_positions(min_valence=2)
        pc = ax.scatter(coords[:, 0], coords[:, 1], c='#ffd500',
                        s=(VIZ_SCALE * 3200 * drawing.avg_stroke_width()
                           * marker_scale * marker_scale * highlight_scale / bb.width()),
                        edgecolors='none')
        pc.set_alpha(1.0)

    # Highlight junctions.
    special_junctions = None
    if junctions:
        special_junctions = [junc for junc in junctions if junc.type == _s.Junction.X]
        junctions = [junc for junc in junctions if junc.type != _s.Junction.X]
        coords = np.zeros((len(junctions), 2))
        colors = np.zeros((len(junctions),), dtype=np.int32)
        it = enumerate(merge_junctions(junctions, drawing) if normalize_highlight
                       else junctions)
        for (i, junc) in it:
            coords[i, :] = junc.position(drawing)
            colors[i] = JUNC_COLOR_IDX[junc.type]
        pc = ax.scatter(coords[:, 0], coords[:, 1], c=JUNC_PALETTE[colors],
                        s=(VIZ_SCALE * 8000 * drawing.avg_stroke_width()
                           * marker_scale * marker_scale * highlight_scale / bb.width()),
                        edgecolors='none')
        pc.set_alpha(0.1)
    if special_junctions:
        coords = np.zeros((len(special_junctions), 2))
        colors = np.zeros((len(special_junctions),), dtype=np.int32)
        it = enumerate(merge_junctions(special_junctions, drawing) if normalize_highlight
                       else special_junctions)
        for (i, junc) in it:
            coords[i, :] = junc.position(drawing)
            colors[i] = JUNC_COLOR_IDX[junc.type]
        pc = ax.scatter(coords[:, 0], coords[:, 1], c=JUNC_PALETTE[colors],
                        s=(VIZ_SCALE * 40000 * drawing.avg_stroke_width()
                           * marker_scale * marker_scale * highlight_scale / bb.width()),
                        edgecolors='none')
        pc.set_alpha(0.25)
        junctions += special_junctions

    if plot_wireframes:
        wireframes = []
        for stroke in (graph.strokes if graph else drawing):
            if len(stroke) > 0:
                xy = np.vstack((stroke.x, stroke.y)).T
                # MPL will remove the final vertex of a closed polygon if closed is false.
                wireframes.append(pat.Polygon(xy, closed=np.array_equal(xy[0], xy[-1]),
                                              fill=False))
        ax.add_collection(col.PatchCollection(wireframes, facecolors='#00000000',
                                              linewidths=VIZ_SCALE * 0.05,
                                              edgecolors='#000000'))
    if junctions:
        for junc in sorted(junctions):
            color = JUNC_COLOR[junc.type]
            line_segments = col.LineCollection(junc.star(drawing),
                                               linewidths=VIZ_SCALE * 0.2 * marker_scale,
                                               capstyle='round', color=color,
                                               linestyle='solid')
            ax.add_collection(line_segments)
    if graph and plot_graph_vertices:
        coords = graph.vertex_positions(min_valence=2)
        pc = ax.scatter(coords[:, 0], coords[:, 1], c='#ffd500', zorder=3,
                        s=VIZ_SCALE * 0.4 * marker_scale, edgecolors='none')
    if graph and plot_faces:
        if graph.n_faces - 1 < len(FACE_PALETTE):
            facecolors = list(FACE_PALETTE)
        else:
            facecolors = [cluster_color(i) for i in graph.map_color(len(FACE_PALETTE))[1:]]
        faces = []
        for fi in range(1, graph.n_faces):
            xy = graph.face_positions(fi)
            faces.append(pat.Polygon(xy, closed=True, fill=True))
        ax.add_collection(col.PatchCollection(faces, alpha=1.0, facecolors=facecolors,
                                              edgecolors=facecolors, linewidths=0))

    # Show probabilities as annotations
    texts = []
    if junctions:
        bbox_props = dict(boxstyle="round", fc="w", ec="0.1", alpha=0.3, lw=0.1)
        
        for junc in junctions:
            if junc.probability < 0: continue

            alt_prob_str = ''
            if junc.alt_probability >= 0:
                alt_prob_str = ',{prob:.2f}'.format(prob=junc.alt_probability)

            coord = junc.position(drawing)
            t = ax.text(coord[0], coord[1],
                '{prob:.2f}'.format(prob=junc.probability) + alt_prob_str,
                color=JUNC_COLOR[junc.type],
                size=font_size, bbox=bbox_props)
            #t.set_path_effects([path_effects.Stroke(linewidth=0.05, foreground='black'),
            #           path_effects.Normal()])
            texts.append(t)
        
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_aspect('equal')
    ax.set_xlim((bb.xmin, bb.xmax))
    ax.set_ylim((bb.ymax, bb.ymin))

    # adjust_text needs to be called after set lim
    # TODO: Way too slow for large drawings...
    # if len(texts) > 0:
    #     arrow_props = dict(arrowstyle='->, head_length=0.05, head_width=0.02', lw=0.1, color="0.5")
    #     adjust_text(texts, ha='left', va='center', expand_align=(1.5, 2.0),
    #                 precision=0.5, lim=10,
    #                 force_text=(0.8, 1.0), force_objects=(0.8, 1.0),
    #                 arrowprops=arrow_props)

    if fname:
        save(fig, fname)
        plt.close(fig)
    else:
        plt.get_current_fig_manager().toolbar.pan()
        plt.show()


def save(fig, fname: str):
    if os.path.dirname(fname):
        os.makedirs(os.path.dirname(fname), exist_ok=True)
    fig.savefig(fname, transparent=True, bbox_inches=0, pad_inches=0)


def _main():
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('drawing', help='drawing to visualize')
    parser.add_argument('--junctions',
                        help='path to JSON file containing junctions to visualize')
    args = parser.parse_args()

    junctions = []
    if junctions:
        with open(args.junctions) as f:
            junctions = json.load(f)['junctions']
    drawing = _s.load(args.drawing)
    scap_file = os.path.join(os.path.dirname(args.drawing),
                             os.path.splitext(os.path.basename(args.drawing))[0] + '.scap')
    if os.path.exists(scap_file):
        raise NotImplementedError('clustered drawings not supported yet')
    else:
        plot(drawing, None, junctions)


if __name__ == '__main__':
    _main()
