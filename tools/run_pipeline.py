#!/usr/bin/env python

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

import _sketching as _s
sys.path.insert(0, os.path.abspath(os.path.split(os.path.abspath(__file__))[0]+'/../python/'))
from sketching.util import Timing
from sketching.junction import to_json, classifier_prediction_from_json

def color_by_reference(*args):
    if not _s.color_by_reference(*args):
        print('WARNING: Map colouring failed. Some adjacent ' +
              'regions will be assigned the same colour.',
              file=sys.stderr)


def to_junctions(all_candidates):
    annotations = []
    for junc in all_candidates:
        pred = _s.Junction([junc.orig_a, junc.orig_b], 't')
        pred.type = junc.type
        pred.probability = junc.prob
        pred.alt_probability = junc.alt_prob
        pred.repr = junc.junction_repr
        annotations.append(pred)
    return annotations


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', help='Input vec')
    parser.add_argument('-s', '--snapshot', help='snapshot path')
    parser.add_argument('--root', type=Path)
    parser.add_argument('--tmpdir', type=Path)
    parser.add_argument('--debug', action='store_true',
                        help='place a breakpoint before execution')
    args = parser.parse_args()

    if args.debug:
        # Give time to attach a debugger.
        breakpoint()

    if args.snapshot:
        os.makedirs(args.snapshot + "/plane", exist_ok=True)
        os.makedirs(args.snapshot + "/candidates", exist_ok=True)
        os.makedirs(args.snapshot + "/v4", exist_ok=True)
        os.makedirs(args.snapshot + "/v5", exist_ok=True)
        os.makedirs(args.snapshot + "/v7", exist_ok=True)
    else:
        os.makedirs("snapshot/plane", exist_ok=True)
        os.makedirs("snapshot/candidates", exist_ok=True)
        os.makedirs("snapshot/v4", exist_ok=True)
        os.makedirs("snapshot/v5", exist_ok=True)
        os.makedirs("snapshot/v7", exist_ok=True)
        args.snapshot = 'snapshot'

    num_intermediate_states = 5

    root = args.root
    path = args.input
    tmpdir = args.tmpdir

    max_per_stroke_states = 5

    regular_plot_params = _s.PlotParams()
    regular_plot_params.show_prediction_index = False;
    regular_plot_params.viz_faces = True
    
    regions_plot_params = _s.PlotParams()
    regions_plot_params.viz_centerlines = False
    regions_plot_params.viz_envelopes = False
    regions_plot_params.connection_width = 500
    regions_plot_params.opaque_faces = True

    annotations_only_plot_params = _s.PlotParams()
    annotations_only_plot_params.viz_centerlines = False
    annotations_only_plot_params.viz_envelopes = False
    annotations_only_plot_params.viz_faces = False
    annotations_only_plot_params.show_prediction_index = False

    connections_only_plot_params = _s.PlotParams()
    connections_only_plot_params.disconnection_color = 0x981ceb
    connections_only_plot_params.viz_centerlines = False
    connections_only_plot_params.viz_envelopes = False
    connections_only_plot_params.viz_faces = False
    connections_only_plot_params.show_prediction_index = False

    centerlines_only_plot_params = _s.PlotParams()

    centerlines_only_plot_params.show_prediction_index = False
    centerlines_only_plot_params.viz_centerlines = False
    centerlines_only_plot_params.envelope_fill = 0x000000

    endpoints_only_plot_params = _s.PlotParams()
    endpoints_only_plot_params.viz_centerlines = False
    endpoints_only_plot_params.viz_ends = True
    endpoints_only_plot_params.viz_envelopes = False
    endpoints_only_plot_params.viz_faces = False
    endpoints_only_plot_params.show_prediction_index = False
    endpoints_only_plot_params.compress = False

    max_colors = len(_s.get_color_palette())

    entry = {'id': os.path.splitext(path)[0]}
    name = os.path.basename(entry['id'])

    path = os.path.join(root, path)
    drawing = _s.load(path)
    drawing.remove_duplicate_vertices()
    media_box = drawing.visual_bounds()
    visual_width = media_box.width()
    # Add some padding so that annotations near the edge don't get cut off.
    media_box.xmin -= 0.015 * visual_width
    media_box.xmax += 0.015 * visual_width
    media_box.ymin -= 0.015 * visual_width
    media_box.ymax += 0.015 * visual_width

    regular_plot_params.media_box = media_box
    regions_plot_params.media_box = media_box
    annotations_only_plot_params.media_box = media_box
    connections_only_plot_params.media_box = media_box
    endpoints_only_plot_params.media_box = media_box
    centerlines_only_plot_params.media_box = media_box

    entry['consolidated'] = os.path.join(tmpdir, f'{name}_consolidated.pdf')
    _s.save_pdf(drawing, entry['consolidated'], regular_plot_params)

    entry['consolidated_centerlines'] =  os.path.join(
        tmpdir, f'{name}_consolidated_centerlines.pdf')
    _s.save_pdf(drawing, entry['consolidated_centerlines'],
                centerlines_only_plot_params)

    #############
    # Version 0 #
    #############
    # Plane graph drawn as faces on input strokes
    graph_v0 = _s.build_plane_graph(drawing)
    bridge_info = _s.find_bridge_locations(graph_v0)
    _s.augment_with_bridges(graph_v0, bridge_info)

    # Read final result to color based on the final correspondence
    graph_color_ref = None
    face_colors_ref = None
    graph_color_ref = graph_v0

    face_colors_ref = _s.map_color(graph_color_ref, max_colors)
    face_colors_v0 = np.zeros(graph_v0.n_faces, dtype=face_colors_ref.dtype)
    color_by_reference(max_colors, graph_color_ref, face_colors_ref,
                       graph_v0, face_colors_v0)

    entry['v0'] = os.path.join(tmpdir, f'{name}_v0.pdf')
    _s.save_pdf(graph_v0, entry['v0'], regular_plot_params,
                face_colors=face_colors_v0)
    entry['v0_regions'] = os.path.join(tmpdir, f'{name}_v0_regions.pdf')
    _s.save_pdf(graph_v0, entry['v0_regions'], regions_plot_params,
                face_colors=face_colors_v0)

    _s.save_json(graph_v0, f"{args.snapshot}/plane/{name}.json")

    dangling_candidates = _s.vanilla_candidates(graph_v0, False)
    for candidate in dangling_candidates:
        candidate.probability = 0
    with open(f'{args.snapshot}/candidates/{name}.json', 'w') as f:
        json.dump({
            'junctions': [to_json(c) for c in dangling_candidates],
        }, f)

    #############
    # Version 4 #
    #############
    predictions = []
    # Original stroke based feature computation
    graph_v4 = _s.StrokeGraph(_s.StrokeGraph.SnappingType.Connection)
    with Timing('V4'):
        failed = False
        intermediate_folder = os.path.join(tmpdir, name)
        os.makedirs(intermediate_folder, exist_ok=True)
        try:
            sol_state, graphs, predictions, _ = _s.nonincremental_nonregion_solve(drawing, 
                _s.StrokeGraph.SnappingType.Connection, _s.FeatureType.OrigStroke, 
                len(drawing), num_intermediate_states, str(intermediate_folder), True, bridge=True)
        except Exception as e:
            failed = True

        if failed:
            raise Exception('Large component')

        graph_v4 = graphs[sol_state]
        predictions = predictions[sol_state]
        predictions = predictions.predictions

    for pred in predictions:
        _s.prediction_use_original_positions(graph_v4, pred)

    _s.save_json(graph_v4, f"{args.snapshot}/v4/{name}.json")
    with open(f"{args.snapshot}/v4/{name}.predjson", 'w') as f:
        json.dump([to_json(pred) for pred in predictions], f)

    for pred in predictions:
        pred.alt_prob = -1
        if pred.junction_repr != '':
            pred.connected = True

    # Read the largest non region gap
    if len(predictions) == 0:
        largest_non_region_gap = 1e6
    else:
        largest_non_region_gap = predictions[0].alt_prob

    # Use the type color
    disconnected_predictions = [pred for pred in predictions if not pred.connected and pred.prob > 0]
    cc_connected_predictions = [pred for pred in predictions if pred.connected]
    for pred in cc_connected_predictions:
        pred.prob = -1
        pred.connected = False
    for pred in disconnected_predictions:
        pred.connected = True

    #############
    # Version 5 #
    #############
    # Solve with the corner candidates given the partial result
    with Timing('V5'):
        graph_v5, predictions, all_prediction = _s.corner_solve(graph_v0, graph_v4,
            _s.StrokeGraph.SnappingType.Connection, _s.FeatureType.OrigStroke, 
            5, True, include_prev_connections=True, largest_non_region_gap=largest_non_region_gap)
    predictions = predictions.predictions

    # This one contains all actual junctions
    all_prediction = all_prediction.predictions
    # predictions = [p for p in predictions if p.connected]
    for pred in predictions:
        if pred.corner_type != _s.Junction.Type.X:
            pred.type = pred.corner_type

    for pred in predictions:
        _s.prediction_use_original_positions(graph_v5, pred)
        pred.alt_prob = -1
        if pred.junction_repr != '':
            pred.connected = True
    for pred in all_prediction:
        _s.prediction_use_original_positions(graph_v5, pred)

    _s.save_json(graph_v5, f"{args.snapshot}/v5/{name}.json")
    with open(f"{args.snapshot}/v5/{name}.predjson", 'w') as f:
        json.dump([to_json(pred) for pred in all_prediction], f)

    for pred in all_prediction:
        _s.prediction_use_original_positions(graph_v5, pred)

    #############
    # Version 6 #
    #############
    # Solve with single connection and non-dangling-non-dangling taking region into account
    graph_v6 = graph_v5.clone()

    json_name = f"{args.snapshot}/v4/{name}.predjson"
    active_candidates = []
    with open(json_name) as f:
        active_candidates_corner = json.load(f)
        active_candidates_corner = [classifier_prediction_from_json(p) for p in active_candidates_corner]
        active_candidates_corner = [p for p in active_candidates_corner]
        active_candidates += active_candidates_corner

    json_name = f"{args.snapshot}/v5/{name}.predjson"
    with open(json_name) as f:
        active_candidates_corner = json.load(f)
        active_candidates_corner = [classifier_prediction_from_json(p) for p in active_candidates_corner]
        active_candidates_corner = [p for p in active_candidates_corner]
        active_candidates += active_candidates_corner

    active_predictions = [pred for pred in active_candidates]
    v4_v5_predictions = [pred for pred in active_predictions if pred.connected]
    active_candidates = to_junctions(active_candidates)

    bridges = []
    for i in range(1, 12):
        with Timing(f'  Finding bridges (round {i})'):
            bridge_info = _s.find_final_bridge_locations(graph_v6, active_predictions, round=i)
            bridges += _s.augment_with_final_bridges(graph_v6, bridge_info, round=i)
    
    def to_annotations(all_candidates, graph):
        annotations = []
        for junc in all_candidates:
            pred = _s.ClassifierPrediction()
            pred.type = junc.type
            # FIXME: Set cand1 and cand2 properly!
            pred.cand1 = 0
            pred.cand2 = 0
            pred.prob = junc.probability
            pred.alt_prob = -1
            pred.orig_a = junc.points[0]
            pred.orig_b = junc.points[1]
            pred.connected = True
            pred.junction_repr = junc.repr
            _s.prediction_use_graph_positions(graph, pred)
            annotations.append(pred)
        return annotations
    predictions = to_annotations(bridges, graph_v0)

    for pred in predictions:
        if pred.corner_type != _s.Junction.Type.X:
            pred.type = pred.corner_type

    for pred in predictions:
        _s.prediction_use_original_positions(graph_v6, pred)
        pred.alt_prob = -1
        if pred.junction_repr != '':
            pred.connected = True
    bridge_predictions = [pred for pred in predictions]

    # Connection from the previous solve steps
    corner_connected_predictions = [pred for pred in active_predictions if pred.connected]
    tmp_prob = []
    for pred in corner_connected_predictions:
        tmp_prob.append(pred.prob)
        pred.prob = -1
        pred.connected = False

    for i, pred in enumerate(corner_connected_predictions):
        pred.prob = tmp_prob[i]

    #############
    # Version 7 #
    #############
    # Solve with connection pairs taking region into account
    discrete_settings = [
        (0.45,  3.0),
        (0.40,  5.0),
        (0.35,  7.0),
        (0.30,  9.0),
        (0.25, 11.0),
        (0.20, 13.0),
        (0.15, 15.0),
        (0.10, 17.0),
        (0.05, 19.0),
        (0.00, 21.0)
    ]
    varying_graph = graph_v6.clone()
    varying_candidates = [junc for junc in active_candidates]
    multi_bridge_candidates = []
    predictions = []
    for i, setting in enumerate(discrete_settings):
        with Timing(f''):
            low_prob, diameter_gap_ratio = setting
            graph_v7, new_predictions = _s.multi_bridge(varying_graph, varying_candidates,
                varying_graph, diameter_gap_ratio, low_prob,
                _s.StrokeGraph.SnappingType.Connection, _s.FeatureType.OrigStroke, 
                True, largest_non_region_gap=float('inf'))
            new_predictions = new_predictions.predictions
            predictions += new_predictions
            new_candidates = to_junctions(new_predictions)
            multi_bridge_candidates += new_candidates
            for junc in varying_candidates:
                for junc2 in new_candidates:
                    if str((junc.points[0], junc.points[1])) == str((junc2.points[0], junc2.points[1])) or \
                        str((junc.points[0], junc.points[1])) == str((junc2.points[1], junc2.points[0])):
                        junc.repr = junc2.repr
                        # print(junc)

            varying_graph = graph_v7.clone()

    # This one contains all actual junctions
    for pred in predictions:
        if pred.corner_type != _s.Junction.Type.X:
            pred.type = pred.corner_type

    for pred in predictions:
        _s.prediction_use_original_positions(graph_v7, pred)
        pred.alt_prob = -1
        if pred.junction_repr != '':
            pred.connected = True

    v7_predictions = [pred for pred in predictions if pred.connected]

    _s.save_json(graph_v7, f"{args.snapshot}/v7/{name}.json")
    with open(f"{args.snapshot}/v7/{name}.predjson", 'w') as f:
        json.dump([to_json(pred) for pred in predictions], f)

    face_colors_v7 = np.zeros(graph_v7.n_faces, dtype=face_colors_v0.dtype)
    color_by_reference(max_colors, graph_color_ref, face_colors_ref,
                       graph_v7, face_colors_v7)

    entry['v7'] = os.path.join(tmpdir, f'{name}_v7.pdf')
    _s.save_pdf(graph_v7, entry['v7'], regular_plot_params,
                face_colors=face_colors_v7)

    entry['v7_regions'] = os.path.join(tmpdir, f'{name}_v7_regions.pdf')
    _s.save_pdf(graph_v7, entry['v7_regions'], regions_plot_params,
                face_colors=face_colors_v7)
    entry['v7_annotations_conn_all'] = os.path.join(tmpdir, f'{name}_v7_annotations_conn_all.pdf')
    _s.save_pdf(graph_v7, entry['v7_annotations_conn_all'],
                annotations_only_plot_params,
                predictions=[pred for pred in predictions if pred.prob > 0])

    # Connection from the previous solve steps
    bridge_connected_predictions = corner_connected_predictions
    for pred in bridge_connected_predictions:
        pred.prob = -1
        pred.connected = False
    entry['v7_cc_conn'] = os.path.join(tmpdir, f'{name}_v7_cc_conn.pdf')
    _s.save_pdf(graph_v7, entry['v7_cc_conn'],
                connections_only_plot_params,
                predictions=bridge_connected_predictions)

    print('Entry: {}'.format(str(entry)))


if __name__ == '__main__':
    main()
