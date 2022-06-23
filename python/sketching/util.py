import sys
import time

import numpy as np
from scipy.signal import savgol_filter

import _sketching as _s


class Timing:
    def __init__(self, msg=''):
        self.msg = msg

    def __enter__(self):
        if self.msg:
            print(self.msg, end='... ', file=sys.stderr, flush=True)
        self.start = time.perf_counter_ns()

    def __exit__(self, _type, _value, _traceback):
        elapsed = 1e-6 * (time.perf_counter_ns() - self.start)
        if elapsed < 10000:
            print(f'{int(elapsed)} ms', file=sys.stderr)
        else:
            print(f'{int(elapsed / 1000)} s', file=sys.stderr)


def fair_savgol(strokes) -> None:
    """
    Fair a stroke using a Savitzky-Golay filter.  This function modifies the
    stroke in-place; make a copy if you need your original stroke.

    Args:
        strokes (_s.Stroke or _s.Drawing)
    """
    if isinstance(strokes, _s.Stroke):
        strokes = (strokes,)
    for stroke in strokes:
        n = len(stroke)
        if n > 3:
            window = min(n, 9)
            if window % 2 == 0:
                window -= 1
            stroke.x[:] = savgol_filter(stroke.x, window, 2)
            stroke.y[:] = savgol_filter(stroke.y, window, 2)
            stroke.invalidate_arclengths()
