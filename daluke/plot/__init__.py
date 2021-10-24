from __future__ import annotations

import matplotlib as mpl
import numpy as np
from pelutils.ds.plot import rc_params_small

def setup_mpl():
    mpl.rcParams.update(rc_params_small)

def running_avg(
    x: np.ndarray,
    y: np.ndarray | None = None, *,
    neighbors            = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """ Calculates the running average assuming even spacing
    If one array of size n is given, it is assumed to run from 0 to n-1 on the x axis
    If two are given, the first are the x axis coordinates
    Returns x and y coordinate arrays of same size """
    if y is None:
        y = x
        x = np.arange(x.size)
    x = x[neighbors:-neighbors]
    kernel = np.arange(1, 2*neighbors+2)
    kernel[-neighbors:] = np.arange(neighbors, 0, -1)
    kernel = kernel / kernel.sum()
    running = np.convolve(y, kernel, mode="valid")
    return x, running

def double_running_avg(
    x: np.ndarray,
    y: np.ndarray | None = None, *,
    inner_neighbors      =   1,
    outer_neighbors      =  12,
    samples              = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """ Running avg. function that produces smoother curves than normal running avg.
    Also handles uneven data spacing better
    This function optionally takes y as `running_avg`
    If both x and y are given, x must be sorted in ascending order
    inner_neighbors: How many neighbors to use for the initial running average
    outer_neighbors: How many neighbors to use for for the second running average
    samples: How many points to sample the running avg. at """
    if y is None:
        y = x
        x = np.arange(x.size)
    x = np.pad(x, pad_width=inner_neighbors)
    y = np.array([*[y[0]]*inner_neighbors, *y, *[y[-1]]*inner_neighbors])
    x, y = running_avg(x, y, neighbors=inner_neighbors)

    # Sampled point along x axis
    extra_sample = outer_neighbors / samples
    # Sample points along x axis
    xx = np.linspace(
        x[0] - extra_sample * (x[-1]-x[0]),
        x[-1] + extra_sample * (x[-1]-x[0]),
        samples + 2 * outer_neighbors,
    )
    # Interpolated points
    yy = np.zeros_like(xx)
    yy[:outer_neighbors] = y[0]
    yy[-outer_neighbors:] = y[-1]

    # Perform interpolation
    x_index = 0
    for k, interp_x in enumerate(xx[outer_neighbors:outer_neighbors+samples], start=outer_neighbors):
        while interp_x >= x[x_index+1]:
            x_index += 1
        a = (y[x_index+1] - y[x_index]) / (x[x_index+1] - x[x_index])
        b = y[x_index] - a * x[x_index]
        yy[k] += (a * interp_x + b)

    return running_avg(xx, yy, neighbors=outer_neighbors)
