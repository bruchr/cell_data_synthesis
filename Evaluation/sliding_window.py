import itertools
import math

import numpy as np


def generateSlices(im_shape, maxWindowSize, overlapPercent):
    """
    Generates a set of sliding windows for a dataset with the specified dimensions and order.
    """

    # If the input data is smaller than the specified window size,
    # clip the window size to the input size on both dimensions
    windowSize = [min(maxWindowSize[ind], im_shape[ind]) for ind in range(len(im_shape))]

    # Compute the window overlap and step size
    if not isinstance(overlapPercent, (list, tuple)):
        overlapPercent = (overlapPercent, overlapPercent, overlapPercent)

    windowOverlap = [int(math.floor(windowSize[ind] * overlapPercent[ind])) for ind in range(len(im_shape))]
    stepSize = [windowSize[ind] - windowOverlap[ind] for ind in range(len(im_shape))]

    # Determine how many windows we will need in order to cover the input data
    last = [im_shape[ind] - windowSize[ind] for ind in range(len(im_shape))]
    offsets = [list(range(0, last[ind]+1, stepSize[ind])) for ind in range(len(im_shape))]

    # Unless the input data dimensions are exact multiples of the step size,
    # we will need one additional row and column of windows to get 100% coverage
    for ind in range(len(im_shape)):
        if len(offsets[ind]) == 0 or offsets[ind][-1] != last[ind]:
            offsets[ind].append(last[ind])

    # Generate the list of windows
    windows = []
    for offset in itertools.product(*offsets):
        windows.append(tuple(
            [slice(offset[ind], offset[ind]+windowSize[ind]) for ind in range(len(im_shape))]
        ))
    overlap = tuple([windowOverlap for ind in range(len(im_shape))])
    return windows, overlap