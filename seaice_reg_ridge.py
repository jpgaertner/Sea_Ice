import numpy as np

from seaice_params import *
from seaice_size import *


### input
# hIceMean
# hSnowMean
# Area
# TIceSnow

### output
# hIceMean
# hSnowMean
# Area
# TIceSnow


def clean_up_advection(hIceMean, hSnowMean, Area, TIceSnow):

    # case 1: negative values
    # calculate overshoots of ice and snow thickness
    os_hIceMean = np.maximum(-hIceMean, 0)
    os_hSnowMean = np.maximum(-hSnowMean, 0)

    # cut off thicknesses and area at zero
    hIceMean = np.maximum(hIceMean, 0)
    hSnowMean = np.maximum(hSnowMean, 0)
    Area = np.maximum(Area, 0)

    # case 2: very thin ice
    # set thicknesses to zero if the ice thickness is very small
    thinIce = np.where(hIceMean <= si_eps)
    hIceMean[thinIce] = 0
    hSnowMean[thinIce] = 0
    TIceSnow[thinIce] = celsius2K

    # case 3: area but no ice and snow
    # set area to zero if no ice or snow is present
    noIceSnow = np.where((hIceMean == 0) & (hSnowMean == 0))
    Area[noIceSnow] = 0

    # case 4: very small area
    # introduce lower boundary for the area (if ice or snow is present)
    iceOrSnow = np.where((hIceMean > 0) | (hSnowMean > 0))
    Area[iceOrSnow] = np.maximum(Area[iceOrSnow], area_floor)

    return hIceMean, hSnowMean, Area, TIceSnow, os_hIceMean, os_hSnowMean


def ridging(Area):

    Area = np.minimum(Area, 1)

    return Area