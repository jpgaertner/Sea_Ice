from veros.core.operators import numpy as npx
from veros.core.operators import update, at

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
    os_hIceMean = npx.maximum(-hIceMean, 0)
    os_hSnowMean = npx.maximum(-hSnowMean, 0)

    # cut off thicknesses and area at zero
    hIceMean = npx.maximum(hIceMean, 0)
    hSnowMean = npx.maximum(hSnowMean, 0)
    Area = npx.maximum(Area, 0)

    # case 2: very thin ice
    # set thicknesses to zero if the ice thickness is very small
    thinIce = (hIceMean <= si_eps)
    hIceMean *= ~thinIce
    hSnowMean *= ~thinIce
    TIceSnow = npx.where(thinIce, celsius2K, TIceSnow)

    # case 3: area but no ice and snow
    # set area to zero if no ice or snow is present
    Area = npx.where((hIceMean == 0) & (hSnowMean == 0), 0, Area)

    # case 4: very small area
    # introduce lower boundary for the area (if ice or snow is present)
    Area = npx.where((hIceMean > 0) | (hSnowMean > 0), npx.maximum(Area, area_floor), Area)


    return hIceMean, hSnowMean, Area, TIceSnow, os_hIceMean, os_hSnowMean


def ridging(Area):

    Area = npx.minimum(Area, 1)

    return Area