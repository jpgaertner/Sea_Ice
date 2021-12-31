
import numpy as np

from seaice_params import *
from seaice_size import *

# no relaxations (RLX)
# disable_area_floor = false

# SEAICE_VARIABLE_SALINITY = false?


###### cleaning up after advection and driver for ice ridging #####

### input
# hIceMean
# hSnowMean
# Area
# TIceSnow

### ouput
# hIceMean
# hSnowMean
# Area
# TIceSnow

def ridging(hIceMean, hSnowMean, Area, TIceSnow):


    ##### pathological cases #####

    # case 1: negative values
    hIceMean.clip(0)
    hSnowMean.clip(0)
    Area.clip(0)

    # case 2: very thin ice
    thinIce = np.where(hIceMean <= si_eps)
    hIceMean[thinIce] = 0
    hSnowMean[thinIce] = 0
    TIceSnow[thinIce] = celsius2K

    # case 3: area but no ice and snow
    noIceSnow = np.where((hIceMean == 0) & (hSnowMean == 0))
    Area[noIceSnow] = 0
    # so the case area with no ice but snow is possible?

    # case 4: very small area
    iceOrSnow = np.where((hIceMean > 0) | (hSnowMean > 0))
    Area[iceOrSnow] = np.maximum(Area[iceOrSnow], area_floor)
    # same question?

    ##### ridging #####

    Area = np.minimum(Area, 1)

    return hIceMean, hSnowMean, Area, TIceSnow