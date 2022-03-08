from veros.core.operators import numpy as npx
from veros import veros_kernel, KernelOutput, veros_routine

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

# clean up overshoots and other pathological cases after advection
@veros_kernel
def clean_up_advection(state):

    # case 1: negative values
    # calculate overshoots of ice and snow thickness
    os_hIceMean = npx.maximum(-state.variables.hIceMean, 0)
    os_hSnowMean = npx.maximum(-state.variables.hSnowMean, 0)

    # cut off thicknesses and area at zero
    hIceMean = npx.maximum(state.variables.hIceMean, 0)
    hSnowMean = npx.maximum(state.variables.hSnowMean, 0)
    Area = npx.maximum(state.variables.Area, 0)

    # case 2: very thin ice
    # set thicknesses to zero if the ice thickness is very small
    thinIce = (hIceMean <= si_eps)
    hIceMean *= ~thinIce
    hSnowMean *= ~thinIce
    for i in range(nITC):
        TIceSnow = update(state.variables.TIceSnow,
            at[:,:,i], npx.where(thinIce, celsius2K, state.variables.TIceSnow[:,:,i]))

    # case 3: area but no ice and snow
    # set area to zero if no ice or snow is present
    Area = npx.where((hIceMean == 0) & (hSnowMean == 0), 0, Area)

    # case 4: very small area
    # introduce lower boundary for the area (if ice or snow is present)
    Area = npx.where((hIceMean > 0) | (hSnowMean > 0),
                        npx.maximum(Area, area_floor), Area)

    return KernelOutput(hIceMean = hIceMean,
                        hSnowMean = hIceMean,
                        Area = Area,
                        TIceSnow = TIceSnow,
                        os_hIceMean = os_hIceMean,
                        os_hSnowMean = os_hSnowMean)

@veros_routine
def update_clean_up_advection(state):

    # retrieve clean up after advection and update state object
    CleanUpAdvection = clean_up_advection(state)
    state.variables.update(CleanUpAdvection)

# cut off ice cover fraction at 1 after advection
@veros_kernel
def ridging(state):

    Area = npx.minimum(state.variables.Area, 1)

    return KernelOutput(Area = Area)

@veros_routine
def update_ridging(state):

    # retrieve ice cover fraction cutoff and update state object
    Ridging = ridging(state)
    state.variables.update(Ridging)