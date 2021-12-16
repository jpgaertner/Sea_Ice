import numpy as np

from seaice_size import *
from seaice_params import *

from seaice_advection import advection


# multidimadvection = true
# seaice_ITD = false
# diffusion = false
# no salinity
# no sitracer (passive tracer, die die dynamik nicht beeinfussen)
# nur cgrid, b grid conversion not needed
# i.e. uc = uIce

### input
# uIce
# vIce
# hIceMean
# hSnowMean
# hIceMeanMask
# Area

### output
# hIceMean
# hSnowMean
# Area


def advdiff(uIce, vIce, hIceMean, hSnowMean, hIceMeanMask, Area):

    # compute cell areas used by all tracers
    xA = dyG * SeaIceMaskU
    yA = dxG * SeaIceMaskV

    # if SEAICEmultiDimAdvection: whole routine toline 570

    recip_hIceMean = np.ones((sNx+2*OLx,sNy+2*OLy))

    # calculate transports of mean thickness through tracer cells
    uTrans = uIce * xA
    vTrans = vIce * yA

    ##### calculate tendency of ice field and do explicit time step #####

    # update mean ice thickness
    gFld = advection(uIce, vIce, uTrans, vTrans, hIceMean, recip_hIceMean)
    hIceMean = (hIceMean + deltatTherm * gFld) * hIceMeanMask # actually just from 1 to N

    # update surface cover fraction
    gFld = advection(uIce, vIce, uTrans, vTrans, Area, recip_hIceMean)
    Area = (Area + deltatTherm * gFld) * hIceMeanMask

    # update mean snow thickness
    gFld = advection(uIce, vIce, uTrans, vTrans, hSnowMean, recip_hIceMean)
    hSnowMean = (hSnowMean + deltatTherm * gFld) * hIceMeanMask

    return hIceMean, hSnowMean, Area
