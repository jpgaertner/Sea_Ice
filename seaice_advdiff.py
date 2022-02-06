import numpy as np

from seaice_size import *
from seaice_params import *

from seaice_advection import advection


# multidimadvection = true
# seaice_ITD = false
# diffusion = false
# no salinity
# no sitracer (passive tracer, die die dynamik nicht beeinfussen)
# nur cgrid, b grid conversion not needed, i.e. uc = uIce

### input
# hIceMean: mean ice thickness [m3/m2] (= hIceActual * Area with Area the sea ice cover fraction and hIceActual = Vol_ice / x_len y_len)
# hSnowMean: mean snow thickness [m3/m2]
# Area: sea ice cover fraction (0 <= Area <= 1)
# uIce: zonal ice velocity [m/s]
# vIce: meridional ice velocity [m/s]

### output
# hIceMean
# hSnowMean
# Area

recip_hIceMean = np.ones((sNy+2*OLy,sNx+2*OLx))


def advdiff(uIce, vIce, hIceMean, hSnowMean, Area):

    # compute cell areas used by all tracers
    xA = dyG * SeaIceMaskU
    yA = dxG * SeaIceMaskV

    # calculate transports of mean thickness through tracer cells
    uTrans = uIce * xA
    vTrans = vIce * yA


    ##### calculate tendency of ice field and do explicit time step #####

    extensiveFld = True #indicates to advect an "extensive" type of ice field

    # update mean ice thickness
    gFld = advection(uIce, vIce, uTrans, vTrans, hIceMean, recip_hIceMean, extensiveFld)
    hIceMean = (hIceMean + deltaTtherm * gFld) * iceMask

    # update surface cover fraction
    gFld = advection(uIce, vIce, uTrans, vTrans, Area, recip_hIceMean, extensiveFld)
    Area = (Area + deltaTtherm * gFld) * iceMask

    # update mean snow thickness
    gFld = advection(uIce, vIce, uTrans, vTrans, hSnowMean, recip_hIceMean, extensiveFld)
    hSnowMean = (hSnowMean + deltaTtherm * gFld) * iceMask


    return hIceMean, hSnowMean, Area
