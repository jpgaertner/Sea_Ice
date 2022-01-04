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
# hIceMeanMask: contains geometry of the set up
# hSnowMean: mean snow thickness [m3/m2]
# Area: sea ice cover fraction (0 <= Area <= 1)
# uIce: zonal ice velocity [m/s]
# vIce: meridional ice velocity [m/s]

### output
# hIceMean
# hSnowMean
# Area


recip_hIceMean = np.ones((sNx+2*OLx,sNy+2*OLy))

def advdiff(uIce, vIce, hIceMean, hSnowMean, hIceMeanMask, Area):

    # compute cell areas used by all tracers
    xA = dyG * SeaIceMaskU
    yA = dxG * SeaIceMaskV

    # calculate transports of mean thickness through tracer cells
    uTrans = uIce * xA
    vTrans = vIce * yA

    ##### calculate tendency of ice field and do explicit time step #####

    extensiveFld = True #indicates to advect an "extensive" type of ice field

    print('before ice', hIceMean)
    # update mean ice thickness
    gFld = advection(uIce, vIce, uTrans, vTrans, hIceMean, recip_hIceMean, extensiveFld)
    hIceMean = (hIceMean + deltatTherm * gFld) * hIceMeanMask # actually just from 1 to N
    print('after ice', hIceMean)


    #print('before area adv', Area)
    # update surface cover fraction
    gFld = advection(uIce, vIce, uTrans, vTrans, Area, recip_hIceMean, extensiveFld)
    Area = (Area + deltatTherm * gFld) * hIceMeanMask
    #print('after area adv', Area)

    # update mean snow thickness
    gFld = advection(uIce, vIce, uTrans, vTrans, hSnowMean, recip_hIceMean, extensiveFld)
    hSnowMean = (hSnowMean + deltatTherm * gFld) * hIceMeanMask

    return hIceMean, hSnowMean, Area
