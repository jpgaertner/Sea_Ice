import numpy as np

from seaice_size import *
from seaice_params import *
from seaice_flux_limiter import limiter
from seaice_fill_overlap import fill_overlap

# calculates the area integrated zonal flux due to advection of a tracer
# using second-order interpolation with a flux limiter



### input
# uFld: CFL number of zonal flow
# deltatLoc: local time step
# tracer: field that is advected/ field of interest (e.h. hIceMean)
# uTrans: zonal volume transport
# maskLocW

### output
# uT: zonal advective flux

recip_deepFacC = 1  #in seaice_grid but not set up


def fluxlimit_adv_x(uFld, tracer, uTrans, deltatLoc, maskLocW):

    # output
    uT = np.zeros((sNx+2*OLx,sNy+2*OLy))

    # local variables
    CrMax = 1e6

    uCFL = np.abs(uFld * deltatLoc * recip_dxC * recip_deepFacC)

    Rjp = (tracer[3:,:] - tracer[2:-1,:]) * maskLocW[3:,:]
    Rj = (tracer[2:-1,:] - tracer[1:-2,:]) * maskLocW[2:-1,:]
    Rjm = (tracer[1:-2,:] - tracer[:-3,:]) * maskLocW[1:-2,:]

    Cr = Rjp.copy()
    uFlow = np.where(uTrans[2:-1,:] > 0)
    Cr[uFlow] = Rjm[uFlow]

    tmp = np.where(np.abs(Rj) * CrMax > np.abs(Cr))
    Cr[tmp] = Cr[tmp] / Rj[tmp]
    tmp2 = np.where(np.abs(Rj) * CrMax <= np.abs(Cr))
    Cr[tmp2] = np.sign(Cr[tmp2]) * CrMax * np.sign(Rj[tmp2])

    # limit Cr
    Cr = limiter(Cr)
    
    uT[2:-1,:] = uTrans[2:-1,:] * (tracer[2:-1,:] + tracer[1:-2,:]) * 0.5 - np.abs(uTrans[2:-1,:]) * ((1 - Cr) + uCFL[2:-1,:] * Cr ) * Rj * 0.5

    uT = fill_overlap(uT)


    return uT