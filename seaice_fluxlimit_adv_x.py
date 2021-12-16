import numpy as np

from seaice_size import *
from seaice_params import *
from seaice_flux_limiter import limiter

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
    Cr = np.zeros((sNx+2*OLx,sNy+2*OLy))

    uCFL = np.abs(uFld * deltatLoc * recip_dxC * recip_deepFacC)

    Rjp = (tracer[3:,:] - tracer[2:-1,:]) * maskLocW[3:,:]
    Rj = (tracer[2:-1,:] - tracer[1:-2,:]) * maskLocW[2:-1,:]
    Rjm = (tracer[1:-2,:] - tracer[:-3,:]) * maskLocW[1:-2,:]

    Cr[2:-1,:] = Rjp
    uFlow = np.where(uTrans > 0)
    Cr[uFlow] = Rjm

    Cr[2:-1,:] = Cr[2:-1,:]/Rj
    tmp = np.where(np.abs(Rj) * CrMax <= np.abs(Cr))
    Cr[tmp] = np.sign(Cr) * CrMax * np.sign(Rj)

    # limit Cr
    Cr = limiter(Cr)

    uT[2:-1,:] = uTrans[2:-1,:] * (tracer[2:-1,:] + tracer[1:-2,:]) * 0.5 - np.abs(uTrans[2:-1,:]) * ((1 - Cr[2:-1,:]) + uCFL * Cr[2:-1,:] ) * Rj * 0.5

    # uT is only defined in [2:-1,:] (no fill overlap)?

    return uT