import numpy as np

from seaice_size import *
from seaice_params import *
from seaice_flux_limiter import limiter

# calculates the area integrated zonal flux due to advection of a tracer
# using second-order interpolation with a flux limiter



### input
# vFld: CFL number of meridional flow
# deltatLoc: local time step
# tracer: field that is advected/ field of interest (e.h. hIceMean)
# vTrans: meridional volume transport
# maskLocW

### output
# vT: zonal advective flux

recip_deepFacC = 1  #in seaice_grid but not set up

def fluxlimit_adv_y(vFld, tracer, vTrans, deltatLoc, maskLocS):

    # output
    vT = np.zeros((sNx+2*OLx,sNy+2*OLy))

    # local variables
    CrMax = 1e6
    Cr = np.zeros((sNx+2*OLx,sNy+2*OLy))

    vCFL = np.abs(vFld * deltatLoc * recip_dyC * recip_deepFacC)

    Rjp = (tracer[:,3:] - tracer[:,2:-1]) * maskLocS[:,3:]
    Rj = (tracer[:,2:-1] - tracer[:,1:-2]) * maskLocS[:,2:-1]
    Rjm = (tracer[:,1:-2] - tracer[:,:-3]) * maskLocS[:,1:-2]

    Cr[2:-1,:] = Rjp
    vFlow = np.where(vTrans > 0)
    Cr[vFlow] = Rjm

    Cr[2:-1,:] = Cr[2:-1,:]/Rj
    tmp = np.where(np.abs(Rj) * CrMax <= np.abs(Cr))
    Cr[tmp] = np.sign(Cr) * CrMax * np.sign(Rj)

    # limit Cr
    Cr = limiter(Cr)

    vT[:,2:-1] = vTrans[:,2:-1] * (tracer[:,2:-1] + tracer[:,1:-2]) * 0.5 - np.abs(vTrans[:,2:-1]) * ((1 - Cr[:,2:-1]) + vCFL * Cr[:,2:-1] ) * Rj * 0.5

    # vT is only defined in [2:-1,:] (no fill overlap)?

    return vT