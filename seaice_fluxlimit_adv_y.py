import numpy as np

from seaice_size import *
from seaice_params import *
from seaice_flux_limiter import limiter
from seaice_fill_overlap import fill_overlap

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

    vCFL = np.abs(vFld * deltatLoc * recip_dyC * recip_deepFacC)

    Rjp = (tracer[:,3:] - tracer[:,2:-1]) * maskLocS[:,3:]
    Rj = (tracer[:,2:-1] - tracer[:,1:-2]) * maskLocS[:,2:-1]
    Rjm = (tracer[:,1:-2] - tracer[:,:-3]) * maskLocS[:,1:-2]

    Cr = Rjp.copy()
    vFlow = np.where(vTrans[:,2:-1] > 0)
    Cr[vFlow] = Rjm[vFlow]

    tmp = np.where(np.abs(Rj) * CrMax > np.abs(Cr))
    Cr[tmp] = Cr[tmp] / Rj[tmp]
    tmp2 = np.where(np.abs(Rj) * CrMax <= np.abs(Cr))
    Cr[tmp2] = np.sign(Cr[tmp2]) * CrMax * np.sign(Rj[tmp2])

    # limit Cr
    Cr = limiter(Cr)

    vT[:,2:-1] = vTrans[:,2:-1] * (tracer[:,2:-1] + tracer[:,1:-2]) * 0.5 - np.abs(vTrans[:,2:-1]) * ((1 - Cr) + vCFL[:,2:-1] * Cr ) * Rj * 0.5

    vT = fill_overlap(vT)


    return vT