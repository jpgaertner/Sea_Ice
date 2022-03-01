from veros.core.operators import numpy as npx
from veros.core.operators import update, at

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


def fluxlimit_adv_x(uFld, tracer, uTrans, deltatLoc, maskLocW):

    CrMax = 1e6

    uCFL = npx.abs(uFld * deltatLoc * recip_dxC)

    Rjp = (tracer[:,3:] - tracer[:,2:-1]) * maskLocW[:,3:]
    Rj = (tracer[:,2:-1] - tracer[:,1:-2]) * maskLocW[:,2:-1]
    Rjm = (tracer[:,1:-2] - tracer[:,:-3]) * maskLocW[:,1:-2]

    Cr = npx.where(uTrans[:,2:-1] > 0, Rjm, Rjp)
    Cr = npx.where(npx.abs(Rj) * CrMax > npx.abs(Cr), Cr / Rj, npx.sign(Cr) * CrMax * npx.sign(Rj))
    Cr = limiter(Cr)
    
    uT = npx.zeros_like(iceMask)
    uT = update(uT, at[:,2:-1], uTrans[:,2:-1] * (tracer[:,2:-1] + tracer[:,1:-2]) \
                        * 0.5 - npx.abs(uTrans[:,2:-1]) * ((1 - Cr) + uCFL[:,2:-1] \
                        * Cr ) * Rj * 0.5)
    uT = fill_overlap(uT)
    

    return uT