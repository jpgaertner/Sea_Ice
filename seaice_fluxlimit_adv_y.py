from veros.core.operators import numpy as npx
from veros.core.operators import update, at

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


def fluxlimit_adv_y(vFld, tracer, vTrans, deltatLoc, maskLocS):

    CrMax = 1e6

    vCFL = npx.abs(vFld * deltatLoc * recip_dyC)

    Rjp = (tracer[3:,:] - tracer[2:-1,:]) * maskLocS[3:,:]
    Rj = (tracer[2:-1,:] - tracer[1:-2,:]) * maskLocS[2:-1,:]
    Rjm = (tracer[1:-2,:] - tracer[:-3,:]) * maskLocS[1:-2,:]

    Cr = npx.where(vTrans[2:-1,:] > 0, Rjm, Rjp)
    Cr = npx.where(npx.abs(Rj) * CrMax > npx.abs(Cr), Cr / Rj, Cr * CrMax * npx.sign(Rj))
    Cr = limiter(Cr)

    vT = npx.zeros_like(iceMask)
    vT = update(vT, at[2:-1,:], vTrans[2:-1,:] * (tracer[2:-1,:] + tracer[1:-2,:]) \
                        * 0.5 - npx.abs(vTrans[2:-1,:]) * ((1 - Cr) + vCFL[2:-1,:] \
                        * Cr ) * Rj * 0.5)
    vT = fill_overlap(vT)


    return vT