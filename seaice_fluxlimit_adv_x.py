from veros.core.operators import numpy as npx
from veros.core.operators import update, at
from veros import veros_kernel

from seaice_size import *
from seaice_params import *

from seaice_flux_limiter import limiter
from seaice_fill_overlap import fill_overlap


# calculates the area integrated zonal flux due to advection using
# second-order interpolation with a flux limiter
@veros_kernel
def calc_ZonalFlux(state, field):

    maskLocW = SeaIceMaskU * maskInW

    # CFL number of zonal flow
    uCFL = npx.abs(state.variables.uIce * state.variables.deltatDyn * recip_dxC)

    # calculate slope ratio Cr
    Rjp = (field[:,3:] - field[:,2:-1]) * maskLocW[:,3:]
    Rj = (field[:,2:-1] - field[:,1:-2]) * maskLocW[:,2:-1]
    Rjm = (field[:,1:-2] - field[:,:-3]) * maskLocW[:,1:-2]

    Cr = npx.where(state.variables.uTrans[:,2:-1] > 0, Rjm, Rjp)
    Cr = npx.where(npx.abs(Rj) * CrMax > npx.abs(Cr),
                        Cr / Rj, npx.sign(Cr) * CrMax * npx.sign(Rj))
    Cr = limiter(Cr)

    # zonal advective fluxes for the given field
    ZonalFlux = npx.zeros_like(iceMask)
    ZonalFlux = update(ZonalFlux, at[:,2:-1], state.variables.uTrans[:,2:-1] * (
                field[:,2:-1] + field[:,1:-2]) * 0.5
                - npx.abs(state.variables.uTrans[:,2:-1]) * ((1 - Cr)
                + uCFL[:,2:-1] * Cr ) * Rj * 0.5)
    ZonalFlux = fill_overlap(ZonalFlux)


    return ZonalFlux