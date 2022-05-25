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
def calc_MeridionalFlux(state, field):

    maskLocS = SeaIceMaskV * maskInS

    # CFL number of meridional flow
    vCFL = npx.abs(state.variables.vIce * state.settings.deltatDyn * recip_dyC)

    # calculate slope ratio Cr
    Rjp = (field[3:,:] - field[2:-1,:]) * maskLocS[3:,:]
    Rj = (field[2:-1,:] - field[1:-2,:]) * maskLocS[2:-1,:]
    Rjm = (field[1:-2,:] - field[:-3,:]) * maskLocS[1:-2,:]

    Cr = npx.where(state.variables.vTrans[2:-1,:] > 0, Rjm, Rjp)
    Cr = npx.where(npx.abs(Rj) * CrMax > npx.abs(Cr),
                    Cr / Rj, npx.sign(Cr) * CrMax * npx.sign(Rj))
    Cr = limiter(Cr)

    # calculate meridional advective fluxes for the given field
    MeridionalFlux = npx.zeros_like(iceMask)
    MeridionalFlux = update(MeridionalFlux, at[2:-1,:], state.variables.vTrans[2:-1,:] * (
                field[2:-1,:] + field[1:-2,:]) * 0.5
                - npx.abs(state.variables.vTrans[2:-1,:]) * ((1 - Cr)
                + vCFL[2:-1,:] * Cr ) * Rj * 0.5)
    MeridionalFlux = fill_overlap(MeridionalFlux)


    return MeridionalFlux