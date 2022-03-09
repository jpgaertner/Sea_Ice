from veros.core.operators import numpy as npx
from veros.core.operators import update, at
from veros import veros_kernel

from seaice_size import *
from seaice_params import *

from seaice_flux_limiter import limiter
from seaice_fill_overlap import fill_overlap


deltatLoc = deltaTdyn #??? is this really needed?

# mask West #??? is this needed later?
maskLocS = SeaIceMaskV * maskInS

# calculates the area integrated zonal flux due to advection using
# second-order interpolation with a flux limiter
@veros_kernel
def calc_MeridionalFlux(state, hIceMean, hSnowMean, Area):

    fields = [hIceMean, hSnowMean, Area]

    state.variables.vTrans = npx.ones_like(hIceMean) * 1000

    # CFL number of meridional flow
    vCFL = npx.abs(state.variables.vIce * deltatLoc * recip_dyC) * 0

    # initialize output array
    MeridionalFlux = npx.zeros((3,nx+2*olx,ny+2*oly))

    # calculate advective fluxes for the fields hIceMean, hSnowMean, Area
    for i in range(3):

        field = fields[i]

        Rjp = (field[3:,:] - field[2:-1,:]) * maskLocS[3:,:]
        Rj = (field[2:-1,:] - field[1:-2,:]) * maskLocS[2:-1,:]
        Rjm = (field[1:-2,:] - field[:-3,:]) * maskLocS[1:-2,:]

        Cr = npx.where(state.variables.vTrans[2:-1,:] > 0, Rjm, Rjp)
        Cr = npx.where(npx.abs(Rj) * CrMax > npx.abs(Cr),
                        Cr / Rj, npx.sign(Cr) * CrMax * npx.sign(Rj))
        Cr = limiter(Cr)

        vF = npx.zeros_like(iceMask)
        vF = update(vF, at[2:-1,:], state.variables.vTrans[2:-1,:] * (
                    field[2:-1,:] + field[1:-2,:]) * 0.5
                    - npx.abs(state.variables.vTrans[2:-1,:]) * ((1 - Cr)
                    + vCFL[2:-1,:] * Cr ) * Rj * 0.5)
        vF = fill_overlap(vF)

        MeridionalFlux = update(MeridionalFlux, at[i], vF)

    return MeridionalFlux