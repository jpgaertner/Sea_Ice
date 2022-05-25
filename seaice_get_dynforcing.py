from veros.core.operators import numpy as npx
from veros import veros_kernel

from seaice_size import SeaIceMaskU, SeaIceMaskV, fCori
from seaice_params import eps_sq, eps, airTurnAngle, rhoAir, \
        airIceDrag, airIceDrag_south


# calculate wind stress over ice from wind and ice velocities
@veros_kernel
def get_dynforcing(state):

    # introduce turning angle (default is zero)
    sinWin = npx.sin(npx.deg2rad(airTurnAngle))
    cosWin = npx.cos(npx.deg2rad(airTurnAngle))

    ##### set up forcing fields #####

    # wind stress is computed on the center of the grid cell and
    # interpolated to u and v points later
    # comute relative wind first
    urel = state.variables.uWind - 0.5 * (
            state.variables.uIce + npx.roll(state.variables.uIce,-1,1) )
    vrel = state.variables.vWind - 0.5 * (
            state.variables.vIce + npx.roll(state.variables.vIce,-1,0) )
    windSpeed = urel**2 + vrel**2

    windSpeed = npx.where(windSpeed < eps_sq, eps, npx.sqrt(windSpeed))

    CDAir = npx.where(fCori < 0, airIceDrag_south, airIceDrag) * rhoAir * windSpeed
    
    # compute ice surface stress
    tauX = CDAir * ( cosWin * urel - npx.sign(fCori) * sinWin * vrel )
    tauY = CDAir * ( cosWin * vrel + npx.sign(fCori) * sinWin * urel )

    # interpolate to u points
    tauX = 0.5 * ( tauX + npx.roll(tauX,1,1) ) * SeaIceMaskU
    # interpolate to v points
    tauY = 0.5 * ( tauY + npx.roll(tauY,1,0) ) * SeaIceMaskV

    return tauX, tauY
