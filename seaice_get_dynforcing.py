from veros.core.operators import numpy as npx

from seaice_size import SeaIceMaskU, SeaIceMaskV, fCori
from seaice_params import eps_sq, eps, airTurnAngle, rhoAir, \
    airIceDrag, airIceDrag_south

# compute surface stresses from atmospheric forcing fields

### input
# uIce: zonal ice velocity [m/s]
# vIce: meridional ice velocity [m/s]
# uWind: zonal wind velocity [m/s]
# vWind: meridional wind velocity [m/s]
# uVel: zonal ocean velocity [m/s]
# vVel: meridional ocean velocity [m/s]

### output
# tauX: zonal wind stress iver ice at u point [N/m2]
# tauY: meridional wind stress over ice at v point [N/m2]


def get_dynforcing(uIce, vIce, uWind, vWind, uVel, vVel):

    # introduce turning angle (default is zero)
    sinWin = npx.sin(npx.deg2rad(airTurnAngle))
    cosWin = npx.cos(npx.deg2rad(airTurnAngle))

    ##### set up forcing fields #####

    # wind stress is computed on the center of the grid cell and
    # interpolated to u and v points later
    # comute relative wind first
    urel = uWind - 0.5 * (uIce + npx.roll(uIce,-1,1))
    vrel = vWind - 0.5 * (vIce + npx.roll(vIce,-1,0))
    windSpeed = urel**2 + vrel**2

    windSpeed = npx.where(windSpeed < eps_sq, eps, npx.sqrt(windSpeed))

    CDAir = npx.where(fCori < 0, airIceDrag_south, airIceDrag) * rhoAir * windSpeed
    
    # compute ice surface stress
    tauX = CDAir * (cosWin * urel - npx.sign(fCori) * sinWin * vrel )
    tauY = CDAir * (cosWin * vrel + npx.sign(fCori) * sinWin * urel )

    # interpolate to u points
    tauX = 0.5 * ( tauX + npx.roll(tauX,1,1) ) * SeaIceMaskU
    # interpolate to v points
    tauY = 0.5 * ( tauY + npx.roll(tauY,1,0) ) * SeaIceMaskV

    return tauX, tauY
