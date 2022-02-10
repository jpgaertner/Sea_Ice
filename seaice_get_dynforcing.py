import numpy as np

from seaice_size import SeaIceMaskU, SeaIceMaskV
from seaice_params import eps_sq, eps, airTurnAngle, rhoAir, \
    airIceDrag, airIceDrag_south, fCori

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
    sinWin = np.sin(np.deg2rad(airTurnAngle))
    cosWin = np.cos(np.deg2rad(airTurnAngle))

    ##### set up forcing fields #####

    # wind stress is computed on the center of the grid cell and
    # interpolated to u and v points later
    # comute relative wind first
    urel = uWind - 0.5 * (uIce + np.roll(uIce,-1,1))
    vrel = vWind - 0.5 * (vIce + np.roll(vIce,-1,0))
    windSpeed = urel**2 + vrel**2

    tmp = np.where(windSpeed < eps_sq)
    windSpeed = np.sqrt(windSpeed)
    windSpeed[tmp] = eps

    CDAir = rhoAir * airIceDrag * windSpeed
    south = np.where(fCori < 0)
    CDAir[south] = rhoAir * airIceDrag_south * windSpeed[south]

    # compute ice surface stress
    tauX = CDAir * (cosWin * urel - np.sign(fCori) * sinWin * vrel )
    tauY = CDAir * (cosWin * vrel + np.sign(fCori) * sinWin * urel )

    # interpolate to u points
    tauX = 0.5 * ( tauX + np.roll(tauX,1,1) ) * SeaIceMaskU
    # interpolate to v points
    tauY = 0.5 * ( tauY + np.roll(tauY,1,0) ) * SeaIceMaskV

    return tauX, tauY
