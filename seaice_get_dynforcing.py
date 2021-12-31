import numpy as np

from seaice_size import *
from seaice_params import *

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

    # initialize fields
    CDAir = np.zeros((sNx+2*OLx,sNy+2*OLy))
    tauX = np.zeros((sNx+2*OLx,sNy+2*OLy))
    tauY = np.zeros((sNx+2*OLx,sNy+2*OLy))

    ##### set up forcing fields #####

    # wind stress is computed on the center of the grid cell and interpolated to u and v points later
    # compute ice surface stress
    u1 = uWind[:-1,:-1] + 0.5 * (uVel[:-1,:-1] + uVel[1:,:-1]) - 0.5 * (uIce[:-1,:-1] + uIce[1:,:-1])
    v1 = vWind[:-1,:-1] + 0.5 * (vVel[:-1,:-1] + vVel[:-1,1:]) - 0.5 * (vIce[:-1,:-1] + vIce[:-1,1:])
    aaa = u1**2 + v1**2

    tmp = np.where(aaa < eps_sq)
    aaa = np.sqrt(aaa)
    aaa[tmp] = eps

    CDAir[:-1,:-1] = rhoAir * waterIceDrag * aaa
    south = np.where(fCori[:-1,:-1] < 0)
    CDAir[south] = rhoAir * waterIceDrag_south * aaa[south]

    # interpolate to u points
    tauX[1:,1:] = 0.5 * (CDAir[1:,1:] * (cosWin * uWind[1:,1:] - np.sign(fCori[1:,1:]) * sinWin * vWind[1:,1:]) + CDAir[:-1,1:] * (cosWin * uWind[:-1,1:] - np.sign(fCori[:-1,1:]) * sinWin * vWind[:-1,1:])) * SeaIceMaskU[1:,1:]

    # interpolate to v points
    tauY[1:,1:] = 0.5 * (CDAir[1:,1:] * (np.sign(fCori[1:,1:]) * sinWin * uWind[1:,1:] + cosWin * vWind[1:,1:]) + CDAir[1:,:-1] * (np.sign(fCori[1:,:-1]) * sinWin * uWind[1:,:-1] + cosWin * vWind[1:,:-1])) * SeaIceMaskV[1:,1:]

    # why is tauX and tauY calculated again?

    return tauX, tauY