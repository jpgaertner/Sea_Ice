import numpy as np

from seaice_size import *
from seaice_params import *

from seaice_fill_overlap import fill_overlap


def seaIceFreeDrift(hIceMean, uVel, vVel, IceSurfStressX0, IceSurfStressY0):


    # initialize fields
    uIceFD = np.zeros((sNx+2*OLx,sNy+2*OLy))
    vIceFD = np.zeros((sNx+2*OLx,sNy+2*OLy))
    uVelCenter = np.zeros((sNx+2*OLx,sNy+2*OLy))
    vVelCenter = np.zeros((sNx+2*OLx,sNy+2*OLy))
    uIceCenter = np.zeros((sNx+2*OLx,sNy+2*OLy))
    vIceCenter = np.zeros((sNx+2*OLx,sNy+2*OLy))
    tauXIceCenter = np.zeros((sNx+2*OLx,sNy+2*OLy))
    tauYIceCenter = np.zeros((sNx+2*OLx,sNy+2*OLy))

    # air-ice stress at cell center
    tauXIceCenter[OLx:-OLx,OLy:-OLy] = 0.5 * (IceSurfStressX0[OLx:-OLx,OLy:-OLy] + IceSurfStressX0[OLx+1:-OLx+1,OLy:-OLy])
    tauYIceCenter[OLx:-OLx,OLy:-OLy] = 0.5 * (IceSurfStressY0[OLx:-OLx,OLy:-OLy] + IceSurfStressY0[OLx:-OLx,OLy+1:-OLy+1])

    # mass of ice per unit area times coriolis factor
    mIceCor = rhoIce * hIceMean[OLx:-OLx,OLy:-OLy] * fCori[OLx:-OLx,OLy:-OLy]

    # ocean surface velocity at the cell center
    uVelCenter[OLx:-OLx,OLy:-OLy] = 0.5 * (uVel[OLx:-OLx,OLy:-OLy] + uVel[OLx+1:-OLx+1,OLy:-OLy])
    vVelCenter[OLx:-OLx,OLy:-OLy] = 0.5 * (vVel[OLx:-OLx,OLy:-OLy] + vVel[OLx:-OLx,OLy+1:-OLy+1])

    # right hand side of the free drift equation
    rhsX = - tauXIceCenter[OLx:-OLx,OLy:-OLy] - mIceCor * vVelCenter[OLx:-OLx,OLy:-OLy]
    rhsY = - tauYIceCenter[OLx:-OLx,OLy:-OLy] + mIceCor * uVelCenter[OLx:-OLx,OLy:-OLy]

    # norm of angle of rhs
    tmp1 = rhsX**2 + rhsY**2
    rhsN = np.zeros((sNx, sNy))
    rhsA = np.zeros((sNx, sNy))
    where1 = np.where(tmp1 > 0)
    rhsN[where1] = np.sqrt(rhsX[where1]**2 + rhsY[where1]**2)
    rhsA[where1] = np.arctan2(rhsY[where1],rhsX[where1])

    # solve for norm
    tmp1 = np.ones((sNx, sNy)) / (rhoConst * waterIceDrag)
    south = np.where(fCori[OLx:-OLx,OLy:-OLy] < 0)
    tmp1[south] = 1 / (rhoConst * waterIceDrag_south)
    tmp2 = tmp1**2 * mIceCor**2
    tmp3 = tmp1**2 * rhsN**2
    tmp4 = tmp2**2 + 4 * tmp3
    solNorm = np.zeros((sNx, sNy))
    where2 = np.where(tmp3 > 0)
    solNorm[where2] = np.sqrt(0.5 * (np.sqrt(tmp4[where2]) - tmp2[where2]))

    # solve for angle
    tmp1 = np.ones((sNx, sNy)) * waterIceDrag * rhoConst
    tmp1[south] = waterIceDrag_south * rhoFresh
    tmp2 = tmp1 * solNorm**2
    tmp3 = mIceCor * solNorm
    tmp4 = tmp2**2 + tmp3**2
    solAngle = np.zeros((sNx, sNy))
    where3 = np.where(tmp3 > 0)
    solAngle[where3] = rhsA[where3] - np.arctan2(tmp3[where3], tmp2[where3])

    # compute uIce, vIce at cell center
    uIceCenter[OLx:-OLx,OLy:-OLy] = uVelCenter[OLx:-OLx,OLy:-OLy] - solNorm * np.cos(solAngle)
    vIceCenter[OLx:-OLx,OLy:-OLy] = vVelCenter[OLx:-OLx,OLy:-OLy] - solNorm * np.sin(solAngle)

    # interpolate to velocity points
    uIceFD[OLx:-OLx,OLy:-OLy] = 0.5 * (uIceCenter[OLx-1:-OLx-1,OLy:-OLy] + uIceCenter[OLx:-OLx,OLy:-OLy])
    vIceFD[OLx:-OLx,OLy:-OLy] = 0.5 * (vIceCenter[OLx:-OLx,OLy-1:-OLy-1] + vIceCenter[OLx:-OLx,OLy:-OLy])

    # fill the overlap regions
    uIceFD = fill_overlap(uIceFD)
    vIceFD = fill_overlap(vIceFD)

    # apply masks
    uIceFD = uIceFD * SeaIceMaskU
    vIceFD = vIceFD * SeaIceMaskV
    
    return uIceFD, vIceFD