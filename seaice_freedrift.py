import numpy as np

from seaice_size import *
from seaice_params import *


def seaIceFreeDrift(hIceMean, uVel, vVel, IceSurfStressX0, IceSurfStressY0):

    # surface level
    kSurf = 0

    # initialize fields
    uIceFD = np.zeros((sNx+2*OLx,sNy+2*OLy))
    vIceFD = np.zeros((sNx+2*OLx,sNy+2*OLy))
    uIceCenter = np.zeros((sNx+2*OLx,sNy+2*OLy))
    vIceCenter = np.zeros((sNx+2*OLx,sNy+2*OLy))


    # air-ice stress at cell center
    tauXIceCenter = 0.5 * (IceSurfStressX0[:-1,:-1] + IceSurfStressX0[1:,:-1])
    tauYIceCenter = 0.5 * (IceSurfStressY0[:-1,:-1] + IceSurfStressY0[:-1,1:])

    # mass of ice per unit area times coriolis factor
    mIceCor = rhoIce * hIceMean[:-1,:-1] * fCori[:-1,:-1]

    # ocean surface velocity at the cell center
    uVelCenter = 0.5 * (uVel[:-1,:-1,kSurf] + uVel[1:,:-1,kSurf])
    vVelCenter = 0.5 * (vVel[:-1,:-1,kSurf] + vVel[:-1,1:,kSurf])

    # right hand side of the free drift equation
    rhsX = - tauXIceCenter - mIceCor * vVelCenter
    rhsY = - tauYIceCenter + mIceCor * uVelCenter

    # norm of angle of rhs
    tmp1 = rhsX**2 + rhsY**2
    where1 = np.where(tmp1 > 0)
    rhsN = np.zeros((sNx+2*OLx-1, sNy+2*OLy-1))
    rhsA = np.zeros((sNx+2*OLx-1, sNy+2*OLy-1))
    rhsN[where1] = np.sqrt(rhsX[where1]**2 + rhsY[where1]**2)
    rhsA[where1] = np.arctan2(rhsY[where1],rhsX[where1])

    # solve for norm
    tmp1 = np.ones((sNx+2*OLx-1, sNy+2*OLy-1)) / (rhoConst * waterIceDrag)
    south = np.where(fCori < 0)
    tmp1[south] = 1 / (rhoFresh * waterIceDrag_south)
    tmp2 = tmp1**2 * mIceCor**2
    tmp3 = tmp1**2 * rhsN**2
    tmp4 = tmp2**2 + 4 * tmp3
    solNorm = np.zeros((sNx+2*OLx-1, sNy+2*OLy-1))
    where2 = np.where(tmp3 > 0)
    solNorm[where2] = np.sqrt(0.5 * np.sqrt(tmp4[where2]) - tmp2[where2])

    # solve for angle
    tmp1 = np.ones((sNx+2*OLx-1, sNy+2*OLy-1)) * waterIceDrag * rhoConst
    tmp1[south] = waterIceDrag_south * rhoFresh
    tmp2 = tmp1 * solNorm**2
    tmp3 = mIceCor * solNorm
    tmp4 = tmp2**2 + tmp3**2
    solAngle = np.zeros((sNx+2*OLx-1, sNy+2*OLy-1))
    where3 = np.where(tmp3 > 0)
    solAngle[where3] = rhsA[where3] - np.arctan2(tmp3[where3], tmp2[where3])

    # compute uIce, vIce at cell center
    uIceCenter[:-1,:-1] = uVelCenter - solNorm * np.cos(solAngle)
    vIceCenter[:-1,:-1] = vVelCenter - solNorm * np.sin(solAngle)

    uIceFD[OLx:-OLx,OLy:-OLy] = 0.5 * (uIceCenter[OLx-1:-OLx-1,OLy:-OLy] + uIceCenter[OLx:-OLx,OLy:-OLy])
    vIceFD[OLx:-OLx,OLy:-OLy] = 0.5 * (vIceCenter[OLx:-OLx,OLy-1:-OLy-1] + vIceCenter[OLx:-OLx,OLy:-OLy])

    #fill the overlap regions

    #call EXCH_UV_XY_RL

    # apply masks
    uIceFD = uIceFD * SeaIceMaskU
    vIceFD = vIceFD * SeaIceMaskV
    
    return uIceFD, vIceFD