import numpy as np

from seaice_size import *
from seaice_params import *

from seaice_fill_overlap import fill_overlap


def seaIceFreeDrift(hIceMean, uVel, vVel, IceSurfStressX0, IceSurfStressY0):

    # initialize fields
    uIceFD = np.zeros((sNy+2*OLy,sNx+2*OLx))
    vIceFD = np.zeros((sNy+2*OLy,sNx+2*OLx))
    uIceCenter = np.zeros((sNy+2*OLy,sNx+2*OLx))
    vIceCenter = np.zeros((sNy+2*OLy,sNx+2*OLx))
    uVelCenter = np.zeros((sNy+2*OLy,sNx+2*OLx))
    vVelCenter = np.zeros((sNy+2*OLy,sNx+2*OLx))
    tauXIceCenter = np.zeros((sNy+2*OLy,sNx+2*OLx))
    tauYIceCenter = np.zeros((sNy+2*OLy,sNx+2*OLx))

    # air-ice stress at cell center
    tauXIceCenter[OLy:-OLy,OLx:-OLx] = 0.5 * (IceSurfStressX0[OLy:-OLy,OLx:-OLx] + IceSurfStressX0[OLy+1:-OLy+1,OLx:-OLx])
    tauYIceCenter[OLy:-OLy,OLx:-OLx] = 0.5 * (IceSurfStressY0[OLy:-OLy,OLx:-OLx] + IceSurfStressY0[OLy:-OLy,OLx+1:-OLx+1])

    # mass of ice per unit area times coriolis factor
    mIceCor = rhoIce * hIceMean[OLy:-OLy,OLx:-OLx] * fCori[OLy:-OLy,OLx:-OLx]

    # ocean surface velocity at the cell center
    uVelCenter[OLy:-OLy,OLx:-OLx] = 0.5 * (uVel[OLy:-OLy,OLx:-OLx] + uVel[OLy+1:-OLy+1,OLx:-OLx])
    vVelCenter[OLy:-OLy,OLx:-OLx] = 0.5 * (vVel[OLy:-OLy,OLx:-OLx] + vVel[OLy:-OLy,OLx+1:-OLx+1])

    # right hand side of the free drift equation
    rhsX = - tauXIceCenter[OLy:-OLy,OLx:-OLx] - mIceCor * vVelCenter[OLy:-OLy,OLx:-OLx]
    rhsY = - tauYIceCenter[OLy:-OLy,OLx:-OLx] + mIceCor * uVelCenter[OLy:-OLy,OLx:-OLx]

    # norm of angle of rhs
    tmp1 = rhsX**2 + rhsY**2
    rhsN = np.zeros((sNy,sNx))
    rhsA = np.zeros((sNy,sNx))
    where1 = np.where(tmp1 > 0)
    rhsN[where1] = np.sqrt(rhsX[where1]**2 + rhsY[where1]**2)
    rhsA[where1] = np.arctan2(rhsY[where1],rhsX[where1])

    # solve for norm
    tmp1 = np.ones((sNy,sNx)) / (rhoConst * waterIceDrag)
    south = np.where(fCori[OLy:-OLy,OLx:-OLx] < 0)
    tmp1[south] = 1 / (rhoConst * waterIceDrag_south)
    tmp2 = tmp1**2 * mIceCor**2
    tmp3 = tmp1**2 * rhsN**2
    tmp4 = tmp2**2 + 4 * tmp3
    solNorm = np.zeros((sNy,sNx))
    where2 = np.where(tmp3 > 0)
    solNorm[where2] = np.sqrt(0.5 * (np.sqrt(tmp4[where2]) - tmp2[where2]))

    # solve for angle
    tmp1 = np.ones((sNy,sNx)) * waterIceDrag * rhoConst
    tmp1[south] = waterIceDrag_south * rhoConst
    tmp2 = tmp1 * solNorm**2
    tmp3 = mIceCor * solNorm
    tmp4 = tmp2**2 + tmp3**2
    solAngle = np.zeros((sNy,sNx))
    where3 = np.where(tmp4 > 0)
    solAngle[where3] = rhsA[where3] - np.arctan2(tmp3[where3], tmp2[where3])

    # compute uIce, vIce at cell center
    uIceCenter[OLy:-OLy,OLx:-OLx] = uVelCenter[OLy:-OLy,OLx:-OLx] - solNorm * np.cos(solAngle)
    vIceCenter[OLy:-OLy,OLx:-OLx] = vVelCenter[OLy:-OLy,OLx:-OLx] - solNorm * np.sin(solAngle)

    uIceCenter = fill_overlap(uIceCenter)
    vIceCenter = fill_overlap(vIceCenter)

    # interpolate to velocity points
    uIceFD[OLy:-OLy,OLx:-OLx] = 0.5 * (uIceCenter[OLy-1:-OLy-1,OLx:-OLx] + uIceCenter[OLy:-OLy,OLx:-OLx])
    vIceFD[OLy:-OLy,OLx:-OLx] = 0.5 * (vIceCenter[OLy:-OLy,OLx-1:-OLx-1] + vIceCenter[OLy:-OLy,OLx:-OLx])

    # fill the overlap regions
    uIceFD = fill_overlap(uIceFD)
    vIceFD = fill_overlap(vIceFD)

    # apply masks
    uIceFD = uIceFD * SeaIceMaskU
    vIceFD = vIceFD * SeaIceMaskV

    # import matplotlib.pyplot as plt
    # plt.contourf(uIceFD[OLy:-OLy,OLx:-OLx])
    # plt.colorbar()
    # plt.show()


    return uIceFD, vIceFD