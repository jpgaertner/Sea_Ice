import numpy as np

from seaice_size import *
from seaice_params import *

from seaice_fill_overlap import fill_overlap


def freedrift_solver(hIceMean, uVel, vVel, IceSurfStressX0, IceSurfStressY0):

    # initialize fields
    uIceFD = np.zeros_like(iceMask)
    vIceFD = np.zeros_like(iceMask)
    uIceCenter = np.zeros_like(iceMask)
    vIceCenter = np.zeros_like(iceMask)
    uVelCenter = np.zeros_like(iceMask)
    vVelCenter = np.zeros_like(iceMask)

    # air-ice stress at cell center
    tauXIceCenter = 0.5 * (IceSurfStressX0 + np.roll(IceSurfStressX0,-1,1))
    tauYIceCenter = 0.5 * (IceSurfStressY0 + np.roll(IceSurfStressY0,-1,0))

    # mass of ice per unit area times coriolis factor
    mIceCor = rhoIce * hIceMean * fCori

    # ocean surface velocity at the cell center
    uVelCenter = 0.5 * (uVel + np.roll(uVel,-1,1))
    vVelCenter = 0.5 * (vVel + np.roll(vVel,-1,0))

    # right hand side of the free drift equation
    rhsX = - tauXIceCenter - mIceCor * vVelCenter
    rhsY = - tauYIceCenter + mIceCor * uVelCenter

    # norm of angle of rhs
    tmp1 = rhsX**2 + rhsY**2
    rhsN = np.zeros_like(iceMask)
    rhsA = np.zeros_like(iceMask)
    where1 = np.where(tmp1 > 0)
    rhsN = rhsN.at[where1].set(np.sqrt(rhsX[where1]**2 + rhsY[where1]**2))
    rhsA = rhsA.at[where1].set(np.arctan2(rhsY[where1],rhsX[where1]))

    # solve for norm
    tmp1 = np.ones_like(iceMask) / (rhoConst * waterIceDrag)
    south = np.where(fCori < 0)
    tmp1 = tmp1.at[south].set(1 / (rhoConst * waterIceDrag_south))
    tmp2 = tmp1**2 * mIceCor**2
    tmp3 = tmp1**2 * rhsN**2
    tmp4 = tmp2**2 + 4 * tmp3
    solNorm = np.zeros_like(iceMask)
    where2 = np.where(tmp3 > 0)
    solNorm = solNorm.at[where2].set(np.sqrt(0.5 * (np.sqrt(tmp4[where2]) - tmp2[where2])))

    # solve for angle
    tmp1 = np.ones_like(iceMask) * waterIceDrag * rhoConst
    tmp1 = tmp1.at[south].set(waterIceDrag_south * rhoConst)
    tmp2 = tmp1 * solNorm**2
    tmp3 = mIceCor * solNorm
    tmp4 = tmp2**2 + tmp3**2
    solAngle = np.zeros_like(iceMask)
    where3 = np.where(tmp4 > 0)
    solAngle = solAngle.at[where3].set(rhsA[where3] - np.arctan2(tmp3[where3], tmp2[where3]))

    # compute uIce, vIce at cell center
    uIceCenter = uVelCenter - solNorm * np.cos(solAngle)
    vIceCenter = vVelCenter - solNorm * np.sin(solAngle)

    uIceCenter = fill_overlap(uIceCenter)
    vIceCenter = fill_overlap(vIceCenter)

    # interpolate to velocity points
    uIceFD = 0.5 * (np.roll(uIceCenter,1,1) + uIceCenter)
    vIceFD = 0.5 * (np.roll(vIceCenter,1,0) + vIceCenter)

    # fill the overlap regions
    uIceFD = fill_overlap(uIceFD)
    vIceFD = fill_overlap(vIceFD)

    # apply masks
    uIceFD = uIceFD * SeaIceMaskU
    vIceFD = vIceFD * SeaIceMaskV


    return uIceFD, vIceFD