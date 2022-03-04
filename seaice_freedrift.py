from veros.core.operators import numpy as npx
from veros import veros_kernel

from seaice_size import *
from seaice_params import *

from seaice_fill_overlap import fill_overlap

# calculate ice velocities from surface stress
@veros_kernel
def freedrift_solver(state):

    # air-ice stress at cell center
    tauXIceCenter = 0.5 * (state.variables.IceSurfStressX0 \
                + npx.roll(state.variables.IceSurfStressX0,-1,1))
    tauYIceCenter = 0.5 * (state.variables.IceSurfStressY0 \
                + npx.roll(state.variables.IceSurfStressY0,-1,0))

    # mass of ice per unit area times coriolis factor
    mIceCor = rhoIce * state.variables.hIceMean * fCori

    # ocean surface velocity at the cell center
    uVelCenter = 0.5 * (state.variables.uVel + npx.roll(state.variables.uVel,-1,1))
    vVelCenter = 0.5 * (state.variables.vVel + npx.roll(state.variables.vVel,-1,0))

    # right hand side of the free drift equation
    rhsX = - tauXIceCenter - mIceCor * vVelCenter
    rhsY = - tauYIceCenter + mIceCor * uVelCenter

    # norm of angle of rhs
    tmp1 = rhsX**2 + rhsY**2
    where1 = (tmp1 > 0)
    rhsN = npx.where(where1, npx.sqrt(rhsX**2 + rhsY**2), 0)
    rhsA = npx.where(where1, npx.arctan2(rhsY,rhsX), 0)

    # solve for norm
    south = (fCori < 0)
    tmp1 = npx.where(south, waterIceDrag_south, waterIceDrag) * rhoConst
    tmp2 = tmp1**2 * mIceCor**2
    tmp3 = tmp1**2 * rhsN**2
    tmp4 = tmp2**2 + 4 * tmp3
    solNorm = npx.where(tmp3 > 0, npx.sqrt(0.5 * (npx.sqrt(tmp4) - tmp2)), 0)

    # solve for angle
    tmp2 = tmp1 * solNorm**2
    tmp3 = mIceCor * solNorm
    tmp4 = tmp2**2 + tmp3**2
    solAngle = npx.where(tmp4 > 0, rhsA - npx.arctan2(tmp3, tmp2), 0)

    # compute uIce, vIce at cell center
    uIceCenter = uVelCenter - solNorm * npx.cos(solAngle)
    vIceCenter = vVelCenter - solNorm * npx.sin(solAngle)

    uIceCenter = fill_overlap(uIceCenter)
    vIceCenter = fill_overlap(vIceCenter)

    # interpolate to velocity points
    uIceFD = 0.5 * (npx.roll(uIceCenter,1,1) + uIceCenter)
    vIceFD = 0.5 * (npx.roll(vIceCenter,1,0) + vIceCenter)

    # fill the overlap regions
    uIceFD = fill_overlap(uIceFD)
    vIceFD = fill_overlap(vIceFD)

    # apply masks
    uIceFD = uIceFD * SeaIceMaskU
    vIceFD = vIceFD * SeaIceMaskV


    return uIceFD, vIceFD