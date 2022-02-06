import numpy as np

from seaice_params import *
from seaice_size import *

from seaice_strainrates import strainrates
from seaice_ocean_drag_coeffs import ocean_drag_coeffs
from seaice_bottomdrag_coeffs import bottomdrag_coeffs
from seaice_global_sum import global_sum
from seaice_averaging import c_point_to_z_point


### input
# uIce: zonal ice velocity
# vIce: meridional ice velocity
# uVel: zonal ocean surface velocity 
# vVel: meridional ocean surface velocity
# hIceMean: mean ice thickness
# Area: ice cover fraction
# press0: ocean surface pressure
# secondOrderBC: flag
# IceSurfStressX0: zonal stress on ice surface at c point
# IceSurfStressY0: meridional stress on ice surface at c point
# R_low: water depth

### output:
# uIce: zonal ice velocity
# vIce: meridional ice velocity


import matplotlib.pyplot as plt


def evp(uIce, vIce, uVel, vVel, hIceMean, Area, press0, secondOrderBC,
    IceSurfStressX0, IceSurfStressY0, SeaIceMassC, SeaIceMassU,
    SeaIceMassV, R_low):

    ##### initializations #####

    useAdaptiveEVP = False
    # change to input with default value = False?
    if useAdaptiveEVP:
        EVPcFac = deltaTdyn 
    else:
        EVPcFac = 0
    # ... aEVPCoeff

    sinWat = np.sin(np.deg2rad(waterTurnAngle))
    cosWat = np.cos(np.deg2rad(waterTurnAngle))

    PlasDefCoeffSq = PlasDefCoeff**2
    recip_PlasDefCoeffSq = 1 / PlasDefCoeffSq

    evpAlpha = 300
    recip_evpAlpha = 1 / evpAlpha
    evpStarFac = 1
    evpRevFac = 1
    recip_evpRevFac = recip_PlasDefCoeffSq

    denom1 = np.ones((sNy+2*OLy,sNx+2*OLx)) / evpAlpha
    denom2 = denom1.copy()

    evpBeta = evpAlpha 
    betaFac = evpBeta * recip_deltaTdyn
    betaFacU = betaFac
    betaFacV = betaFac
    betaFacP1 = betaFac + evpStarFac * recip_deltaTdyn
    betaFacP1U = betaFacP1
    betaFacP1V = betaFacP1

    # copy previous time step (n-1) of uIce, vIce
    uIceNm1 = uIce.copy()
    vIceNm1 = vIce.copy()

    # initialize adaptive EVP specific fields
    evpAlphaC = np.ones((sNy+2*OLy,sNx+2*OLx)) * evpAlpha
    evpAlphaZ = np.ones((sNy+2*OLy,sNx+2*OLx)) * evpAlpha
    evpBetaU = np.ones((sNy+2*OLy,sNx+2*OLx)) * evpBeta
    evpBetaV = np.ones((sNy+2*OLy,sNx+2*OLx)) * evpBeta

    # initialize fractional areas at velocity points
    areaW = np.zeros((sNy+2*OLy,sNx+2*OLx))
    areaS = np.zeros((sNy+2*OLy,sNx+2*OLx))
    areaW[:,1:] = 0.5 * (Area[:,1:] + Area[:,:-1])
    areaS[1:,:] = 0.5 * (Area[1:,:] + Area[:-1,:])


    ##### main loop #####

    # initializations
    e12Csq = np.zeros((sNy+2*OLy,sNx+2*OLx))
    stressDivX = np.zeros((sNy+2*OLy,sNx+2*OLx))
    stressDivY = np.zeros((sNy+2*OLy,sNx+2*OLx))
    pressC = np.zeros((sNy+2*OLy,sNx+2*OLx))
    sigma1 = np.zeros((sNy+2*OLy,sNx+2*OLx))
    sigma2 = np.zeros((sNy+2*OLy,sNx+2*OLx))
    sigma12 = np.zeros((sNy+2*OLy,sNx+2*OLx))
    denomU = np.zeros((sNy+2*OLy,sNx+2*OLx))
    denomV = np.zeros((sNy+2*OLy,sNx+2*OLx))
    resSig = np.array([None]*nEVPsteps)
    resU = np.array([None]*nEVPsteps)

    for i in range(nEVPsteps):
        # calculate strain rates and bulk moduli/ viscosities
        e11, e22, e12 = strainrates(uIce, vIce, secondOrderBC)

        # save previous (p-1) iteration
        sig1Pm1 = sigma1.copy()
        sig2Pm1 = sigma2.copy()
        sig12Pm1 = sigma12.copy()
        uIcePm1 = uIce.copy()
        vIcePm1 = vIce.copy()

        ep = e11 + e22
        em = e11 - e22

        # use area weighted average of squares of e12 (more accurate)
        e12Csq[1:-1,1:-1] = 0.25 * recip_rA[1:-1,1:-1] * (
            rAz[1:-1,1:-1] * e12[1:-1,1:-1]**2 + rAz[1:-1,2:]
            * e12[1:-1,2:]**2 + rAz[2:,1:-1] * e12[2:,1:-1]**2
            + rAz[2:,2:] * e12[2:,2:]**2)
        deltaSq = ep**2 + recip_PlasDefCoeffSq * em**2 \
            + recip_PlasDefCoeffSq * 4 * e12Csq
        deltaC = np.sqrt(deltaSq)

        # smooth regularization of delta for better differentiability
        deltaCreg = deltaC + deltaMin

        zetaC = 0.5 * (press0 * (1 + tensileStrFac)) / deltaCreg

        if useAdaptiveEVP:
            evpAlphaC = np.sqrt(zetaC * EVPcFac / np.maximum(
                SeaIceMassC, 1e-4) * recip_rA) * iceMask
            evpAlphaC = np.maximum(evpAlphaC, aEVPalphaMin)

        # calculate zeta, delta on z points
        zetaZ = c_point_to_z_point(zetaC)
        deltaZ = c_point_to_z_point(deltaC)

        # recalculate pressure
        pressC = (press0 * (1 - pressReplFac) + 2 * zetaC * deltaC
            * pressReplFac / (1 + tensileStrFac)) * (1 - tensileStrFac)

        # divergence strain rates at c points times p / divided by delta minus 1
        div = (2 * zetaC * ep - pressC) * iceMask
        # tension strain rates at c points times p / divided by delta
        tension = 2 * zetaC * em * iceMask
        # shear strain rates at z points times p / divided by delta
        shear = 2 * zetaZ * e12


        ##### first step stress equations #####

        if useAdaptiveEVP:
            denom1 = 1 / evpAlphaC
            denom2 = denom1.copy()
            
        # calculate sigma1, sigma2 on c points
        sigma1 = (sigma1 * (evpAlphaC - evpRevFac) + div) \
            * denom1 * iceMask
        sigma2 = (sigma2 * (evpAlphaC - evpRevFac) + tension \
            * recip_evpRevFac) * denom2 * iceMask

        # recover sigma11 and sigma22
        sig11 = 0.5 * (sigma1 + sigma2)
        sig22 = 0.5 * (sigma1 - sigma2)

        # calculate sigma12 on z points
        if useAdaptiveEVP:
            evpAlphaZ[1:-1,1:-1] = 0.25 * (evpAlphaC[1:-1,1:-1]
                + evpAlphaC[:-2,:-2] + evpAlphaC[1:-1,:-2]
                + evpAlphaC[:-2,1:-1])
            denom2 = 1 / evpAlphaZ
        sigma12 = (sigma12 * (evpAlphaZ - evpRevFac) + shear
            * recip_evpRevFac) * denom2

        # calculate divergence of stress tensor
        stressDivX[1:-1,1:-1] = (sig11[1:-1,1:-1] * dyF[1:-1,1:-1]
            - sig11[1:-1,:-2] * dyF[1:-1,:-2] + sigma12[2:,1:-1]
            * dxV[2:,1:-1] - sigma12[1:-1,1:-1] * dxV[1:-1,1:-1]) \
            * recip_rAw[1:-1,1:-1]
        stressDivY[1:-1,1:-1] = (sig22[1:-1,1:-1] * dxF[1:-1,1:-1]
            - sig22[:-2,1:-1] * dxF[:-2,1:-1] + sigma12[1:-1,2:]
            * dyU[1:-1,2:]- sigma12[1:-1,1:-1] * dyU[1:-1,1:-1]) \
            * recip_rAs[1:-1,1:-1]

        sig1Pm1 = (sigma1 - sig1Pm1) * evpAlphaC
        sig2Pm1 = (sigma2 - sig2Pm1) * evpAlphaC
        sig12Pm1 = (sigma12 - sig12Pm1) * evpAlphaZ

        resSig[i] = (sig1Pm1**2 + sig2Pm1**2 + sig12Pm1**2).sum()
        resSig = global_sum(resSig)

        # set up right hand side for stepping the velocity field
        cDrag = ocean_drag_coeffs(uIce, vIce, uVel, vVel)
        cBotC = bottomdrag_coeffs(uIce, vIce, hIceMean, Area, R_low)

        # over open ocean...
        locMaskU = SeaIceMassU.copy()
        locMaskV = SeaIceMassV.copy()
        maskU = np.where(locMaskU != 0)
        locMaskU[maskU] = 1
        maskV = np.where(locMaskV != 0)
        locMaskV[maskV] = 1

        # set up anti symmetric drag force and add in ice ocean stress (average to correct velocity points)
        IceSurfStressX = np.zeros((sNy+2*OLy,sNx+2*OLx))
        IceSurfStressY = np.zeros((sNy+2*OLy,sNx+2*OLx))
        IceSurfStressX[:-1,1:] = IceSurfStressX0[:-1,1:] + (0.5 * (
            cDrag[:-1,1:] + cDrag[:-1,:-1]) * cosWat * uVel[:-1,1:]
            - np.sign(fCori[:-1,1:]) * sinWat * 0.5 * (
            cDrag[:-1,1:] * 0.5 * (vVel[:-1,1:] - vIce[:-1,1:]
            + vVel[1:,1:] - vIce[1:,1:])
            + cDrag[:-1,:-1] * 0.5 * (vVel[:-1,:-1] - vIce[:-1,:-1]
            + vVel[1:,:-1] - vIce[1:,:-1]))
            * locMaskU[:-1,1:]) * areaW[:-1,1:]
        IceSurfStressY[1:,:-1] = IceSurfStressY0[1:,:-1] + (0.5 * (
            cDrag[1:,:-1] + cDrag[:-1,:-1]) * cosWat * vVel[1:,:-1]
            + np.sign(fCori[1:,:-1]) * sinWat * 0.5 * (
                cDrag[1:,:-1] * 0.5 * (uVel[1:,:-1] - uIce[1:,:-1]
                + uVel[1:,1:] - uIce[1:,1:])
                + cDrag[:-1,:-1] * 0.5 * (uVel[:-1,:-1] - uIce[:-1,:-1]
                + uVel[:-1,1:] - uIce[:-1,1:]))
                * locMaskV[1:,:-1]) * areaS[1:,:-1]

        # add coriolis terms
        IceSurfStressX[:-1,1:] = IceSurfStressX[:-1,1:] + 0.5 * (
            SeaIceMassC[:-1,1:] * fCori[:-1,1:] * 0.5 * (
            vIce[:-1,1:] + vIce[1:,1:])
            + SeaIceMassC[:-1,:-1] * fCori[:-1,:-1] * 0.5 * (
            vIce[:-1,:-1] + vIce[1:,:-1]))
        IceSurfStressY[1:,:-1] = IceSurfStressY[1:,:-1] - 0.5 * (
            SeaIceMassC[1:,:-1] * fCori[1:,:-1] * 0.5 * (
            uIce[1:,:-1] + uIce[1:,1:])
            + SeaIceMassC[:-1,:-1] * fCori[:-1,:-1] * 0.5 * (
            uIce[:-1,:-1] + uIce[:-1,1:]))

        # step momentum equations with ice-ocean stress treated implicitly
        if useAdaptiveEVP:
            evpBetaU[:,1:] = 0.5 * (evpAlphaC[:,:-1] + evpAlphaC[:,1:])
            evpBetaV[1:,:] = 0.5 * (evpAlphaC[:-1,:] + evpAlphaC[1:,:])

        betaFacU = evpBetaU * recip_deltaTdyn
        betaFacV = evpBetaV * recip_deltaTdyn
        tmp = evpStarFac * recip_deltaTdyn
        betaFacP1V = betaFacV + tmp
        betaFacP1U = betaFacU + tmp
        denomU[:,1:] = (SeaIceMassU[:,1:] * betaFacP1U[:,1:] + 0.5 * (
            cDrag[:,1:] + cDrag[:,:-1]) * cosWat * areaW[:,1:]) \
            + areaW[:,1:] * 0.5 * (cBotC[:,1:] + cBotC[:,:-1])
        denomV[1:,:] = (SeaIceMassV[1:,:] * betaFacP1V[1:,:] + 0.5 * (
            cDrag[1:,:] + cDrag[:-1,:]) * cosWat * areaS[1:,:]) \
            + areaS[1:,:] * 0.5 * (cBotC[1:,:] + cBotC[:-1,:])

        denomU0 = np.where(denomU == 0)
        denomU[denomU0] = 1
        denomV0 = np.where(denomV == 0)
        denomV[denomV0] = 1


        uIce = SeaIceMaskU * (SeaIceMassU * betaFacU * uIce
        + SeaIceMassU * recip_deltaTdyn * evpStarFac * uIceNm1
        + IceSurfStressX + stressDivX) / denomU
        vIce = SeaIceMaskV * (SeaIceMassV * betaFacV * vIce
        + SeaIceMassV * recip_deltaTdyn * evpStarFac * vIceNm1
        + IceSurfStressY + stressDivY) / denomV


        # fig, axs = plt.subplots(1,2, figsize=(7,3))
        # ax1 = axs[0].pcolormesh(uIce[OLy:-OLy,OLx:-OLx])
        # plt.colorbar(ax1, ax = axs[0])
        # axs[0].set_title('uIce')
        # ax2 = axs[1].pcolormesh(vIce[OLy:-OLy,OLx:-OLx])
        # plt.colorbar(ax2, ax = axs[1])
        # axs[1].set_title('vIce')
        # fig.tight_layout()
        # plt.show()


        uIce = fill_overlap(uIce)
        vIce = fill_overlap(vIce)

        uIcePm1 = SeaIceMaskU * (uIce - uIcePm1) * evpBetaU
        vIcePm1 = SeaIceMaskV * (vIce - vIcePm1) * evpBetaV

        resU[i] = uIcePm1**2 + vIcePm1**2
        resU = global_sum(resU)

    fig, axs = plt.subplots(1,2, figsize=(7,3))
    ax1 = axs[0].pcolormesh(sigma1[OLy:-OLy,OLx:-OLx])
    plt.colorbar(ax1, ax = axs[0])
    axs[0].set_title('sigma1')
    ax2 = axs[1].pcolormesh(sigma2[OLy:-OLy,OLx:-OLx])
    plt.colorbar(ax2, ax = axs[1])
    axs[1].set_title('sigma2')
    fig.tight_layout()
    plt.show()


    return uIce, vIce