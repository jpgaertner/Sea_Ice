import numpy as np

from seaice_params import *
from seaice_size import *

from seaice_strainrates import strainrates
from seaice_ocean_drag_coeffs import ocean_drag_coeffs
from seaice_bottomdrag_coeffs import bottomdrag_coeffs

### input
# uIce: zonal ice velocity
# vIce: meridional ice velocity
# uVel: zonal ocean surface velocity 
# vVel: meridional ocean surface velocity
# hIceMean: mean ice thickness
# Area: ice cover fraction
# press0: ocean surface pressure
# secondOrderBC: flag
# IceSurfStressX0: zonal stress on ice surface at c point?
# IceSurfStressY0: meridional stress on ice surface at c point

### output:
# uIce: zonal ice velocity
# vIce: meridional ice velocity

# ???
nEVPsteps = 1
deltaTevp = 1
nEVPstep = 1
sigma1 = 1
sigma2 = 1
sigma12 = 1
rAz = 1
rAw = 1
recip_rAw = 1
rAs = 1
recip_rAs = 1
tensileStrFac = 1

SeaIceMassC = 1
SeaIceMassU = 1
SeaIceMassV = 1


def evp(uIce, vIce, uVel, vVel, hIceMean, Area, press0, secondOrderBC, IceSurfStressX0, IceSurfStressY0):

    ##### initializations #####

    useAdaptiveEVP = False
    #      IF ( SEAICEaEvpCoeff .NE. UNSET_RL ) useAdaptiveEVP = .TRUE.
    if useAdaptiveEVP:
        EVPcFac = deltaTdyn
    # why the difference in deltas? deltatDyn is used here for the first time. why is deltaTtherm used in advection?
    else:
        EVPcFac = 0

    sinWat = np.sin(np.deg2rad(waterTurnAngle))
    cosWat = np.cos(np.deg2rad(waterTurnAngle))

    PlasDefCoeffSq = PlasDefCoeff**2
    recip_PlasDefCoeffSq = 1 / PlasDefCoeffSq
    deltaMinSq = deltaMin**2 #what does deltaMin mean?

    #nEVPSteps = SEAICEnEVPstarSteps with SEAICEnEVPstarSteps = INT(SEAICE_deltaTdyn/SEAICE_deltaTevp)
    #SEAICE_deltaTevp = ?

    evpAlpha = 2 * evpTauRelax / deltaTevp
    #what does evpTauRelax mean?
    recip_evpAlpha = 1 / evpAlpha
    evpStarFac = 1
    evpRevFac = 1
    recip_evpRevFac = recip_PlasDefCoeffSq

    denom1 = np.ones((sNy+2*OLy,sNx+2*OLx)) / (evpAlpha)
    denom2 = denom1.copy() #why?

    evpBeta = evpAlpha #why?
    betaFac = evpBeta * recip_deltaTdyn
    betaFacU = betaFac
    betaFacV = betaFac
    betaFacP1 = betaFac + evpStarFac * recip_deltaTdyn
    betaFacP1U = betaFacP1
    betaFacP1V = betaFacP1

    nEVPstepsMax = nEVPsteps #really needed? is nEVPsteps an input?

    # make local copies of uIce, vIce
    uIceLoc = uIce.copy()
    vIceLoc = vIce.copy()

    # initialize adaptive EVP specific fields
    evpAlphaC = np.ones((sNy+2*OLy,sNx+2*OLx)) * evpAlpha
    evpAlphaZ = np.ones((sNy+2*OLy,sNx+2*OLx)) * evpAlpha
    evpBetaU = np.ones((sNy+2*OLy,sNx+2*OLx)) * evpBeta
    evpBetaV = np.ones((sNy+2*OLy,sNx+2*OLx)) * evpBeta

    # initialize fractional areas at velocity points
    areaW = np.zeros((sNy+2*OLy,sNx+2*OLx))
    areaS = np.zeros((sNy+2*OLy,sNx+2*OLx))
    areaW[OLy:-OLy,OLx:-OLx] = 0.5 * (Area[OLy:-OLy,OLx:-OLx] + Area[OLy:-OLy,OLx-1:-OLx-1])
    areaS[OLy:-OLy,OLx:-OLx] = 0.5 * (Area[OLy:-OLy,OLx:-OLx] + Area[OLy-1:-OLy-1,OLx:-OLx])


    ##### main loop #####

    e12Csq = np.zeros((sNy+2*OLy,sNx+2*OLx))
    deltaSq = np.zeros((sNy+2*OLy,sNx+2*OLx))
    deltaC = np.zeros((sNy+2*OLy,sNx+2*OLx))
    zetaC = np.zeros((sNy+2*OLy,sNx+2*OLx))
    zetaZ = np.zeros((sNy+2*OLy,sNx+2*OLx))
    deltaZ = np.zeros((sNy+2*OLy,sNx+2*OLx))
    stressDivX = np.zeros((sNy+2*OLy,sNx+2*OLx))
    stressDivY = np.zeros((sNy+2*OLy,sNx+2*OLx))
    pressC = np.zeros((sNy+2*OLy,sNx+2*OLx))
    div = np.zeros((sNy+2*OLy,sNx+2*OLx))
    tension = np.zeros((sNy+2*OLy,sNx+2*OLx))
    shear = np.zeros((sNy+2*OLy,sNx+2*OLx))
    sig11 = np.zeros((sNy+2*OLy,sNx+2*OLx))
    sig22 = np.zeros((sNy+2*OLy,sNx+2*OLx))
    resTile = np.zeros((sNy+2*OLy,sNx+2*OLx))


    for i in range(nEVPstepsMax):
        if i <= nEVPstep:
            # calculate strain rates and bulk moduli/ viscosities
            e11, e22, e12 = strainrates(uIce, vIce, secondOrderBC)

            # save previous iteration
            # doesnt this result in eg sig11Pm1 - sigma1 = 0?
            sig11Pm1 = sigma1.copy()
            sig22Pm1 = sigma2.copy()
            sig12Pm1 = sigma12.copy()
            uIcePm1 = uIce.copy()
            vIcePm1 = vIce.copy()

            ep = e11 + e22
            em = e11 - e22

            # use area weighted average of squares of e12 (more accurate)
            e12Csq[1:-1,1:-1] = 0.25 * recip_rA[1:-1,1:-1] * (rAz[1:-1,1:-1] * e12[1:-1,1:-1]**2 + rAz[1:-1,2:] * e12[1:-1,2:]**2 + rAz[2:,1:-1] * e12[2:,1:-1]**2 + rAz[2:,2:] * e12[2:,2:]**2)
            deltaSq[OLy-1:-OLy+1,OLx-1:-OLx+1] = ep[OLy-1:-OLy+1,OLx-1:-OLx+1]**2 + recip_PlasDefCoeffSq * em[OLy-1:-OLy+1,OLx-1:-OLx+1]**2 + recip_PlasDefCoeffSq * 4 * e12Csq[OLy-1:-OLy+1,OLx-1:-OLx+1]
            deltaC[OLy-1:-OLy+1,OLx-1:-OLx+1] = np.sqrt(deltaSq[OLy-1:-OLy+1,OLx-1:-OLx+1])

            # smooth regularization of delta for better differentiability
            deltaCreg = deltaC + deltaMin

            zetaC[OLy-1:-OLy+1,OLx-1:-OLx+1] = 0.5 * (press0[OLy-1:-OLy+1,OLx-1:-OLx+1] * (1 + tensileStrFac[OLy-1:-OLy+1,OLx-1:-OLx+1])) / deltaCreg

            if useAdaptiveEVP:
                evpAlphaC[OLy-1:-OLy+1,OLx-1:-OLx+1] = np.sqrt(zetaC[OLy-1:-OLy+1,OLx-1:-OLx+1] * EVPcFac / np.maximum(SeaIceMassC[OLy-1:-OLy+1,OLx-1:-OLx+1], 1e-4) * recip_rA[OLy-1:-OLy+1,OLx-1:-OLx+1]) * hIceMeanMask[OLy-1:-OLy+1,OLx-1:-OLx+1]
                evpAlphaC[OLy-1:-OLy+1,OLx-1:-OLx+1] = np.maximum(evpAlphaC[OLy-1:-OLy+1,OLx-1:-OLx+1], evpAlphaMin)

            # calculate zetaZ, deltaZ by simple averaging
            sumNorm = hIceMean[OLy:-OLy+1,OLx:-OLx+1] + hIceMean[OLy:-OLy+1,OLx-1:-OLx] + hIceMean[OLy-1:-OLy,OLx:-OLx+1] + hIceMean[OLy-1:-OLy,OLx-1:-OLx]
            if sumNorm > 0:
                sumNorm = 1 / sumNorm
            zetaZ[OLy:-OLy+1,OLx:-OLx+1] = sumNorm * (zetaC[OLy:-OLy+1,OLx:-OLx+1] + zetaC[OLy-1:-OLy,OLx-1:-OLx] + zetaC[OLy:-OLy+1,OLx-1:-OLx] + zetaC[OLy-1:-OLy,OLx:-OLx+1])
            deltaZ[OLy:-OLy+1,OLx:-OLx+1] = sumNorm * (deltaC[OLy:-OLy+1,OLx:-OLx+1] + deltaC[OLy-1:-OLy,OLx-1:-OLx] + deltaC[OLy:-OLy+1,OLx-1:-OLx] + deltaC[OLy-1:-OLy,OLx:-OLx+1])

            # recalculate pressure
            pressC[OLy-1:-OLy+1,OLx-1:-OLx+1] = (press0[OLy-1:-OLy+1,OLx-1:-OLx+1] * (1 - pressReplFac) + 2 * zetaC[OLy-1:-OLy+1,OLx-1:-OLx+1] * deltaC[OLy-1:-OLy+1,OLx-1:-OLx+1] * pressReplFac / (1 + tensileStrFac[OLy-1:-OLy+1,OLx-1:-OLx+1])) * (1 - tensileStrFac[OLy-1:-OLy+1,OLx-1:-OLx+1])
            
            div[OLy-1:-OLy,OLx-1:-OLx] = (2 * zetaC[OLy-1:-OLy,OLx-1:-OLx] * ep[OLy-1:-OLy,OLx-1:-OLx] - pressC[OLy-1:-OLy,OLx-1:-OLx]) * hIceMeanMask[OLy-1:-OLy,OLx-1:-OLx]
            tension[OLy-1:-OLy,OLx-1:-OLx] = 2 * zetaC[OLy-1:-OLy,OLx-1:-OLx] * em[OLy-1:-OLy,OLx-1:-OLx] * hIceMeanMask[OLy-1:-OLy,OLx-1:-OLx]
            shear[OLy:-OLy+1,OLx:-OLx+1] = 2 * zetaZ[OLy:-OLy+1,OLx:-OLx+1] * e12[OLy:-OLy+1,OLx:-OLx+1]


            ##### first step stress equations #####

            if useAdaptiveEVP:
                denom1[OLy-1:-OLy,OLx-1:OLx] = 1 / evpAlphaC[OLy-1:-OLy,OLx-1:OLx]
                denom2[OLy-1:-OLy,OLx-1:OLx] = denom1[OLy-1:-OLy,OLx-1:OLx]
            
            # calculate sigma1, sigma2 on c points
            sigma1[OLy-1:-OLy,OLx-1:OLx] = (sigma1[OLy-1:-OLy,OLx-1:OLx] * (evpAlphaC[OLy-1:-OLy,OLx-1:OLx] - evpRevFac) + div[OLy-1:-OLy,OLx-1:OLx]) * denom1[OLy-1:-OLy,OLx-1:OLx] * hIceMeanMask[OLy-1:-OLy,OLx-1:OLx]
            sigma2[OLy-1:-OLy,OLx-1:OLx] = (sigma2[OLy-1:-OLy,OLx-1:OLx] * (evpAlphaC[OLy-1:-OLy,OLx-1:OLx] - evpRevFac) + tension[OLy-1:-OLy,OLx-1:OLx] * recip_evpRevFac) * denom2[OLy-1:-OLy,OLx-1:OLx] * hIceMeanMask[OLy-1:-OLy,OLx-1:OLx]

            # recover sigma11 and sigma22
            sig11[OLy-1:-OLy,OLx-1:OLx] = 0.5 * (sigma1[OLy-1:-OLy,OLx-1:OLx] + sigma2[OLy-1:-OLy,OLx-1:OLx])
            sig22[OLy-1:-OLy,OLx-1:OLx] = 0.5 * (sigma1[OLy-1:-OLy,OLx-1:OLx] - sigma2[OLy-1:-OLy,OLx-1:OLx])

            # calculate sigma12 on z points
            if useAdaptiveEVP:
                evpAlphaZ[OLy:-OLy+1,OLx:-OLx+1] = 0.25 * (evpAlphaC[OLy:-OLy+1,OLx:-OLx+1] + evpAlphaC[OLy-1:-OLy,OLx-1:-OLx] + evpAlphaC[OLy:-OLy+1,OLx-1:-OLx] + evpAlphaC[OLy-1:-OLy,OLx:-OLx+1])
                denom2[OLy:-OLy+1,OLx:-OLx+1] = 1 / evpAlphaZ[OLy:-OLy+1,OLx:-OLx+1]
            sigma12[OLy:-OLy+1,OLx:-OLx+1] = (sigma12[OLy:-OLy+1,OLx:-OLx+1] * (evpAlphaZ[OLy:-OLy+1,OLx:-OLx+1] - evpRevFac) + shear[OLy:-OLy+1,OLx:-OLx+1] * recip_evpRevFac) * denom2[OLy:-OLy+1,OLx:-OLx+1]

            # calculate divergence of stress tensor
            stressDivX[OLy:-OLy,OLx:-OLx] = (sig11[OLy:-OLy,OLx:-OLx] * dyF[OLy:-OLy,OLx:-OLx] - sig11[OLy:-OLy,OLx-1:-OLx-1] * dyF[OLy:-OLy,OLx-1:-OLx-1] + sigma12[OLy+1:-OLy+1,OLx:-OLx] * dxV[OLy+1:-OLy+1,OLx:-OLx] - sigma12[OLy:-OLy,OLx:-OLx] * dxV[OLy:-OLy,OLx:-OLx]) * recip_rAw[OLy:-OLy,OLx:-OLx]
            stressDivY[OLy:-OLy,OLx:-OLx] = (sig22[OLy:-OLy,OLx:-OLx] * dxF[OLy:-OLy,OLx:-OLx] - sig22[OLy-1:-OLy-1,OLx:-OLx] * dxF[OLy-1:-OLy-1,OLx:-OLx] + sigma12[OLy:-OLy,OLx+1:-OLx+1] * dyU[OLy:-OLy,OLx+1:-OLx+1] - sigma12[OLy:-OLy,OLx:-OLx] * dyU[OLy:-OLy,OLx:-OLx]) * recip_rAs[OLy:-OLy,OLx:-OLx]

            sig11Pm1[OLy:-OLy,OLx:-OLx] = (sigma1[OLy:-OLy,OLx:-OLx] - sig11Pm1[OLy:-OLy,OLx:-OLx]) * evpAlphaC[OLy:-OLy,OLx:-OLx]
            sig22Pm1[OLy:-OLy,OLx:-OLx] = (sigma2[OLy:-OLy,OLx:-OLx] - sig22Pm1[OLy:-OLy,OLx:-OLx]) * evpAlphaC[OLy:-OLy,OLx:-OLx]
            sig12Pm1[OLy:-OLy,OLx:-OLx] = (sigma12[OLy:-OLy,OLx:-OLx] - sig12Pm1[OLy:-OLy,OLx:-OLx]) * evpAlphaZ[OLy:-OLy,OLx:-OLx]

            # 
            resTile = resTile + sig11Pm1[OLy:-OLy,OLx:-OLx]**2 + sig22Pm1[OLy:-OLy,OLx:-OLx]**2 + sig12Pm1[OLy:-OLy,OLx:-OLx]**2
            resLoc = 0

            # call GLOBAL_SUM_TILE_RL(resTile, resLoc)
            resLoc = np.sqrt(resLoc)

            # set up right hand side for stepping the velocity field
            cDrag = ocean_drag_coeffs(uIce, vIce, uVel, vVel)
            cBotC = bottomdrag_coeffs(uIce, vIce, hIceMean, Area)

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
            IceSurfStressX[OLy:-OLy,OLx:-OLx] = IceSurfStressX0[OLy:-OLy,OLx:-OLx] + (0.5 * (cDrag[OLy:-OLy,OLx:-OLx] + cDrag[OLy:-OLy,OLx-1:-OLx-1]) * cosWat * uVel[OLy:-OLy,OLx:-OLx] - np.sign(fCori[OLy:-OLy,OLx:-OLx]) * sinWat * 0.5 * (cDrag[OLy:-OLy,OLx:-OLx] * 0.5 * (vVel[OLy:-OLy,OLx:-OLx] - vIce[OLy:-OLy,OLx:-OLx] + vVel[OLy+1:-OLy+1,OLx:-OLx] - vIce[OLy+1:-OLy+1,OLx:-OLx]) + cDrag[OLy:-OLy,OLx-1:-OLx-1] * 0.5 * (vVel[OLy:-OLy,OLx-1:-OLx-1] - vIce[OLy:-OLy,OLx-1:-OLx-1] + vVel[OLy+1:-OLy+1,OLx-1:-OLx-1] - vIce[OLy+1:-OLy+1,OLx-1:-OLx-1])) * locMaskU[OLy:-OLy,OLx:-OLx]) * areaW[OLy:-OLy,OLx:-OLx]
            IceSurfStressY[OLy:-OLy,OLx:-OLx] = IceSurfStressY0[OLy:-OLy,OLx:-OLx] + (0.5 * (cDrag[OLy:-OLy,OLx:-OLx] + cDrag[OLy-1:-OLy-1,OLx:-OLx]) * cosWat * vVel[OLy:-OLy,OLx:-OLx] + np.sign(fCori[OLy:-OLy,OLx:-OLx]) * sinWat * 0.5 * (cDrag[OLy:-OLy,OLx:-OLx] * 0.5 * (uVel[OLy:-OLy,OLx:-OLx] - uIce[OLy:-OLy,OLx:-OLx] + uVel[OLy:-OLy,OLx+1:-OLx+1] - uIce[OLy:-OLy,OLx+1:-OLx+1]) + cDrag[OLy-1:-OLy-1,OLx:-OLx] * 0.5 * (uVel[OLy-1:-OLy-1,OLx:-OLx] - uIce[OLy-1:-OLy-1,OLx:-OLx] + uVel[OLy-1:-OLy-1,OLx+1:-OLx+1] - uIce[OLy-1:-OLy-1,OLx+1:-OLx+1])) * locMaskV[OLy:-OLy,OLx:-OLx]) * areaS[OLy:-OLy,OLx:-OLx]

            # add coriolis terms
            IceSurfStressX[OLy:-OLy,OLx:-OLx] = IceSurfStressX[OLy:-OLy,OLx:-OLx] + 0.5 * (SeaIceMassC[OLy:-OLy,OLx:-OLx] + fCori[OLy:-OLy,OLx:-OLx] * 0.5 * (vIce[OLy:-OLy,OLx:-OLx] + vIce[OLy+1:-OLy+1,OLx:-OLx]) + SeaIceMassC[OLy:-OLy,OLx-1:-OLx-1] * fCori[OLy:-OLy,OLx-1:-OLx-1] * 0.5 * (vIce[OLy:-OLy,OLx-1:-OLx-1] + vIce[OLy+1:-OLy+1,OLx-1:-OLx-1]))
            IceSurfStressY[OLy:-OLy,OLx:-OLx] = IceSurfStressY[OLy:-OLy,OLx:-OLx] - 0.5 * (SeaIceMassC[OLy:-OLy,OLx:-OLx] * fCori[OLy:-OLy,OLx:-OLx] * 0.5 * (uIce[OLy:-OLy,OLx:-OLx] + uIce[OLy:-OLy,OLx+1:-OLx+1]) + SeaIceMassC[OLy-1:-OLy-1,OLx:-OLx] * fCori[OLy-1:-OLy-1,OLx:-OLx] * 0.5 * (uIce[OLy-1:-OLy-1,OLx:-OLx] + uIce[OLy-1:-OLy-1,OLx+1:-OLx+1]))

            # step momentum equations with ice-ocean stress treated implicitly
            if useAdaptiveEVP:
                evpBetaU[OLy:-OLy,OLx:-OLx] = 0.5 * (evpAlphaC[OLy:-OLy,OLx-1:-OLx-1] + evpAlphaC[OLy:-OLy,OLx:-OLx])
                evpBetaV[OLy:-OLy,OLx:-OLx] = 0.5 * (evpAlphaC[OLy-1:-OLy-1,OLx:-OLx] + evpAlphaC[OLy:-OLy,OLx:-OLx])

            betaFacU = evpBetaU[OLy:-OLy,OLx:-OLx] * recip_deltaTdyn
            betaFacV = evpBetaV[OLy:-OLy,OLx:-OLx] * recip_deltaTdyn
            tmp = evpStarFac * recip_deltaTdyn
            betaFacP1V = betaFacV * tmp
            betaFacP1U = betaFacU + tmp
            denomU = (SeaIceMassU[OLy:-OLy,OLx:-OLx] * betaFacP1U + 0.5 * (cDrag[OLy:-OLy,OLx:-OLx] + cDrag[OLy:-OLy,OLx-1:-OLx-1]) * cosWat * areaW[OLy:-OLy,OLx:-OLx]) + areaW[OLy:-OLy,OLx:-OLx] * 0.5 * (cBotC[OLy:-OLy,OLx:-OLx] + cBotC[OLy:-OLy,OLx-1:-OLx-1])
            denomV = (SeaIceMassV[OLy:-OLy,OLx:-OLx] * betaFacP1V + 0.5 * (cDrag[OLy:-OLy,OLx:-OLx] + cDrag[OLy-1:-OLy-1,OLx:-OLx]) * cosWat * areaS[OLy:-OLy,OLx:-OLx]) + areaS[OLy:-OLy,OLx:-OLx] * 0.5 * (cBotC[OLy:-OLy,OLx:-OLx] + cBotC[OLy-1:-OLy-1,OLx:-OLx])

            denomU0 = np.where(denomU == 0)
            denomU[denomU0] = 1
            denomV0 = np.where(denomV == 0)
            denomV[denomV0] = 1

            uIce[OLy:-OLy,OLx:-OLx] = SeaIceMaskU[OLy:-OLy,OLx:-OLx] * (SeaIceMassU[OLy:-OLy,OLx:-OLx] * betaFacU * uIce[OLy:-OLy,OLx:-OLx] + SeaIceMassU[OLy:-OLy,OLx:-OLx] * recip_deltaTdyn * evpStarFac * uIceLoc[OLy:-OLy,OLx:-OLx] + IceSurfStressX[OLy:-OLy,OLx:-OLx] + stressDivX[OLy:-OLy,OLx:-OLx]) / denomU
            vIce[OLy:-OLy,OLx:-OLx] = SeaIceMaskV[OLy:-OLy,OLx:-OLx] * (SeaIceMassV[OLy:-OLy,OLx:-OLx] * betaFacV * vIce[OLy:-OLy,OLx:-OLx] + SeaIceMassV[OLy:-OLy,OLx:-OLx] * recip_deltaTdyn * evpStarFac * vIceLoc[OLy:-OLy,OLx:-OLx] + IceSurfStressY[OLy:-OLy,OLx:-OLx] + stressDivY[OLy:-OLy,OLx:-OLx]) / denomV

            uIce = fill_overlap(uIce)
            vIce = fill_overlap(vIce)

            uIcePm1[OLy:-OLy,OLx:-OLx] = SeaIceMaskU[OLy:-OLy,OLx:-OLx] * (uIce[OLy:-OLy,OLx:-OLx] - uIcePm1[OLy:-OLy,OLx:-OLx]) * evpBetaU[OLy:-OLy,OLx:-OLx]
            vIcePm1[OLy:-OLy,OLx:-OLx] = SeaIceMaskV[OLy:-OLy,OLx:-OLx] * (vIce[OLy:-OLy,OLx:-OLx] - vIcePm1[OLy:-OLy,OLx:-OLx]) * evpBetaV[OLy:-OLy,OLx:-OLx]

            resTile = resTile + uIcePm1[OLy:-OLy,OLx:-OLx]** + vIcePm1[OLy:-OLy,OLx:-OLx]**2
            resLoc = 0

            #CALL GLOBAL_SUM_TILE_RL( resTile, resloc)
            resLoc = np.sqrt(resLoc)


    return uIce, vIce