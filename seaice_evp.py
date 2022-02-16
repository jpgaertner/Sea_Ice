import numpy as np

from seaice_params import *
from seaice_size import *

from dynamics_routines import strainrates, viscosities, \
    ocean_drag_coeffs, bottomdrag_coeffs, calc_stressdiv, calc_stress

from seaice_global_sum import global_sum
from seaice_averaging import c_point_to_z_point
from seaice_fill_overlap import fill_overlap_uv

### input
# uIce: zonal ice velocity
# vIce: meridional ice velocity
# uVel: zonal ocean surface velocity
# vVel: meridional ocean surface velocity
# hIceMean: mean ice thickness
# Area: ice cover fraction
# press0: maximum compressive stres
# IceSurfStressX0: zonal stress on ice surface at c point
# IceSurfStressY0: meridional stress on ice surface at c point
# R_low: water depth

### output:
# uIce: zonal ice velocity
# vIce: meridional ice velocity


evpTol = 1e-5
computeEvpResidual = True
printEvpResidual   = False
plotEvpResidual    = False
#
evpAlpha        = 500
evpBeta         = evpAlpha
useAdaptiveEVP  = True
aEVPalphaMin    = 5
aEvpCoeff       = 0.5
explicitDrag    = False
#
nEVPsteps = 500

def evp(uIce, vIce, uVel, vVel, hIceMean, Area, press0, secondOrderBC,
        IceSurfStressX0, IceSurfStressY0, SeaIceMassC, SeaIceMassU,
        SeaIceMassV, R_low):

    ##### initializations #####

    # change to input with default value = False?
    if useAdaptiveEVP:
        aEVPcStar = 4
        EVPcFac = deltaTdyn * aEVPcStar * ( np.pi * aEvpCoeff ) ** 2
    else:
        EVPcFac = 0
    # ... aEVPCoeff

    myTime, myIter = 0, 0
    sinWat = np.sin(np.deg2rad(waterTurnAngle))
    cosWat = np.cos(np.deg2rad(waterTurnAngle))

    zero2d = np.zeros((sNy+2*OLy,sNx+2*OLx))

    denom1 = 1 / evpAlpha
    denom2 = denom1

    # copy previous time step (n-1) of uIce, vIce
    uIceNm1 = uIce.copy()
    vIceNm1 = vIce.copy()

    # initialize adaptive EVP specific fields
    evpAlphaC = evpAlpha
    evpAlphaZ = evpAlpha
    evpBetaU  = evpBeta
    evpBetaV  = evpBeta

    # initialize fractional areas at velocity points
    areaW = 0.5 * (Area + np.roll(Area,1,1))
    areaS = 0.5 * (Area + np.roll(Area,1,0))

    ##### main loop #####

    # initializations
    # should initialised elsewhere (but this will work, too, just more
    # expensive)
    # sigma1  = zero2d.copy()
    # sigma2  = zero2d.copy()
    sigma11 = zero2d.copy()
    sigma22 = zero2d.copy()
    sigma12 = zero2d.copy()
    resSig  = np.array([None]*(nEVPsteps+1))
    resU    = np.array([None]*(nEVPsteps+1))

    iEVP = -1
    resEVP = evpTol*2
    while resEVP > evpTol and iEVP < nEVPsteps:
        iEVP = iEVP + 1

        if computeEvpResidual:
            # save previous (p-1) iteration for residual computation
            sig11Pm1 = sigma11.copy()
            sig22Pm1 = sigma22.copy()
            sig12Pm1 = sigma12.copy()
            uIcePm1  = uIce.copy()
            vIcePm1  = vIce.copy()

        # calculate strain rates and bulk moduli/ viscosities
        e11, e22, e12 = strainrates(uIce, vIce)

        zeta, eta, press = viscosities(e11,e22,e12,press0,iEVP,myTime,myIter)

        sig11, sig22, sig12 = calc_stress(
            e11, e22, e12, zeta, eta, press, iEVP, myTime, myIter)

        ##### first step stress equations #####
        # following Kimmritz et al. (2016)

        if useAdaptiveEVP:
            evpAlphaC = np.sqrt(zeta * EVPcFac / np.maximum(
                SeaIceMassC, 1e-4) * recip_rA) * iceMask
            evpAlphaC = np.maximum(evpAlphaC, aEVPalphaMin)
            denom1 = 1. / evpAlphaC
            denom2 = denom1.copy()

        sigma11 = sigma11 + (sig11 - sigma11) * denom1 * iceMask
        sigma22 = sigma22 + (sig22 - sigma22) * denom2 * iceMask

        # calculate sigma12 on z points
        if useAdaptiveEVP:
            evpAlphaZ = 0.5*( evpAlphaC + np.roll(evpAlphaC,1,0) )
            evpAlphaZ = 0.5*( evpAlphaZ + np.roll(evpAlphaC,1,1) )
            denom2 = 1. / evpAlphaZ

        sigma12 = sigma12 + (sig12 - sigma12) * denom2

        # import matplotlib.pyplot as plt
        # plt.clf(); plt.pcolormesh(sigma1); plt.colorbar(); plt.show()
        # sigma12 = fill_overlap(sigma12)

        # set up right hand side for stepping the velocity field
        # following Kimmritz et al. (2016)

        # calculate divergence of stress tensor
        stressDivX, stressDivY = calc_stressdiv(
            sigma11, sigma22, sigma12, iEVP, myTime, myIter)

        # drag coefficients for implicit/explicit treatment of drag
        cDrag = ocean_drag_coeffs(uIce, vIce, uVel, vVel)
        cBotC = bottomdrag_coeffs(uIce, vIce, hIceMean, Area, R_low)

        # over open ocean..., see comments in MITgcm: pkg/seaice/seaice_evp.F
        locMaskU = SeaIceMassU.copy()
        locMaskV = SeaIceMassV.copy()
        maskU = np.where(locMaskU != 0)
        locMaskU[maskU] = 1
        maskV = np.where(locMaskV != 0)
        locMaskV[maskV] = 1

        # set up anti symmetric drag force and add in ice ocean stress
        # (average to correct velocity points)
        duAtC = 0.5 * ( uVel-uIce + np.roll(uVel-uIce,-1,1) )
        dvAtC = 0.5 * ( vVel-vIce + np.roll(vVel-vIce,-1,0) )
        IceSurfStressX = IceSurfStressX0 + (
            0.5 * ( cDrag + np.roll(cDrag,1,1) ) * cosWat *  uVel
            - np.sign(fCori) * sinWat * 0.5 * (
                cDrag * dvAtC + np.roll(cDrag * dvAtC,1,1)
            ) * locMaskU
        ) * areaW
        IceSurfStressY = IceSurfStressY0 + (
            0.5 * ( cDrag + np.roll(cDrag,1,0) ) * cosWat * vVel
            + np.sign(fCori) * sinWat * 0.5 * (
                cDrag * duAtC  + np.roll(cDrag * duAtC,1,0)
            ) * locMaskV
        ) * areaS

        # add coriolis terms
        fvAtC = SeaIceMassC * fCori * 0.5 * ( vIce + np.roll(vIce,-1,0) )
        fuAtC = SeaIceMassC * fCori * 0.5 * ( uIce + np.roll(uIce,-1,1) )
        IceSurfStressX = IceSurfStressX + 0.5 * ( fvAtC + np.roll(fvAtC,1,1) )
        IceSurfStressY = IceSurfStressY - 0.5 * ( fuAtC + np.roll(fuAtC,1,0) )

        if useAdaptiveEVP:
            evpBetaU = 0.5 * ( evpAlphaC + np.roll(evpAlphaC,1,1) )
            evpBetaV = 0.5 * ( evpAlphaC + np.roll(evpAlphaC,1,0) )

        rMassU = 1./np.where(SeaIceMassU==0,np.Inf,SeaIceMassU)
        rMassV = 1./np.where(SeaIceMassV==0,np.Inf,SeaIceMassV)
        dragU = 0.5 * ( cDrag + np.roll(cDrag,1,1) ) * cosWat * areaW \
              + 0.5 * ( cBotC + np.roll(cBotC,1,1) )          * areaW
        dragV = 0.5 * ( cDrag + np.roll(cDrag,1,0) ) * cosWat * areaS \
              + 0.5 * ( cBotC + np.roll(cBotC,1,0) )          * areaS

        # step momentum equations with ice-ocean stress treated ...
        if explicitDrag:
            # ... explicitly
            IceSurfStressX = IceSurfStressX - uIce * dragU
            IceSurfStressY = IceSurfStressY - vIce * dragV
            denomU = 1.
            denomV = 1.
        else:
            # ... or implicitly
            denomU = 1. + dragU * deltaTdyn*rMassU/evpBetaU
            denomV = 1. + dragV * deltaTdyn*rMassV/evpBetaV

        # step momentum equations following Kimmritz et al. (2016)
        uIce = SeaIceMaskU * (
            uIce + (
                deltaTdyn*rMassU * ( IceSurfStressX + stressDivX )
                + ( uIceNm1 - uIce )
            ) / evpBetaU
        ) / denomU
        vIce = SeaIceMaskV * (
            vIce + (
                deltaTdyn*rMassV * ( IceSurfStressY + stressDivY )
                + ( vIceNm1 - vIce )
            ) / evpBetaV
        ) / denomV

        # uIce = fill_overlap(uIce)
        # vIce = fill_overlap(vIce)
        uIce, vIce = fill_overlap_uv(uIce, vIce)

        # residual computation
        if computeEvpResidual:
            sig11Pm1 = (sigma11 - sig11Pm1) * evpAlphaC * iceMask
            sig22Pm1 = (sigma22 - sig22Pm1) * evpAlphaC * iceMask
            sig12Pm1 = (sigma12 - sig12Pm1) * evpAlphaZ #* maskZ

            uIcePm1 = SeaIceMaskU * ( uIce - uIcePm1 ) * evpBetaU
            vIcePm1 = SeaIceMaskV * ( vIce - vIcePm1 ) * evpBetaV

            # if not explicitDrag:
            #     IceSurfStressX = IceSurfStressX - uIce * dragU
            #     IceSurfStressY = IceSurfStressY - vIce * dragV

            # uIcePm1 = ( SeaIceMassU * (uIce - uIceNm1)*recip_deltaTdyn
            #             - (IceSurfStressX + stressDivX)
            #            ) * SeaIceMaskU
            # vIcePm1 = ( SeaIceMassV * (vIce - vIceNm1)*recip_deltaTdyn
            #             - (IceSurfStressY + stressDivY)
            #            ) * SeaIceMaskV
            resSig[iEVP] = (sig11Pm1**2 + sig22Pm1**2
                         + sig12Pm1**2)[OLy:-OLy,OLx:-OLx].sum()
            resU[iEVP]   = ( uIcePm1**2
                          + vIcePm1**2 )[OLy:-OLy,OLx:-OLx].sum()
            resU[iEVP]   = global_sum(resU[iEVP])
            resSig[iEVP] = global_sum(resSig[iEVP])

            resEVP = resU[iEVP]
            if iEVP==0: resEVP0 = resEVP
            resEVP = resEVP/resEVP0

            if printEvpResidual:
                print ( 'evp resU, resSigma: %i %e %e'%(
                    iEVP, resU[iEVP], resSig[iEVP] ) )
            # print(i)
            # print(uIce.max(),vIce.max())
            # print(sigma1.max(), sigma2.max(), sigma12.max())

            # import matplotlib.pyplot as plt
            # fig2, ax = plt.subplots(nrows=2,ncols=1,sharex=True)
            # csf0=ax[0].pcolormesh(e12)
            # ax[0].set_title('e12')
            # plt.colorbar(csf0,ax=ax[0])
            # csf1=ax[1].pcolormesh(uIce)
            # plt.colorbar(csf1,ax=ax[1])
            # ax[1].set_title('uIce')
            # plt.show()


    if computeEvpResidual and plotEvpResidual:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(nrows=2,ncols=1,sharex=True)
        ax[0].semilogy(resU[:],'x-')
        ax[0].set_title('resU')
        ax[1].semilogy(resSig[:],'x-')
        ax[1].set_title('resSig')
        # s12 = sigma12 + np.roll(sigma12,-1,0)
        # s12 = 0.25*(s12 + np.roll(s12,-1,1))
        # s1=( sigma1 + np.sqrt(sigma2**2 + 4*s12**2) )/press
        # s2=( sigma1 - np.sqrt(sigma2**2 + 4*s12**2) )/press
        # csf0=ax[0].plot(s1.ravel(),s2.ravel(),'.');#plt.colorbar(csf0,ax=ax[0])
        # ax[0].plot([-1.4,0.1],[-1.4,0.1],'k-'); #plt.colorbar(csf0,ax=ax[0])
        # ax[0].set_title('sigma1')
        # csf1=ax[1].pcolormesh(sigma12); plt.colorbar(csf1,ax=ax[1])
        # ax[1].set_title('sigma2')
        plt.show()
        # print(resU)
        # print(resSig)


    return uIce, vIce
