from veros.core.operators import numpy as npx
from veros.core.operators import update, at
from veros import veros_kernel

from seaice_params import *
from seaice_size import *

from dynamics_routines import strainrates, viscosities, \
    ocean_drag_coeffs, bottomdrag_coeffs, calc_stressdiv, calc_stress

from seaice_global_sum import global_sum
from seaice_fill_overlap import fill_overlap_uv

### input
# uIce: zonal ice velocity
# vIce: meridional ice velocity
# uVel: zonal ocean surface velocity
# vVel: meridional ocean surface velocity
# hIceMean: mean ice thickness
# Area: ice cover fraction
# press0: maximum compressive stres
# WindForcingX: zonal stress on ice surface at c point
# WindForcingY: meridional stress on ice surface at c point
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
nEVPsteps = 500

@veros_kernel
def evp_solver(state):

    if useAdaptiveEVP:
        aEVPcStar = 4
        EVPcFac = deltaTdyn * aEVPcStar * ( npx.pi * aEvpCoeff ) ** 2
    else:
        EVPcFac = 0

    sinWat = npx.sin(npx.deg2rad(waterTurnAngle))
    cosWat = npx.cos(npx.deg2rad(waterTurnAngle))

    denom1 = 1 / evpAlpha
    denom2 = denom1

    # copy previous time step (n-1) of uIce, vIce
    uIceNm1 = update(state.variables.uIce, at[:,:], state.variables.uIce)
    vIceNm1 = update(state.variables.vIce, at[:,:], state.variables.vIce)

    # initialize adaptive EVP specific fields
    evpAlphaC = evpAlpha
    evpAlphaZ = evpAlpha
    evpBetaU  = evpBeta
    evpBetaV  = evpBeta

    # initialize fractional areas at velocity points
    areaW = 0.5 * (state.variables.Area + npx.roll(state.variables.Area,1,1))
    areaS = 0.5 * (state.variables.Area + npx.roll(state.variables.Area,1,0))

    zero2d = npx.zeros_like(iceMask)

    # should initialised elsewhere (but this will work, too, just more
    # expensive) #???
    # sigma1  = zero2d.copy()
    # sigma2  = zero2d.copy()
    sigma11 = update(zero2d, at[:,:], zero2d)
    sigma22 = update(zero2d, at[:,:], zero2d)
    sigma12 = update(zero2d, at[:,:], zero2d)
    resSig  = npx.zeros(nEVPsteps+1)
    resU    = npx.zeros(nEVPsteps+1)


    iEVP = -1
    resEVP = evpTol*2
    #while resEVP > evpTol and iEVP < nEVPsteps:
    for iEVP in range (nEVPsteps):
        print(iEVP)
        #iEVP = iEVP + 1

        if computeEvpResidual:
            # save previous (p-1) iteration for residual computation
            sig11Pm1 = update(sigma11, at[:,:], sigma11)
            sig22Pm1 = update(sigma22, at[:,:], sigma22)
            sig12Pm1 = update(sigma12, at[:,:], sigma12)
            uIcePm1  = update(state.variables.uIce, at[:,:], state.variables.uIce)
            vIcePm1  = update(state.variables.vIce, at[:,:], state.variables.vIce)

        # calculate strain rates and bulk moduli/ viscosities
        e11, e22, e12 = strainrates(state.variables.uIce, state.variables.vIce)

        zeta, eta, press = viscosities(state,e11,e22,e12,iEVP)

        sig11, sig22, sig12 = calc_stress(
            e11, e22, e12, zeta, eta, press, iEVP)

        ##### first step stress equations #####
        # following Kimmritz et al. (2016)

        if useAdaptiveEVP:
            evpAlphaC = npx.sqrt(zeta * EVPcFac / npx.maximum(
                state.variables.SeaIceMassC, 1e-4) * recip_rA) * iceMask
            evpAlphaC = npx.maximum(evpAlphaC, aEVPalphaMin)
            denom1 = 1. / evpAlphaC
            denom2 = update(denom1, at[:,:], denom1)

        sigma11 = sigma11 + (sig11 - sigma11) * denom1 * iceMask
        sigma22 = sigma22 + (sig22 - sigma22) * denom2 * iceMask

        # calculate sigma12 on z points
        if useAdaptiveEVP:
            evpAlphaZ = 0.5*( evpAlphaC + npx.roll(evpAlphaC,1,0) )
            evpAlphaZ = 0.5*( evpAlphaZ + npx.roll(evpAlphaC,1,1) )
            denom2 = 1. / evpAlphaZ

        sigma12 = sigma12 + (sig12 - sigma12) * denom2

        # import matplotlib.pyplot as plt
        # plt.clf(); plt.pcolormesh(sigma1); plt.colorbar(); plt.show()
        # sigma12 = fill_overlap(sigma12)

        # set up right hand side for stepping the velocity field
        # following Kimmritz et al. (2016)

        # calculate divergence of stress tensor
        stressDivX, stressDivY = calc_stressdiv(
            sigma11, sigma22, sigma12, iEVP)

        # drag coefficients for implicit/explicit treatment of drag
        cDrag = ocean_drag_coeffs(state, state.variables.uIce, state.variables.vIce)
        cBotC = bottomdrag_coeffs(state, state.variables.uIce, state.variables.vIce)

        # over open ocean..., see comments in MITgcm: pkg/seaice/seaice_evp.F
        locMaskU = update(state.variables.SeaIceMassU, at[:,:], state.variables.SeaIceMassU)
        locMaskV = update(state.variables.SeaIceMassV, at[:,:], state.variables.SeaIceMassU)
        locMaskU = npx.where(locMaskU != 0, 1, locMaskU)
        locMaskV = npx.where(locMaskV != 0, 1, locMaskV)

        # set up anti symmetric drag force and add in ice ocean stress
        # (average to correct velocity points)
        duAtC = 0.5 * ( state.variables.uVel-state.variables.uIce
                + npx.roll(state.variables.uVel-state.variables.uIce,-1,1) )
        dvAtC = 0.5 * ( state.variables.vVel-state.variables.vIce
                + npx.roll(state.variables.vVel-state.variables.vIce,-1,0) )
        ForcingX = state.variables.WindForcingX + (
            0.5 * ( cDrag + npx.roll(cDrag,1,1) ) * cosWat *  state.variables.uVel
            - npx.sign(fCori) * sinWat * 0.5 * (
                cDrag * dvAtC + npx.roll(cDrag * dvAtC,1,1)
            ) * locMaskU
        ) * areaW
        ForcingY = state.variables.WindForcingY + (
            0.5 * ( cDrag + npx.roll(cDrag,1,0) ) * cosWat * state.variables.vVel
            + npx.sign(fCori) * sinWat * 0.5 * (
                cDrag * duAtC  + npx.roll(cDrag * duAtC,1,0)
            ) * locMaskV
        ) * areaS

        # add coriolis terms
        fvAtC = state.variables.SeaIceMassC * fCori * 0.5 \
                * ( state.variables.vIce + npx.roll(state.variables.vIce,-1,0) )
        fuAtC = state.variables.SeaIceMassC * fCori * 0.5 \
                * ( state.variables.uIce + npx.roll(state.variables.uIce,-1,1) )
        ForcingX = ForcingX + 0.5 * ( fvAtC + npx.roll(fvAtC,1,1) )
        ForcingY = ForcingY - 0.5 * ( fuAtC + npx.roll(fuAtC,1,0) )

        if useAdaptiveEVP:
            evpBetaU = 0.5 * ( evpAlphaC + npx.roll(evpAlphaC,1,1) )
            evpBetaV = 0.5 * ( evpAlphaC + npx.roll(evpAlphaC,1,0) )

        rMassU = 1./npx.where(state.variables.SeaIceMassU==0,npx.inf,state.variables.SeaIceMassU)
        rMassV = 1./npx.where(state.variables.SeaIceMassV==0,npx.inf,state.variables.SeaIceMassV)
        dragU = 0.5 * ( cDrag + npx.roll(cDrag,1,1) ) * cosWat * areaW \
              + 0.5 * ( cBotC + npx.roll(cBotC,1,1) )          * areaW
        dragV = 0.5 * ( cDrag + npx.roll(cDrag,1,0) ) * cosWat * areaS \
              + 0.5 * ( cBotC + npx.roll(cBotC,1,0) )          * areaS

        # step momentum equations with ice-ocean stress treated ...
        if explicitDrag:
            # ... explicitly
            ForcingX = ForcingX - state.variables.uIce * dragU
            ForcingY = ForcingY - state.variables.vIce * dragV
            denomU = 1.
            denomV = 1.
        else:
            # ... or implicitly
            denomU = 1. + dragU * deltaTdyn*rMassU/evpBetaU
            denomV = 1. + dragV * deltaTdyn*rMassV/evpBetaV

        # step momentum equations following Kimmritz et al. (2016)
        state.variables.uIce = SeaIceMaskU * (
            state.variables.uIce + (
                deltaTdyn*rMassU * ( ForcingX + stressDivX )
                + ( uIceNm1 - state.variables.uIce )
            ) / evpBetaU
        ) / denomU
        state.variables.vIce = SeaIceMaskV * (
            state.variables.vIce + (
                deltaTdyn*rMassV * ( ForcingY + stressDivY )
                + ( vIceNm1 - state.variables.vIce )
            ) / evpBetaV
        ) / denomV

        # uIce = fill_overlap(uIce)
        # vIce = fill_overlap(vIce)
        state.variables.uIce, state.variables.vIce = fill_overlap_uv(
                                state.variables.uIce, state.variables.vIce)

        # residual computation
        if computeEvpResidual:
            sig11Pm1 = (sigma11 - sig11Pm1) * evpAlphaC * iceMask
            sig22Pm1 = (sigma22 - sig22Pm1) * evpAlphaC * iceMask
            sig12Pm1 = (sigma12 - sig12Pm1) * evpAlphaZ #* maskZ

            uIcePm1 = SeaIceMaskU * ( state.variables.uIce - uIcePm1 ) * evpBetaU
            vIcePm1 = SeaIceMaskV * ( state.variables.vIce - vIcePm1 ) * evpBetaV

            # if not explicitDrag:
            #     ForcingX = ForcingX - uIce * dragU
            #     ForcingY = ForcingY - vIce * dragV

            # uIcePm1 = ( SeaIceMassU * (uIce - uIceNm1)*recip_deltaTdyn
            #             - (ForcingX + stressDivX)
            #            ) * SeaIceMaskU
            # vIcePm1 = ( SeaIceMassV * (vIce - vIceNm1)*recip_deltaTdyn
            #             - (ForcingY + stressDivY)
            #            ) * SeaIceMaskV

            resSig = update(resSig, at[iEVP], (sig11Pm1**2 + sig22Pm1**2
                        + sig12Pm1**2)[oly:-oly,olx:-olx].sum())
            resSig = update(resSig, at[iEVP], global_sum(resSig[iEVP]))
            resU = update(resU, at[iEVP], (uIcePm1**2
                        + vIcePm1**2 )[oly:-oly,olx:-olx].sum())
            resU = update(resU, at[iEVP], global_sum(resU[iEVP]))

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
        # s12 = sigma12 + npx.roll(sigma12,-1,0)
        # s12 = 0.25*(s12 + npx.roll(s12,-1,1))
        # s1=( sigma1 + npx.sqrt(sigma2**2 + 4*s12**2) )/press
        # s2=( sigma1 - npx.sqrt(sigma2**2 + 4*s12**2) )/press
        # csf0=ax[0].plot(s1.ravel(),s2.ravel(),'.');#plt.colorbar(csf0,ax=ax[0])
        # ax[0].plot([-1.4,0.1],[-1.4,0.1],'k-'); #plt.colorbar(csf0,ax=ax[0])
        # ax[0].set_title('sigma1')
        # csf1=ax[1].pcolormesh(sigma12); plt.colorbar(csf1,ax=ax[1])
        # ax[1].set_title('sigma2')
        plt.show()
        # print(resU)
        # print(resSig)


    return state.variables.uIce, state.variables.vIce