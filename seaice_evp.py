from veros.core.operators import numpy as npx
from veros.core.operators import update, at, for_loop
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
useAdaptiveEVP  = False #now mEVP is used
aEVPalphaMin    = 5
aEvpCoeff       = 0.5
explicitDrag    = False
nEVPsteps = 500

@veros_kernel
def evp_solver_body(iEVP, arg_body):

    state       = arg_body[0]
    uIce        = arg_body[1]
    vIce        = arg_body[2]
    uIceNm1     = arg_body[3]
    vIceNm1     = arg_body[4]
    sigma11     = arg_body[5]
    sigma22     = arg_body[6]
    sigma12     = arg_body[7]
    denom1      = arg_body[8]
    denom2      = arg_body[9]
    EVPcFac     = arg_body[10]
    evpAlphaC   = arg_body[11]
    evpAlphaZ   = arg_body[12]
    evpBetaU    = arg_body[13]
    evpBetaV    = arg_body[14]
    resSig      = arg_body[15]
    resU        = arg_body[16]

    if computeEvpResidual:
        # save previous (p-1) iteration for residual computation
        sig11Pm1 = sigma11
        sig22Pm1 = sigma22
        sig12Pm1 = sigma12
        uIcePm1  = uIce
        vIcePm1  = vIce

    # calculate strain rates and bulk moduli/ viscosities
    e11, e22, e12 = strainrates(state,uIce,vIce,iEVP)

    zeta, eta, press = viscosities(state,e11,e22,e12,iEVP)

    sig11, sig22, sig12 = calc_stress(e11,e22,e12,zeta,eta,press,iEVP)

    ##### first step stress equations following Kimmritz et al. (2016) #####

    if useAdaptiveEVP:
        evpAlphaC = npx.sqrt(zeta * EVPcFac / npx.maximum(
            state.variables.SeaIceMassC, 1e-4) * recip_rA) * iceMask
        evpAlphaC = npx.maximum(evpAlphaC, aEVPalphaMin)
        denom1 = 1. / evpAlphaC
        denom2 = denom1


    sigma11 = sigma11 + (sig11 - sigma11) * denom1 * iceMask
    sigma22 = sigma22 + (sig22 - sigma22) * denom2 * iceMask

    # calculate sigma12 on z points
    if useAdaptiveEVP:
        evpAlphaZ = 0.5*( evpAlphaC + npx.roll(evpAlphaC,1,0) )
        evpAlphaZ = 0.5*( evpAlphaZ + npx.roll(evpAlphaZ,1,1) )
        denom2 = 1. / evpAlphaZ

    sigma12 = sigma12 + (sig12 - sigma12) * denom2

    # import matplotlib.pyplot as plt
    # plt.clf(); plt.pcolormesh(sigma1); plt.colorbar(); plt.show()
    # sigma12 = fill_overlap(sigma12)

    # set up right hand side for stepping the velocity field
    # following Kimmritz et al. (2016)

    # calculate divergence of stress tensor
    stressDivX, stressDivY = calc_stressdiv(sigma11, sigma22, sigma12, iEVP)

    # drag coefficients for implicit/explicit treatment of drag
    cDrag = ocean_drag_coeffs(state, uIce, vIce)
    cBotC = bottomdrag_coeffs(state, uIce, vIce)

    # over open ocean..., see comments in MITgcm: pkg/seaice/seaice_evp.F
    locMaskU = state.variables.SeaIceMassU
    locMaskV = state.variables.SeaIceMassV
    locMaskU = npx.where(locMaskU != 0, 1, locMaskU)
    locMaskV = npx.where(locMaskV != 0, 1, locMaskV)

    # calculate velocity of ice relative to ocean surface
    # and interpolate to c points
    duAtC = 0.5 * ( state.variables.uVel-uIce
            + npx.roll(state.variables.uVel-uIce,-1,1) )
    dvAtC = 0.5 * ( state.variables.vVel-vIce
            + npx.roll(state.variables.vVel-vIce,-1,0) )

    # calculate forcing from wind and ocean stress at velocity points
    ForcingX = state.variables.WindForcingX + (
        0.5 * ( cDrag + npx.roll(cDrag,1,1) ) * cosWat * state.variables.uVel
        - npx.sign(fCori) * sinWat * 0.5 * (
            cDrag * dvAtC + npx.roll(cDrag * dvAtC,1,1)
        ) * locMaskU
    ) * state.variables.AreaW
    ForcingY = state.variables.WindForcingY + (
        0.5 * ( cDrag + npx.roll(cDrag,1,0) ) * cosWat * state.variables.vVel
        + npx.sign(fCori) * sinWat * 0.5 * (
            cDrag * duAtC  + npx.roll(cDrag * duAtC,1,0)
        ) * locMaskV
    ) * state.variables.AreaS

    # add coriolis terms
    fvAtC = state.variables.SeaIceMassC * fCori * 0.5 \
            * ( vIce + npx.roll(vIce,-1,0) )
    fuAtC = state.variables.SeaIceMassC * fCori * 0.5 \
            * ( uIce + npx.roll(uIce,-1,1) )
    ForcingX = ForcingX + 0.5 * ( fvAtC + npx.roll(fvAtC,1,1) )
    ForcingY = ForcingY - 0.5 * ( fuAtC + npx.roll(fuAtC,1,0) )

    if useAdaptiveEVP:
        evpBetaU = 0.5 * ( evpAlphaC + npx.roll(evpAlphaC,1,1) )
        evpBetaV = 0.5 * ( evpAlphaC + npx.roll(evpAlphaC,1,0) )

    rMassU = 1./npx.where(state.variables.SeaIceMassU==0,npx.inf,state.variables.SeaIceMassU)
    rMassV = 1./npx.where(state.variables.SeaIceMassV==0,npx.inf,state.variables.SeaIceMassV)
    dragU = 0.5 * ( cDrag + npx.roll(cDrag,1,1) ) * cosWat * state.variables.AreaW \
          + 0.5 * ( cBotC + npx.roll(cBotC,1,1) )          * state.variables.AreaW
    dragV = 0.5 * ( cDrag + npx.roll(cDrag,1,0) ) * cosWat * state.variables.AreaS \
          + 0.5 * ( cBotC + npx.roll(cBotC,1,0) )          * state.variables.AreaS

    # step momentum equations with ice-ocean stress treated ...
    if explicitDrag:
        # ... explicitly
        ForcingX = ForcingX - uIce * dragU
        ForcingY = ForcingY - vIce * dragV
        denomU = 1.
        denomV = 1.
    else:
        # ... or implicitly
        denomU = 1. + dragU * state.settings.deltatDyn*rMassU/evpBetaU
        denomV = 1. + dragV * state.settings.deltatDyn*rMassV/evpBetaV

    # step momentum equations following Kimmritz et al. (2016)
    uIce = SeaIceMaskU * (
        uIce + (
            state.settings.deltatDyn*rMassU * ( ForcingX + stressDivX )
            + ( uIceNm1 - uIce )
        ) / evpBetaU
    ) / denomU
    vIce = SeaIceMaskV * (
        vIce + (
            state.settings.deltatDyn*rMassV * ( ForcingY + stressDivY )
            + ( vIceNm1 - vIce )
        ) / evpBetaV
    ) / denomV

    # uIce = fill_overlap(uIce)
    # vIce = fill_overlap(vIce)
    uIce, vIce = fill_overlap_uv(
                            uIce, vIce)

    # residual computation
    if computeEvpResidual:
        sig11Pm1 = (sigma11 - sig11Pm1) * evpAlphaC * iceMask
        sig22Pm1 = (sigma22 - sig22Pm1) * evpAlphaC * iceMask
        sig12Pm1 = (sigma12 - sig12Pm1) * evpAlphaZ #* maskZ

        uIcePm1 = SeaIceMaskU * ( uIce - uIcePm1 ) * evpBetaU
        vIcePm1 = SeaIceMaskV * ( vIce - vIcePm1 ) * evpBetaV

        # if not explicitDrag:
        #     ForcingX = ForcingX - uIce * dragU
        #     ForcingY = ForcingY - vIce * dragV

        # uIcePm1 = ( SeaIceMassU * (uIce - uIceNm1)*recip_deltatDyn
        #             - (ForcingX + stressDivX)
        #            ) * SeaIceMaskU
        # vIcePm1 = ( SeaIceMassV * (vIce - vIceNm1)*recip_deltatDyn
        #             - (ForcingY + stressDivY)
        #            ) * SeaIceMaskV

        resSig = update(resSig, at[iEVP], (sig11Pm1**2 + sig22Pm1**2
                    + sig12Pm1**2)[oly:-oly,olx:-olx].sum())
        resSig = update(resSig, at[iEVP], global_sum(resSig[iEVP]))
        resU = update(resU, at[iEVP], (uIcePm1**2
                    + vIcePm1**2 )[oly:-oly,olx:-olx].sum())
        resU = update(resU, at[iEVP], global_sum(resU[iEVP]))

        resEVP = resU[iEVP]
        resEVP0 = resU[0]
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

    return [state, uIce, vIce, uIceNm1, vIceNm1, sigma11, sigma22, sigma12, denom1, denom2,
            EVPcFac, evpAlphaC, evpAlphaZ, evpBetaU, evpBetaV, resSig, resU]


@veros_kernel
def evp_solver(state):

    if useAdaptiveEVP:
        aEVPcStar = 4
        EVPcFac = state.settings.deltatDyn * aEVPcStar * ( npx.pi * aEvpCoeff ) ** 2
    else:
        EVPcFac = 0

    denom1 = npx.ones_like(iceMask)*1 / evpAlpha
    denom2 = denom1

    # copy previous time step (n-1) of uIce, vIce
    uIceNm1 = state.variables.uIce
    vIceNm1 = state.variables.vIce

    uIce = state.variables.uIce
    vIce = state.variables.vIce
    sigma11 = state.variables.sigma11
    sigma22 = state.variables.sigma22
    sigma12 = state.variables.sigma12

    # initialize adaptive EVP specific fields
    evpAlphaC = npx.ones_like(iceMask)*evpAlpha
    evpAlphaZ = npx.ones_like(iceMask)*evpAlpha
    evpBetaU  = npx.ones_like(iceMask)*evpBeta
    evpBetaV  = npx.ones_like(iceMask)*evpBeta

    resSig  = npx.zeros(nEVPsteps+1) #???
    resU    = npx.zeros(nEVPsteps+1) #TODO why does this not lead to a division by zero?

    # to avoid dividing by zero in the first iteration
    resEVP0 = 1e-5 #TODO are these necessary
    resEVP = evpTol*2 #???
    # for iEVP in range (nEVPsteps):
    #     #print(iEVP)
    #     if iEVP ==1:
    #         uIce, vIce, sigma11, sigma22, sigma12, resEVP0 = evp_solver_body(
    #             state, uIce, vIce, uIceNm1, vIceNm1, sigma11, sigma22, sigma12, denom1, denom2,
    #             iEVP, EVPcFac, evpAlphaC, evpAlphaZ, evpBetaU, evpBetaV, resSig, resU, resEVP0)
    #     else:
    #         uIce, vIce, sigma11, sigma22, sigma12, _ = evp_solver_body(
    #             state, uIce, vIce, uIceNm1, vIceNm1, sigma11, sigma22, sigma12, denom1, denom2,
    #             iEVP, EVPcFac, evpAlphaC, evpAlphaZ, evpBetaU, evpBetaV, resSig, resU, resEVP0)

    arg_body = [state, uIce, vIce, uIceNm1, vIceNm1, sigma11, sigma22, sigma12, denom1, denom2,
                EVPcFac, evpAlphaC, evpAlphaZ, evpBetaU, evpBetaV, resSig, resU]


    arg_body = for_loop(0,400,evp_solver_body,arg_body)

    uIce = arg_body[1]
    vIce = arg_body[2]
    resSig = arg_body[15]
    resU = arg_body[16]




    if computeEvpResidual and plotEvpResidual:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(nrows=2,ncols=1,sharex=True)
        ax[0].semilogy(resU[:],'x-')
        ax[0].set_title('resU')
        ax[1].semilogy(resSig[:],'x-')
        ax[1].set_title('resSig')
        # s12 = sigma12 + npx.roll(sigma12,-1,0)
        # s12 = 0.25*(s12 + npx.roll(s12,-1,1))
        # s1=( sigma1 + npx.sqrt(sigma2**2 + 4*s12**2) )/press # I changed press -> 0.5 * press
        # s2=( sigma1 - npx.sqrt(sigma2**2 + 4*s12**2) )/press
        # csf0=ax[0].plot(s1.ravel(),s2.ravel(),'.');#plt.colorbar(csf0,ax=ax[0])
        # ax[0].plot([-1.4,0.1],[-1.4,0.1],'k-'); #plt.colorbar(csf0,ax=ax[0])
        # ax[0].set_title('sigma1')
        # csf1=ax[1].pcolormesh(sigma12); plt.colorbar(csf1,ax=ax[1])
        # ax[1].set_title('sigma2')
        plt.show()
        # print(resU)
        # print(resSig)


    return uIce, vIce