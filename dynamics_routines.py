from veros.core.operators import numpy as npx

from seaice_params import *
from seaice_size import *

secondOrderBC = False

def calc_ice_strength(hIceMean, Area):
    return SeaIceStrength * hIceMean * npx.exp(-cStar * (1 - Area)) * iceMask

def ocean_drag_coeffs(uIce, vIce, uVel, vVel):

    ### input:
    # uIce: zonal ice velocity
    # vIce: meridional ice velocity
    # uVel: zonal ocean velocity
    # vVel: meridional ice velocity

    ### output:
    # cDrag: linear drag coefficient between ice and ocean at c point
    # (this coefficient creates a linear relationship between
    # ice-ocean stress difference and ice-ocean velocity difference)


    # get ice-water drag coefficient times density
    dragCoeff = npx.where(fCori < 0, waterIceDrag_south, waterIceDrag) * rhoConst

    # calculate linear drag coefficient
    du = (uIce - uVel)*maskInW
    dv = (vIce - vVel)*maskInS
    tmpVar = 0.25 * ( du**2 + npx.roll(du,-1,1)**2
                    + dv**2 + npx.roll(dv,-1,0)**2 )

    cDrag = npx.where(dragCoeff**2 * tmpVar > cDragMin**2,
                        dragCoeff * npx.sqrt(tmpVar), cDragMin)
    cDrag = cDrag * iceMask


    return cDrag

def bottomdrag_coeffs(uIce, vIce, hIceMean, Area, R_low):

    ### input:
    # uIce: zonal ice velocity
    # vIce: meridional ice velocity
    # hIceMean: mean ice thickness
    # Area: ice cover fraction
    # R_low: water depth

    ### output:
    # cBot: non-linear drag coefficient for ice-bottom drag

    fac = 10. #scales the soft maximum for more accuracy
    recip_fac = 1. / fac

    tmpFld = npx.zeros_like(uIce)

    tmpFld = 0.25 * ( (uIce*maskInW)**2 + npx.roll(uIce*maskInW,-1,1)**2
                    + (vIce*maskInS)**2 + npx.roll(vIce*maskInS,-1,1)**2 )
    tmpFld = basalDragK2 / npx.sqrt(tmpFld + basalDragU0**2)

    hCrit = npx.abs(R_low) * Area / basalDragK1

    # we want to have some soft maximum for better differentiability:
    # max(a,b;k) = ln(exp(k*a)+exp(k*b))/k
    # In our case, b=0, so exp(k*b) = 1.
    # max(a,0;k) = ln(exp(k*a)+1)/k
    # If k*a gets too large, EXP will overflow, but for the anticipated
    # values of hActual < 100m, and k=10, this should be very unlikely
    cBot = npx.where(Area > 0.01,
            tmpFld
                * npx.log(npx.exp(fac * (hIceMean-hCrit)) + 1.) * recip_fac
                * npx.exp(-cBasalStar * (1. - Area)),
                0.)

    return cBot

def strainrates(uFld, vFld):

    ### input
    # uFld: zonal ice field velocity
    # vFld: meridional ice Field velocity
    # secondOrderBC: flag

    ### output
    # e11: 1,1 component of strain rate tensor
    # e22: 2,2 component of strain rate tensor
    # e12: 1,2 component of strain rate tensor

    # abbreviations at c points
    dudx = ( npx.roll(uFld,-1,axis=1) - uFld ) * recip_dxF
    uave = ( npx.roll(uFld,-1,axis=1) + uFld ) * 0.5
    dvdy = ( npx.roll(vFld,-1,axis=0) - vFld ) * recip_dyF
    vave = ( npx.roll(vFld,-1,axis=0) + vFld ) * 0.5

    # evaluate strain rates at c points
    e11 = (dudx + vave * k2AtC) * maskInC
    e22 = (dvdy + uave * k1AtC) * maskInC

    # abbreviations at z points
    dudy = ( uFld - npx.roll(uFld,1,axis=0) ) * recip_dyU
    uave = ( uFld + npx.roll(uFld,1,axis=0) ) * 0.5
    dvdx = ( vFld - npx.roll(vFld,1,axis=1) ) * recip_dxV
    vave = ( vFld + npx.roll(vFld,1,axis=1) ) * 0.5

    # evaluate strain rate at z points
    mskZ = iceMask*npx.roll(iceMask,1,axis=1)
    mskZ =    mskZ*npx.roll(   mskZ,1,axis=0)
    e12 = 0.5 * (dudy + dvdx - k1AtZ * vave - k2AtZ * uave ) * mskZ
    if noSlip:
        hFacU = SeaIceMaskU - npx.roll(SeaIceMaskU,1,axis=0)
        hFacV = SeaIceMaskV - npx.roll(SeaIceMaskV,1,axis=1)
        e12   = e12 + ( 2.0 * uave * recip_dyU * hFacU
                      + 2.0 * vave * recip_dxV * hFacV )

    if secondOrderBC:
        hFacU = ( SeaIceMaskU - npx.roll(SeaIceMaskU,1,0) ) / 3.
        hFacV = ( SeaIceMaskV - npx.roll(SeaIceMaskV,1,1) ) / 3.
        hFacU = hFacU * (npx.roll(SeaIceMaskU, 2,0) * npx.roll(SeaIceMaskU,1,0)
                       + npx.roll(SeaIceMaskU,-1,0) * SeaIceMaskU )
        hFacV = hFacV * (npx.roll(SeaIceMaskV, 2,1) * npx.roll(SeaIceMaskV,1,1)
                       + npx.roll(SeaIceMaskV,-1,1) * SeaIceMaskV )
        # right hand sided dv/dx = (9*v(i,j)-v(i+1,j))/(4*dxv(i,j)-dxv(i+1,j))
        # according to a Taylor expansion to 2nd order. We assume that dxv
        # varies very slowly, so that the denominator simplifies to 3*dxv(i,j),
        # then dv/dx = (6*v(i,j)+3*v(i,j)-v(i+1,j))/(3*dxv(i,j))
        #            = 2*v(i,j)/dxv(i,j) + (3*v(i,j)-v(i+1,j))/(3*dxv(i,j))
        # the left hand sided dv/dx is analogously
        #            = - 2*v(i-1,j)/dxv(i,j)-(3*v(i-1,j)-v(i-2,j))/(3*dxv(i,j))
        # the first term is the first order part, which is already added.
        # For e12 we only need 0.5 of this gradient and vave = is either
        # 0.5*v(i,j) or 0.5*v(i-1,j) near the boundary so that we need an
        # extra factor of 2. This explains the six. du/dy is analogous.
        # The masking is ugly, but hopefully effective.
        e12 = e12 + 0.5 * (
            recip_dyU * ( 6. * uave
                          - npx.roll(uFld, 2,0) * npx.roll(SeaIceMaskU,1,0)
                          - npx.roll(uFld,-1,0) * SeaIceMaskU ) * hFacU
          + recip_dxV * ( 6. * vave
                          - npx.roll(vFld, 2,1) * npx.roll(SeaIceMaskV,1,1)
                          - npx.roll(vFld,-1,1) * SeaIceMaskV ) * hFacV
        )

    return e11, e22, e12

def viscosities(e11,e22,e12,press0,iStep,myTime,myIter):

    """Usage: zeta, eta, press =
          viscosities(e11,e22,e12,press0,iStep,myTime,myIter)
    ### input
    # e11: 1,1 component of strain rate tensor
    # e22: 2,2 component of strain rate tensor
    # e12: 1,2 component of strain rate tensor
    # press0: maximum compressive stres

    ### ouput
    # zeta, eta : bulk and shear viscosity
    # press : replacement pressure
    """

    # from seaice_params import PlasDefCoeff, deltaMin, \
    #     tensileStrFac, pressReplFac

    recip_PlasDefCoeffSq = 1. / PlasDefCoeff**2

    # use area weighted average of squares of e12 (more accurate)
    e12Csq = rAz * e12**2
    e12Csq =                     e12Csq + npx.roll(e12Csq,-1,0)
    e12Csq = 0.25 * recip_rA * ( e12Csq + npx.roll(e12Csq,-1,1) )

    deltaSq = (e11+e22)**2 + recip_PlasDefCoeffSq * (
        (e11-e22)**2 + 4. * e12Csq )
    deltaC = npx.sqrt(deltaSq)

    # smooth regularization of delta for better differentiability
    # deltaCreg = deltaC + deltaMin
    # deltaCreg = npx.sqrt( deltaSq + deltaMin**2 )
    deltaCreg = npx.maximum(deltaC,deltaMin)

    zeta = 0.5 * (press0 * (1 + tensileStrFac)) / deltaCreg
    eta  = zeta * recip_PlasDefCoeffSq

    # recalculate pressure
    press = ( press0 * (1 - pressReplFac)
              + 2. * zeta * deltaC * pressReplFac / (1 + tensileStrFac)
             ) * (1 - tensileStrFac)

    return zeta, eta, press


def calc_stress(e11, e22, e12, zeta, eta, press, iStep, myTime, myIter):
    from seaice_averaging import c_point_to_z_point
    sig11 = zeta*(e11 + e22) + eta*(e11 - e22) - 0.5 * press
    sig22 = zeta*(e11 + e22) - eta*(e11 - e22) - 0.5 * press
    sig12 = 2. * e12 * c_point_to_z_point(eta)
    return sig11, sig22, sig12

def calc_stressdiv(sig11, sig22, sig12, iStep, myTime, myIter):
    stressDivX = (
          sig11*dyF - npx.roll(sig11*dyF, 1,axis=1)
        - sig12*dxV + npx.roll(sig12*dxV,-1,axis=0)
    ) * recip_rAw
    stressDivY = (
          sig22*dxF - npx.roll(sig22*dxF, 1,axis=0)
        - sig12*dyU + npx.roll(sig12*dyU,-1,axis=1)
    ) * recip_rAs
    return stressDivX, stressDivY

def calc_lhs(uIce, vIce, zeta, eta, press,
             hIceMean, Area, areaW, areaS,
             SeaIceMassC, SeaIceMassU, SeaIceMassV,
             cDrag, cBotC, R_low,
             iStep, myTime, myIter):

    recip_deltaT = 1./deltaTdyn
    bdfAlpha = 1.
    sinWat = npx.sin(npx.deg2rad(waterTurnAngle))
    cosWat = npx.cos(npx.deg2rad(waterTurnAngle))

    #
    e11, e22, e12          = strainrates(uIce, vIce)
    # zeta, eta, press       = viscosities(
    #     e11, e22, e12, press0, iStep, myIter, myTime)
    sig11, sig22, sig12    = calc_stress(
        e11, e22, e12, zeta, eta, press, iStep, myIter, myTime)
    stressDivX, stressDivY = calc_stressdiv(
        sig11, sig22, sig12, iStep, myIter, myTime)

    # sum up for symmetric drag contributions
    symDrag = cDrag*cosWat + cBotC

    # mass*(uIce)/deltaT - dsigma/dx
    uIceLHS = bdfAlpha*SeaIceMassU*recip_deltaT*uIce - stressDivX
    # mass*(vIce)/deltaT - dsigma/dy
    vIceLHS = bdfAlpha*SeaIceMassV*recip_deltaT*vIce - stressDivY
    # coriols terms: - mass*f*vIce
    #                + mass*f*uIce
    fuAtC = SeaIceMassC * fCori * 0.5 * ( uIce + npx.roll(uIce,-1,1) )
    fvAtC = SeaIceMassC * fCori * 0.5 * ( vIce + npx.roll(vIce,-1,0) )
    uIceLHS = uIceLHS - 0.5 * ( fvAtC + npx.roll(fvAtC,1,1) )
    vIceLHS = vIceLHS + 0.5 * ( fuAtC + npx.roll(fuAtC,1,0) )
    # ocean-ice and bottom drag terms: + (Cdrag*cosWat+Cb)*uIce - vIce*sinWat)
    uAtC = 0.5 * ( uIce + npx.roll(uIce,-1,1) )
    vAtC = 0.5 * ( vIce + npx.roll(vIce,-1,0) )
    uIceLHS = uIceLHS + (
        0.5 * ( symDrag + npx.roll(symDrag,1,1) ) *  uIce
        - npx.sign(fCori) * sinWat * 0.5 * (
            symDrag * vAtC + npx.roll(symDrag * vAtC,1,1)
        )
    ) * areaW
    vIceLHS = vIceLHS + (
        0.5 * ( symDrag + npx.roll(symDrag,1,0) ) * vIce
        + npx.sign(fCori) * sinWat * 0.5 * (
            symDrag * uAtC  + npx.roll(symDrag * uAtC,1,0)
        )
    ) * areaS
    # momentum advection (not now)

    # apply masks for interior (important when we have open boundaries)
    return uIceLHS*maskInW, vIceLHS*maskInS

def calc_rhs(uIceRHSfix, vIceRHSfix, areaW, areaS,
             uVel, vVel, cDrag,
             iStep, myTime, myIter):

    sinWat = npx.sin(npx.deg2rad(waterTurnAngle))
    cosWat = npx.cos(npx.deg2rad(waterTurnAngle))

    # ice-velocity independent contribution to drag terms
    # - Cdrag*(uVel*cosWat - vVel*sinWat)/(vVel*cosWat + uVel*sinWat)
    # ( remember to average to correct velocity points )
    uAtC = 0.5 * ( uVel + npx.roll(uVel,-1,1) )
    vAtC = 0.5 * ( vVel + npx.roll(vVel,-1,0) )
    uIceRHS = uIceRHSfix + (
        0.5 * ( cDrag + npx.roll(cDrag,1,1) ) * cosWat *  uVel
        - npx.sign(fCori) * sinWat * 0.5 * (
            cDrag * vAtC + npx.roll(cDrag * vAtC,1,1)
        )
    ) * areaW
    vIceRHS = vIceRHSfix + (
        0.5 * ( cDrag + npx.roll(cDrag,1,0) ) * cosWat * vVel
        + npx.sign(fCori) * sinWat * 0.5 * (
            cDrag * uAtC  + npx.roll(cDrag * uAtC,1,0)
        )
    ) * areaS

    return uIceRHS*maskInW, vIceRHS*maskInS

def calc_residual(uIce, vIce, hIceMean, hSnowMean, Area,
                  SeaIceMassC, SeaIceMassU, SeaIceMassV,
                  forcingU, forcingV, uVel, vVel, R_low,
                  iStep, myTime, myIter):

    # initialize fractional areas at velocity points
    areaW = 0.5 * (Area + npx.roll(Area,1,1))
    areaS = 0.5 * (Area + npx.roll(Area,1,0))

    # # set up mass per unit area
    # SeaIceMassC = rhoIce * hIceMean
    # # if SEAICEaddSnowMass (true)
    # SeaIceMassC= SeaIceMassC + rhoSnow * hSnowMean
    # SeaIceMassU = 0.5 * ( SeaIceMassC + npx.roll(SeaIceMassC,1,1) )
    # SeaIceMassV = 0.5 * ( SeaIceMassC + npx.roll(SeaIceMassC,1,0) )
    # calculate ice strength
    press0 = calc_ice_strength(hIceMean, Area)
    #
    cDrag = ocean_drag_coeffs(uIce, vIce, uVel, vVel)
    cBotC = bottomdrag_coeffs(uIce, vIce, hIceMean, Area, R_low)
    e11, e22, e12    = strainrates(uIce, vIce)
    zeta, eta, press = viscosities(
        e11, e22, e12, press0, iStep, myIter, myTime)

    Au, Av = calc_lhs(uIce, vIce, zeta, eta, press,
                      hIceMean, Area, areaW, areaS,
                      SeaIceMassC, SeaIceMassU, SeaIceMassV,
                      cDrag, cBotC, R_low,
                      iStep, myIter, myTime)
    bu, bv = calc_rhs(forcingU, forcingV, areaW, areaS,
                      uVel, vVel, cDrag,
                      iStep, myIter, myTime)

    # residual by vector component
    return Au-bu, Av-bv
