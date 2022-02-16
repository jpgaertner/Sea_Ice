import numpy as np

from seaice_params import *
from seaice_size import *

secondOrderBC = False

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
    dragCoeff = np.ones(uIce.shape) * waterIceDrag * rhoConst
    south = np.where(fCori < 0)
    dragCoeff[south] = waterIceDrag_south * rhoConst

    # calculate linear drag coefficient
    cDrag = np.ones(uIce.shape) * cDragMin
    du = (uIce - uVel)*maskInW
    dv = (vIce - vVel)*maskInS
    tmpVar = 0.25 * ( du**2 + np.roll(du,-1,1)**2
                    + dv**2 + np.roll(dv,-1,0)**2 )

    tmp = np.where(dragCoeff**2 * tmpVar > cDragMin**2)
    cDrag[tmp] = dragCoeff[tmp] * np.sqrt(tmpVar[tmp])
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

    tmpFld = np.zeros((sNy+2*OLy,sNx+2*OLx))
    isArea = np.where(Area[:-1,:-1] > 0.01)

    tmpFld = 0.25 * ( (uIce*maskInW)**2 + np.roll(uIce*maskInW,-1,1)**2
                    + (vIce*maskInS)**2 + np.roll(vIce*maskInS,-1,1)**2 )
    tmpFld = basalDragK2 / np.sqrt(tmpFld + basalDragU0**2)

    hCrit = np.abs(R_low) * Area / basalDragK1

    # we want to have some soft maximum for better differentiability:
    # max(a,b;k) = ln(exp(k*a)+exp(k*b))/k
    # In our case, b=0, so exp(k*b) = 1.
    # max(a,0;k) = ln(exp(k*a)+1)/k
    # If k*a gets too large, EXP will overflow, but for the anticipated
    # values of hActual < 100m, and k=10, this should be very unlikely
    cBot = np.where(Area > 0.01,
            tmpFld
                * np.log(np.exp(fac * (hIceMean-hCrit)) + 1.) * recip_fac
                * np.exp(-cBasalStar * (1. - Area)),
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
    dudx = ( np.roll(uFld,-1,axis=1) - uFld ) * recip_dxF
    uave = ( np.roll(uFld,-1,axis=1) + uFld ) * 0.5
    dvdy = ( np.roll(vFld,-1,axis=0) - vFld ) * recip_dyF
    vave = ( np.roll(vFld,-1,axis=0) + vFld ) * 0.5

    # evaluate strain rates at c points
    e11 = (dudx + vave * k2AtC) * maskInC
    e22 = (dvdy + uave * k1AtC) * maskInC

    # abbreviations at z points
    dudy = ( uFld - np.roll(uFld,1,axis=0) ) * recip_dyU
    uave = ( uFld + np.roll(uFld,1,axis=0) ) * 0.5
    dvdx = ( vFld - np.roll(vFld,1,axis=1) ) * recip_dxV
    vave = ( vFld + np.roll(vFld,1,axis=1) ) * 0.5

    # evaluate strain rate at z points
    mskZ = iceMask*np.roll(iceMask,1,axis=1)
    mskZ =    mskZ*np.roll(   mskZ,1,axis=0)
    e12 = 0.5 * (dudy + dvdx - k1AtZ * vave - k2AtZ * uave ) * mskZ
    if noSlip:
        hFacU = SeaIceMaskU - np.roll(SeaIceMaskU,1,axis=0)
        hFacV = SeaIceMaskV - np.roll(SeaIceMaskV,1,axis=1)
        e12   = e12 + ( 2.0 * uave * recip_dyU * hFacU
                      + 2.0 * vave * recip_dxV * hFacV )

    if secondOrderBC:
        hFacU = ( SeaIceMaskU - np.roll(SeaIceMaskU,1,0) ) / 3.
        hFacV = ( SeaIceMaskV - np.roll(SeaIceMaskV,1,1) ) / 3.
        hFacU = hFacU * (np.roll(SeaIceMaskU, 2,0) * np.roll(SeaIceMaskU,1,0)
                       + np.roll(SeaIceMaskU,-1,0) * SeaIceMaskU )
        hFacV = hFacV * (np.roll(SeaIceMaskV, 2,1) * np.roll(SeaIceMaskV,1,1)
                       + np.roll(SeaIceMaskV,-1,1) * SeaIceMaskV )
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
                          - np.roll(uFld, 2,0) * np.roll(SeaIceMaskU,1,0)
                          - np.roll(uFld,-1,0) * SeaIceMaskU ) * hFacU
          + recip_dxV * ( 6. * vave
                          - np.roll(vFld, 2,1) * np.roll(SeaIceMaskV,1,1)
                          - np.roll(vFld,-1,1) * SeaIceMaskV ) * hFacV
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
    e12Csq =                     e12Csq + np.roll(e12Csq,-1,0)
    e12Csq = 0.25 * recip_rA * ( e12Csq + np.roll(e12Csq,-1,1) )

    deltaSq = (e11+e22)**2 + recip_PlasDefCoeffSq * (
        (e11-e22)**2 + 4. * e12Csq )
    deltaC = np.sqrt(deltaSq)

    # smooth regularization of delta for better differentiability
    # deltaCreg = deltaC + deltaMin
    # deltaCreg = np.sqrt( deltaSq + deltaMin**2 )
    deltaCreg = np.maximum(deltaC,deltaMin)

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
          sig11*dyF - np.roll(sig11*dyF, 1,axis=1)
        - sig12*dxV + np.roll(sig12*dxV,-1,axis=0)
    ) * recip_rAw
    stressDivY = (
          sig22*dxF - np.roll(sig22*dxF, 1,axis=0)
        - sig12*dyU + np.roll(sig12*dyU,-1,axis=1)
    ) * recip_rAs
    return stressDivX, stressDivY

def calc_lhs(uIce, vIce, zeta, eta, press,
             hIceMean, Area, areaW, areaS,
             SeaIceMassC, SeaIceMassU, SeaIceMassV,
             cDrag, cBotC, R_low,
             iStep, myTime, myIter):

    recip_deltaT = 1./deltaTdyn
    bdfAlpha = 1.
    sinWat = np.sin(np.deg2rad(waterTurnAngle))
    cosWat = np.cos(np.deg2rad(waterTurnAngle))

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
    fuAtC = SeaIceMassC * fCori * 0.5 * ( uIce + np.roll(uIce,-1,1) )
    fvAtC = SeaIceMassC * fCori * 0.5 * ( vIce + np.roll(vIce,-1,0) )
    uIceLHS = uIceLHS - 0.5 * ( fvAtC + np.roll(fvAtC,1,1) )
    vIceLHS = vIceLHS + 0.5 * ( fuAtC + np.roll(fuAtC,1,0) )
    # ocean-ice and bottom drag terms: + (Cdrag*cosWat+Cb)*uIce - vIce*sinWat)
    uAtC = 0.5 * ( uIce + np.roll(uIce,-1,1) )
    vAtC = 0.5 * ( vIce + np.roll(vIce,-1,0) )
    uIceLHS = uIceLHS + (
        0.5 * ( symDrag + np.roll(symDrag,1,1) ) *  uIce
        - np.sign(fCori) * sinWat * 0.5 * (
            symDrag * vAtC + np.roll(symDrag * vAtC,1,1)
        )
    ) * areaW
    vIceLHS = vIceLHS + (
        0.5 * ( symDrag + np.roll(symDrag,1,0) ) * vIce
        + np.sign(fCori) * sinWat * 0.5 * (
            symDrag * uAtC  + np.roll(symDrag * uAtC,1,0)
        )
    ) * areaS
    # momentum advection (not now)

    # apply masks for interior (important when we have open boundaries)
    return uIceLHS*maskInW, vIceLHS*maskInS

def calc_rhs(uIceRHSfix, vIceRHSfix, areaW, areaS,
             uVel, vVel, cDrag,
             iStep, myTime, myIter):

    sinWat = np.sin(np.deg2rad(waterTurnAngle))
    cosWat = np.cos(np.deg2rad(waterTurnAngle))

    # ice-velocity independent contribution to drag terms
    # - Cdrag*(uVel*cosWat - vVel*sinWat)/(vVel*cosWat + uVel*sinWat)
    # ( remember to average to correct velocity points )
    uAtC = 0.5 * ( uVel + np.roll(uVel,-1,1) )
    vAtC = 0.5 * ( vVel + np.roll(vVel,-1,0) )
    uIceRHS = uIceRHSfix + (
        0.5 * ( cDrag + np.roll(cDrag,1,1) ) * cosWat *  uVel
        - np.sign(fCori) * sinWat * 0.5 * (
            cDrag * vAtC + np.roll(cDrag * vAtC,1,1)
        )
    ) * areaW
    vIceRHS = vIceRHSfix + (
        0.5 * ( cDrag + np.roll(cDrag,1,0) ) * cosWat * vVel
        + np.sign(fCori) * sinWat * 0.5 * (
            cDrag * uAtC  + np.roll(cDrag * uAtC,1,0)
        )
    ) * areaS

    return uIceRHS*maskInW, vIceRHS*maskInS

def calc_residual(uIce, vIce, hIceMean, Area,
                  zeta, eta, press, cDrag, cBotC,
                  SeaIceMassC, SeaIceMassU, SeaIceMassV,
                  forcingU, forcingV, uVel, vVel, R_low,
                  iStep, myTime, myIter):

    # initialize fractional areas at velocity points
    areaW = 0.5 * (Area + np.roll(Area,1,1))
    areaS = 0.5 * (Area + np.roll(Area,1,0))

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
