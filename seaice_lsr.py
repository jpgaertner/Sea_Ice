import numpy as np
# from numpy import linalg as npla
from scipy.sparse import linalg as spla
from scipy.sparse.linalg import LinearOperator
# from scipy.optimize import fsolve

from seaice_params import *
from seaice_size import *

from seaice_strainrates import strainrates
from seaice_viscosities import viscosities
from seaice_ocean_drag_coeffs import ocean_drag_coeffs
from seaice_bottomdrag_coeffs import bottomdrag_coeffs
from seaice_global_sum import global_sum
from seaice_fill_overlap import fill_overlap_uv
from seaice_averaging import c_point_to_z_point

useStrengthImplicitCoupling = True
secondOrderBC = False

def calc_rhs_lsr(uIceRHSfix, vIceRHSfix, areaW, areaS,
                 uIce, vIce, uVel, vVel, cDrag, zeta, eta, press,
                 SeaIceMassC,
                 iStep, myIter, myTime):

    sinWat = np.sin(np.deg2rad(waterTurnAngle))
    cosWat = np.cos(np.deg2rad(waterTurnAngle))

    duAtC = 0.5 * ( uVel-uIce + np.roll(uVel-uIce,-1,1) )
    dvAtC = 0.5 * ( vVel-vIce + np.roll(vVel-vIce,-1,0) )
    uIceRHS = uIceRHSfix + (
        0.5 * ( cDrag + np.roll(cDrag,1,1) ) * cosWat *  uVel
        - np.sign(fCori) * sinWat * 0.5 * (
            cDrag * dvAtC + np.roll(cDrag * dvAtC,1,1)
        )
    ) * areaW
    vIceRHS = vIceRHSfix + (
        0.5 * ( cDrag + np.roll(cDrag,1,0) ) * cosWat * vVel
        + np.sign(fCori) * sinWat * 0.5 * (
            cDrag * duAtC  + np.roll(cDrag * duAtC,1,0)
        )
    ) * areaS
    # coriols terms: + mass*f*vIce
    #                - mass*f*uIce
    fuAtC = SeaIceMassC * fCori * 0.5 * ( uIce + np.roll(uIce,-1,1) )
    fvAtC = SeaIceMassC * fCori * 0.5 * ( vIce + np.roll(vIce,-1,0) )
    uIceRHS = uIceRHS + 0.5 * ( fvAtC + np.roll(fvAtC,1,1) )
    vIceRHS = vIceRHS - 0.5 * ( fuAtC + np.roll(fuAtC,1,0) )

    # messy explicit parts of div(sigma)
    mskZ  = iceMask*np.roll(iceMask,1,axis=1)
    mskZ  =    mskZ*np.roll(   mskZ,1,axis=0)

    # u - component
    sig11 = (zeta-eta) *       (np.roll(vIce,-1,0)-vIce)*recip_dyF \
          + (zeta+eta) * 0.5 * (np.roll(vIce,-1,0)+vIce)*k2AtC \
          - 0.5 * press
    hFacM = SeaIceMaskV - np.roll(SeaIceMaskV,1,1)
    sig12 =      c_point_to_z_point(eta, noSlip) * (
        (
            ( vIce - np.roll(vIce,1,1) ) * recip_dxV
          - ( vIce + np.roll(vIce,1,1) ) * 0.5 * k1AtZ
        ) * mskZ
        +   ( vIce + np.roll(vIce,1,1) ) * hFacM * 2.0
    )

    if useStrengthImplicitCoupling:
        sig12 = sig12 + c_point_to_z_point(zeta, noSlip) * (
        (
            ( uIce - np.roll(uIce,1,0) ) * recip_dyU
        ) * mskZ
        +   ( uIce + np.roll(uIce,1,0) ) * hFacM * 2.0
    )

    uIceRHS = uIceRHS + (
          sig11*dyF - np.roll(sig11*dyF, 1,axis=1)
        - sig12*dxV + np.roll(sig12*dxV,-1,axis=0)
    ) * recip_rAw

    # v - component
    sig22 = (zeta-eta) *       (np.roll(uIce,-1,1)-uIce)*recip_dxF \
          + (zeta+eta) * 0.5 * (np.roll(uIce,-1,1)+uIce)*k1AtC \
          - 0.5 * press
    hFacM = SeaIceMaskU - np.roll(SeaIceMaskU,1,0)
    sig12 =      c_point_to_z_point(eta, noSlip) * (
        (
            ( uIce - np.roll(uIce,1,0) ) * recip_dyU
          - ( uIce + np.roll(uIce,1,0) ) * 0.5 * k2AtZ
        ) * mskZ
        +   ( uIce + np.roll(uIce,1,0) ) * hFacM * 2.0
    )

    if useStrengthImplicitCoupling:
        sig12 = sig12 + c_point_to_z_point(zeta, noSlip) * (
        (
            ( vIce - np.roll(vIce,1,1) ) * recip_dxV
        ) * mskZ
        +   ( vIce + np.roll(vIce,1,1) ) * hFacM * 2.0
    )

    vIceRHS = vIceRHS + (
          sig22*dxF - np.roll(sig22*dxF, 1,axis=0)
        - sig12*dyU + np.roll(sig12*dyU,-1,axis=1)
    ) * recip_rAs

    return uIceRHS*maskInW, vIceRHS*maskInS

def lsr_coefficents(zeta, eta, dragSym, seaiceMassU, seaiceMassV,
                    areaW, areaS, iStep, myTime, myIter):

    if useStrengthImplicitCoupling:
        strImpCplFac = 1.
    else:
        strImpCplFac = 0.

    # needs to be sorted out, can be 1.5/deltaTdyn
    bdfAlphaOverDt = 1./deltaTdyn

    # coefficients of uIce(i,j) and vIce(i,j) belonging to ...
    # ... d/dx (eta+zeta) d/dx u
    UXX = dyF * ( zeta + eta ) * recip_dxF
    # ... d/dy eta+zeta dv/dy
    VYY = dxF * ( zeta + eta ) * recip_dyF
    # ... d/dy eta d/dy u
    UYY = dxV * c_point_to_z_point(eta+strImpCplFac*zeta, noSlip) * recip_dyU
    # ... d/dx eta dv/dx
    VXX = dyU * c_point_to_z_point(eta+strImpCplFac*zeta, noSlip) * recip_dxV
    # ... d/dy eta k2 u
    UYM = dxV * c_point_to_z_point( eta, noSlip ) * k2AtZ * 0.5
    # ... d/dx eta k1 v
    VXM = dyU * c_point_to_z_point( eta, noSlip ) * k1AtZ * 0.5
    # ... d/dx (zeta-eta) k1 u
    UXM = dyF * ( zeta - eta ) * k1AtC * 0.5
    # ... d/dy (zeta-eta) k2 v
    VYM = dxF * ( zeta - eta ) * k2AtC * 0.5

    # assemble coefficient matrix, beware of sign convention: because
    # this is the left hand side we calculate -grad(sigma), but the
    # coefficients u/vRt1/2 of u/vIce(i+/-1,j) are counted on the
    # right hand side

    # Coefficients for solving uIce :

    # apply boundary conditions according to slip factor
    # for no slip, set u on boundary to zero: u(j+/-1)=-u(j)
    # for the free slip case sigma_12 = 0
    hFacM = np.roll(SeaIceMaskU, 1,0)
    hFacP = np.roll(SeaIceMaskU,-1,0)

    # coefficients for uIce(i-1,j)
    AU = np.roll( - UXX + UXM,1,1 ) * SeaIceMaskU * recip_rAw
    # coefficients for uIce(i+1,j)
    CU =        ( - UXX - UXM     ) * SeaIceMaskU * recip_rAw
    # # coefficients of uIce(i,j-1)
    # uRt1 =        ( UYY + UYM )     * SeaIceMaskU * recip_rAw * hFacM
    # # coefficients of uIce(i,j+1)
    # uRt2 = np.roll( UYY - UYM,-1,0) * SeaIceMaskU * recip_rAw * hFacP
    # # coefficients for uIce(i,j)
    # BU = - AU - CU + uRt1 + uRt2
    # coefficients of uIce(i,j-1)
    uRt1 =        ( UYY + UYM )     * SeaIceMaskU * recip_rAw
    # coefficients of uIce(i,j+1)
    uRt2 = np.roll( UYY - UYM,-1,0) * SeaIceMaskU * recip_rAw
    # coefficients for uIce(i,j)
    BU = - AU - CU + ( (2.-hFacM)*uRt1 + (2.-hFacP)*uRt2 ) * SeaIceMaskU
    # rest coefficients of uIce(i,j+/-1)
    uRt1 = uRt1 * hFacM
    uRt2 = uRt2 * hFacP

    # Coefficients for solving vIce :

    # apply boundary conditions according to slip factor
    # for no slip, set u on boundary to zero: v(i+/-1)=-v(i)
    # for the free slip case sigma_12 = 0
    hFacM = np.roll(SeaIceMaskV, 1,1)
    hFacP = np.roll(SeaIceMaskV,-1,1)

    # coefficients for vIce(i,j-1)
    AV = np.roll( - VYY + VYM,1,0 ) * SeaIceMaskV * recip_rAs
    # coefficients for vIce(i,j+1)
    CV =        ( - VYY - VYM     ) * SeaIceMaskV * recip_rAs
    # # coefficients for vIce(i-1,j)
    # vRt1 =        ( VXX + VXM )     * SeaIceMaskV * recip_rAs * hFacM
    # # coefficients for vIce(i+1,j)
    # vRt2 = np.roll( VXX - VXM,-1,1) * SeaIceMaskV * recip_rAs * hFacP
    # # coefficients for vIce(i,j)
    # BV = - AV - CV + vRt1 + vRt2
    # coefficients for vIce(i-1,j)
    vRt1 =        ( VXX + VXM )     * SeaIceMaskV * recip_rAs
    # coefficients for vIce(i+1,j)
    vRt2 = np.roll( VXX - VXM,-1,1) * SeaIceMaskV * recip_rAs
    # coefficients for vIce(i,j)
    BV = - AV - CV + ( (2.-hFacM)*vRt1 + (2.-hFacP)*vRt2 ) * SeaIceMaskV
    # rest coefficients of vIce(i+/-1,j)
    vRt1 = vRt1 * hFacM
    vRt2 = vRt2 * hFacP

    # here we need add the contribution from the time derivative (in
    # bdfAlphaOverDt) and the symmetric drag term; must be done after
    # normalizing
    BU   = BU + SeaIceMaskU * ( bdfAlphaOverDt*seaiceMassU
        + 0.5 * ( dragSym + np.roll(dragSym,1,1) )*areaW
    )
    BV   = BV + SeaIceMaskV * ( bdfAlphaOverDt*seaiceMassV
        + 0.5 * ( dragSym + np.roll(dragSym,1,0) )*areaS
    )

    # this is clearly a hack: make sure that diagonals BU/BV are non-zero.
    # When scaling the surface ice-ocean stress by AREA, then there will
    # be many cases of zero diagonal entries.
    BU[BU==0] = 1.
    BV[BV==0] = 1.

    # # not really necessary as long as we do not have open boundaries
    # BU[         maskInC*np.roll(maskInC,1,1)==0]=1.
    # AU   = AU  *maskInC*np.roll(maskInC,1,1)
    # CU   = CU  *maskInC*np.roll(maskInC,1,1)
    # uRt1 = uRt1*maskInC*np.roll(maskInC,1,1)
    # uRt2 = uRt2*maskInC*np.roll(maskInC,1,1)

    # BV[         maskInC*np.roll(maskInC,1,0)==0]=1.
    # AV   = AV  *maskInC*np.roll(maskInC,1,0)
    # CV   = CV  *maskInC*np.roll(maskInC,1,0)
    # vRt1 = vRt1*maskInC*np.roll(maskInC,1,0)
    # vRt2 = vRt2*maskInC*np.roll(maskInC,1,0)

    return AU, BU, CU, AV, BV, CV, uRt1, uRt2, vRt1, vRt1

def lsr_residual( rhsU, rhsV, uRt1, uRt2, vRt1, vRt2,
                  AU, BU, CU, AV, BV, CV, uFld, vFld,
                  calcMeanResid, myTime, myIter ):

    uRes = rhsU + (
        + uRt1*np.roll(uFld,1,0) + uRt2*np.roll(uFld,-1,0)
        - ( AU*np.roll(uFld,1,1) + BU*uFld + CU*np.roll(uFld,-1,1) )
    )
    vRes = rhsV + (
        + vRt1*np.roll(vFld,1,1) + vRt2*np.roll(vFld,-1,1)
        - ( AV*np.roll(vFld,1,0) + BV*vFld + CV*np.roll(vFld,-1,0) )
    )
    if calcMeanResid:
        residU = (uRes*uRes*rAw*maskInW
                  *maskInC*np.roll(maskInC,1,1))[OLy:-OLy,OLx:-OLx].sum()
        residV = (vRes*vRes*rAs*maskInS
                  *maskInC*np.roll(maskInC,1,0))[OLy:-OLy,OLx:-OLx].sum()
        residU = global_sum( residU )
        residV = global_sum( residV )
        # scale residuals by globalArea so that they do not get
        # ridiculously large
        if residU>0. : residU = np.sqrt(residU/globalArea)
        if residV>0. : residV = np.sqrt(residV/globalArea)
    else:
        residU = 0.
        residV = 0.

    return residU, residV, uRes, vRes

def lsr_tridiagu(AU, BU, CU, uRt1, uRt2, rhsU, uIce):

    iMin =  OLx
    iMax =  OLx + sNx
    cuu  = CU.copy()
    uRt  = np.zeros(uIce.shape)
    a3   = np.zeros(uIce.shape)
    a3[:,iMin] = - AU[:,iMin]*uIce[:,iMin-1]
    a3[:,iMax] = - CU[:,iMax]*uIce[:,iMax+1]
    uRt = ( rhsU + a3 + uRt1*np.roll(uIce,1,0) + uRt2*np.roll(uIce,-1,0)
           ) * SeaIceMaskU
    # zebra loop
    for k in range(2):
        # begin
        cuu[k::2,iMin] = cuu[k::2,iMin]/BU[k::2,iMin]
        uRt[k::2,iMin] = uRt[k::2,iMin]/BU[k::2,iMin]
        # forward sweep
        for i in range(iMin+1,iMax):
            bet = BU[k::2,i]-AU[k::2,i]*cuu[k::2,i-1]
            cuu[k::2,i] = cuu[k::2,i]/bet
            uRt[k::2,i] = (uRt[k::2,i]-AU[k::2,i]*uRt[k::2,i-1])/bet

        # backward sweep
        for i in range(iMin,iMax-1,-1):
            uRt[k::2,i]=uRt[k::2,i]-cuu[k::2,i]*uRt[k::2,i+1]

    return uRt

def lsr_tridiagv(AV, BV, CV, vRt1, vRt2, rhsV, vIce):

    jMin =  OLy
    jMax =  OLy + sNy
    cvv  = CV.copy()
    vRt  = np.zeros(vIce.shape)
    a3   = np.zeros(vIce.shape)
    a3[jMin,:] = - AV[jMin,:]*vIce[jMin-1,:]
    a3[jMax,:] = - CV[jMax,:]*vIce[jMax+1,:]
    vRt = ( rhsV + a3 + vRt1*np.roll(vIce,1,1) + vRt2*np.roll(vIce,-1,1)
           ) * SeaIceMaskV
    # zebra loop
    for k in range(2):
        # begin
        cvv[jMin,k::2] = cvv[jMin,k::2]/BV[jMin,k::2]
        vRt[jMin,k::2] = vRt[jMin,k::2]/BV[jMin,k::2]
        # forward sweep
        for j in range(jMin+1,jMax):
            bet = BV[j,k::2]-AV[j,k::2]*cvv[j-1,k::2]
            cvv[j,k::2] = cvv[j,k::2]/bet
            vRt[j,k::2] = (vRt[j,k::2]-AV[j,k::2]*vRt[j-1,k::2])/bet

        # backward sweep
        for j in range(jMin,jMax-1,-1):
            vRt[j,k::2]=vRt[j,k::2]-cvv[j,k::2]*vRt[j+1,k::2]

    return vRt

def lsr_solver(uIce, vIce, uVel, vVel, hIceMean, Area,
               press0, forcingU, forcingV,
               SeaIceMassC, SeaIceMassU, SeaIceMassV,
               R_low, myTime, myIter):

    computeLsrResidual = True

    recip_deltaT = 1./deltaTdyn
    bdfAlpha = 1.
    sinWat = np.sin(np.deg2rad(waterTurnAngle))
    cosWat = np.cos(np.deg2rad(waterTurnAngle))
    areaW = 0.5 * (Area + np.roll(Area,1,1))
    areaS = 0.5 * (Area + np.roll(Area,1,0))

    # copy previous time step (n-1) of uIce, vIce
    uIceNm1 = uIce.copy()
    vIceNm1 = vIce.copy()
    uIceC   = uIce.copy()
    vIceC   = vIce.copy()

    # this does not change in Picard iteration
    # mass*(uIceNm1)/deltaT
    uIceRHSfix = forcingU + SeaIceMassU*uIceNm1*recip_deltaT
    # mass*(vIceNm1)/deltaT
    vIceRHSfix = forcingV + SeaIceMassV*vIceNm1*recip_deltaT

    linTol = 1e-12
    lsrRelax = 1.05
    lsrRelaxU = lsrRelax
    lsrRelaxV = lsrRelax
    nLsr = 10
    nLin = 100
    computeLsrResidual = True
    residual = np.array([None]*nLsr)

    for iLsr in range(nLsr):
        if iLsr==0:
            # This is the predictor time step
            uIceC = uIce.copy()
            vIceC = vIce.copy()
        elif iLsr==1 and nLsr <= 2:
            # This is the modified Euler step
            uIce  = 0.5*( uIce+uIceNm1 )
            vIce  = 0.5*( vIce+vIceNm1 )
            uIceC = uIce.copy()
            vIceC = vIce.copy()
            pass
        else:
            # This is the case for nLsr > 2, and here we use
            # a different iterative scheme. u/vIceC = u/vIce is unstable, and
            # different stabilisation methods are possible.

            # This is stable but slow to converge.
            # uIceC = 0.5*( uIce+uIceNm1 )
            # vIceC = 0.5*( vIce+vIceNm1 )
            # This converges slightly faster than the previous lines.
            uIceC = 0.5*( uIce+uIceC )
            vIceC = 0.5*( vIce+vIceC )

        cDrag = ocean_drag_coeffs(uIceC, vIceC, uVel, vVel)
        cBotC = 0.*bottomdrag_coeffs(uIceC, vIceC, hIceMean, Area, R_low)
        e11, e22, e12    = strainrates(uIceC, vIceC, secondOrderBC)
        zeta, eta, press = viscosities(
            e11, e22, e12, press0, iLsr, myTime, myIter)

        uIceRHS, vIceRHS = calc_rhs_lsr(
            uIceRHSfix, vIceRHSfix, areaW, areaS,
            uIceC, vIceC, uVel, vVel, cDrag, zeta, eta, press,
            SeaIceMassC, iLsr, myTime, myIter)

        AU,BU,CU,AV,BV,CV,uRt1,uRt2,vRt1,vRt2 = lsr_coefficents(
            zeta, eta, cDrag+cBotC, SeaIceMassU, SeaIceMassV,
            areaW, areaS,
            iLsr, myTime, myIter)

        if computeLsrResidual:
            residUpre, residVpre, uRes, vRes = lsr_residual(
                uIceRHS, vIceRHS, uRt1, uRt2, vRt1, vRt2,
                AU, BU, CU, AV, BV, CV, uIce, vIce,
                True, myTime, myIter )
            print ( 'pre  lin: %i    %e %e'%(iLsr, residUpre, residVpre) )

        iLin = 0.
        doIterU = True
        doIterV = True
        while iLin < nLin and (doIterV or doIterV):
            iLin = iLin+1

            if doIterU:
                uNm1 = uIce.copy()
                uTmp = lsr_tridiagu( AU, BU, CU, uRt1, uRt2, uIceRHS,
                                     uIce )
            if doIterV:
                vNm1 = vIce.copy()
                vTmp = lsr_tridiagv( AV, BV, CV, vRt1, vRt2, vIceRHS,
                                     vIce )

            # over relaxation step
            lsrRelaxU = 1.05
            lsrRelaxV = 1.05
            if doIterU: uIce = uIce + lsrRelaxU * ( uTmp - uIce )
            if doIterV: vIce = vIce + lsrRelaxV * ( vTmp - vIce )

            uIce, vIce = fill_overlap_uv( uIce, vIce )

            if iLin>1:
                doIterU = np.sqrt((uIce-uNm1)**2).max() > linTol
                doIterV = np.sqrt((uIce-uNm1)**2).max() > linTol

            # resU = np.sqrt((uIce-uNm1)**2).max()
            # resV = np.sqrt((uIce-uNm1)**2).max()
            # print(resU,resV,doIterU,doIterV)

            # residU, residV, uRes, vRes = lsr_residual(
            #     uIceRHS, vIceRHS, uRt1, uRt2, vRt1, vRt2,
            #     AU, BU, CU, AV, BV, CV, uIce, vIce,
            #     True, myTime, myIter )
            # print ( 'in lin: %i %i %e %e'%(iLsr, iLin, residU, residV ) )
            # doIterU = residU > linTol
            # doIterV = residV > linTol

        if computeLsrResidual:
            residUpost, residVpost, uRes, vRes = lsr_residual(
                uIceRHS, vIceRHS, uRt1, uRt2, vRt1, vRt2,
                AU, BU, CU, AV, BV, CV, uIce, vIce,
                True, myTime, myIter )
            print ( 'post lin: %i %i %e %e'%(
                iLsr, iLin, residUpost, residVpost ) )

        residual[iLsr] = np.sqrt(residUpre**2 + residVpre**2)
    if computeLsrResidual:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(nrows=2,ncols=1,sharex=True)
        ax[0].semilogy(residual[:],'x-')
        ax[0].set_title('residual')
        plt.show()


    return uIce, vIce
