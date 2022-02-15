import numpy as np
# from numpy import linalg as npla
from scipy.sparse import linalg as spla
from scipy.sparse.linalg import LinearOperator
import scipy.sparse as sp
from scipy.linalg import solve
# from scipy.sparse import csr_matrix
# from scipy.sparse.linalg import spsolve_triangular
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

useLsrZebra = True
useStrengthImplicitCoupling = True
secondOrderBC = False
lsrRelax = 1.05
lsrRelaxU = lsrRelax
lsrRelaxV = lsrRelax
nonLinTol = 1e-5
linTol = 1e-7
nLsr = 100
nLin = 100

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
    # for the free slip case sig12 = 0
    hFacM = np.roll(SeaIceMaskU, 1,0)
    hFacP = np.roll(SeaIceMaskU,-1,0)

    # coefficients for uIce(i-1,j)
    AU = np.roll( - UXX + UXM,1,1 ) * SeaIceMaskU * recip_rAw
    # coefficients for uIce(i+1,j)
    CU =        ( - UXX - UXM     ) * SeaIceMaskU * recip_rAw
    # coefficients of uIce(i,j-1)
    uRt1 =        ( UYY + UYM )     * SeaIceMaskU * recip_rAw
    # coefficients of uIce(i,j+1)
    uRt2 = np.roll( UYY - UYM,-1,0) * SeaIceMaskU * recip_rAw
    # coefficients for uIce(i,j)
    BU = - AU - CU + (2.-hFacM)*uRt1 + (2.-hFacP)*uRt2
    # reset coefficients of uIce(i,j+/-1)
    # uRt1 = uRt1 * hFacM
    # uRt2 = uRt2 * hFacP

    # Coefficients for solving vIce :

    # apply boundary conditions according to slip factor
    # for no slip, set u on boundary to zero: v(i+/-1)=-v(i)
    # for the free slip case sig12 = 0
    hFacM = np.roll(SeaIceMaskV, 1,1)
    hFacP = np.roll(SeaIceMaskV,-1,1)

    # coefficients for vIce(i,j-1)
    AV = np.roll( - VYY + VYM,1,0 ) * SeaIceMaskV * recip_rAs
    # coefficients for vIce(i,j+1)
    CV =        ( - VYY - VYM     ) * SeaIceMaskV * recip_rAs
    # coefficients for vIce(i-1,j)
    vRt1 =        ( VXX + VXM )     * SeaIceMaskV * recip_rAs
    # coefficients for vIce(i+1,j)
    vRt2 = np.roll( VXX - VXM,-1,1) * SeaIceMaskV * recip_rAs
    # coefficients for vIce(i,j)
    BV = - AV - CV + (2.-hFacM)*vRt1 + (2.-hFacP)*vRt2
    # reset coefficients of vIce(i+/-1,j)
    # vRt1 = vRt1 * hFacM
    # vRt2 = vRt2 * hFacP

    # here we need add the contribution from the time derivative (in
    # bdfAlphaOverDt) and the symmetric drag term; must be done after
    # normalizing
    BU   = SeaIceMaskU * ( BU + bdfAlphaOverDt*seaiceMassU
        + 0.5 * ( dragSym + np.roll(dragSym,1,1) )*areaW
    )
    BV   = SeaIceMaskV * ( BV + bdfAlphaOverDt*seaiceMassV
        + 0.5 * ( dragSym + np.roll(dragSym,1,0) )*areaS
    )

    # this is clearly a hack: make sure that diagonals BU/BV are non-zero.
    # When scaling the surface ice-ocean stress by AREA, then there will
    # be many cases of zero diagonal entries.
    # BU[BU==0] = 1.
    # BV[BV==0] = 1.
    BU = np.where(BU==0, 1., BU)
    BV = np.where(BV==0, 1., BV)

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

    return AU, BU, CU, AV, BV, CV, uRt1, uRt2, vRt1, vRt2

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

def tridiag(a,b,c,d):
    m,n = d.shape
    w = np.zeros((m,n-1))#,dtype='float64')

    # begin
    w[:,0] = c[:,0]/b[:,0]
    d[:,0] = d[:,0]/b[:,0]

    # forward sweep
    for i in range(1,n-1):
        bet = 1./(b[:,i] - a[:,i]*w[:,i-1])
        w[:,i] = c[:,i]*bet
        d[:,i] = (d[:,i] - a[:,i]*d[:,i-1])*bet

    i = n-1
    d[:,i] = (d[:,i] - a[:,i]*d[:,i-1])/(b[:,i] - a[:,i]*w[:,i-1])
    # backward sweep
    for i in range(n-1,0,-1):
        d[:,i-1] = d[:,i-1] - w[:,i-1]*d[:,i]

    return d

def lsr_tridiagu(AU, BU, CU, uRt1, uRt2, rhsU, uIc):

    iMin = max(OLx-2,2)
    iMax = max(OLx + sNx, 2*OLx+sNx-2)
    iMxx = iMax-1
    # initialisation
    cuu = CU.copy()
    # zebra loop
    if useLsrZebra: ks = 2
    else:           ks = 1
    for k in range(OLy,OLy+ks):
        # boundary conditions
        uIc[k::ks,:]    = rhsU[k::ks,:] \
            + uRt1[k::ks,:]*np.roll(uIc, 1,0)[k::ks,:] \
            + uRt2[k::ks,:]*np.roll(uIc,-1,0)[k::ks,:]
        uIc[k::ks,iMin] = uIc[k::ks,iMin] - AU[k::ks,iMin]*uIc[k::ks,iMin-1]
        uIc[k::ks,iMxx] = uIc[k::ks,iMxx] - CU[k::ks,iMxx]*uIc[k::ks,iMxx+1]
        uIc[k::ks,:]    = uIc[k::ks,:] * SeaIceMaskU[k::ks,:]
        # b = uIc[k::ks,iMin:iMax]
        # uIc[k::ks,iMin:iMax]= tridiag(AU[k::ks,iMin:iMax],
        #                               BU[k::ks,iMin:iMax],
        #                               CU[k::ks,iMin:iMax],
        #                               b)
        # begin
        cuu[k::ks,iMin] = cuu[k::ks,iMin]/BU[k::ks,iMin]
        uIc[k::ks,iMin] = uIc[k::ks,iMin]/BU[k::ks,iMin]
        # forward sweep
        for i in range(iMin+1,iMax):
            bet = BU[k::ks,i]-AU[k::ks,i]*cuu[k::ks,i-1]
            cuu[k::ks,i] = cuu[k::ks,i]/bet
            uIc[k::ks,i] = (uIc[k::ks,i] - AU[k::ks,i]*uIc[k::ks,i-1])/bet

        # backward sweep
        for i in range(iMax-1,iMin,-1):
            uIc[k::ks,i-1]=uIc[k::ks,i-1]-cuu[k::ks,i-1]*uIc[k::ks,i]

    return uIc

def lsr_tridiagv(AV, BV, CV, vRt1, vRt2, rhsV, vIc):

    jMin = max(OLy-2,2)
    jMax = max(OLy + sNy, 2*OLy+sNy-2)
    jMxx = jMax-1
    # initialisation
    cvv = CV.copy()
    # zebra loop
    if useLsrZebra: ks = 2
    else:           ks = 1
    for k in range(OLx,OLx+ks):
        # boundary conditions
        vIc[:,k::ks]    = rhsV[:,k::ks] \
            + vRt1[:,k::ks]*np.roll(vIc, 1,1)[:,k::ks] \
            + vRt2[:,k::ks]*np.roll(vIc,-1,1)[:,k::ks]
        vIc[jMin,k::ks] = vIc[jMin,k::ks] - AV[jMin,k::ks]*vIc[jMin-1,k::ks]
        vIc[jMxx,k::ks] = vIc[jMxx,k::ks] - CV[jMxx,k::ks]*vIc[jMxx+1,k::ks]
        vIc[:,k::ks]    = vIc[:,k::ks] * SeaIceMaskV[:,k::ks]
        # b = vIc[jMin:jMax,k::ks]
        # vIc[jMin:jMax,k::ks] = tridiag(AV[jMin:jMax,k::ks].swapaxes(0,1),
        #                                BV[jMin:jMax,k::ks].swapaxes(0,1),
        #                                CV[jMin:jMax,k::ks].swapaxes(0,1),
        #                                b.swapaxes(0,1)).swapaxes(1,0)
        # begin
        cvv[jMin,k::ks] = cvv[jMin,k::ks]/BV[jMin,k::ks]
        vIc[jMin,k::ks] = vIc[jMin,k::ks]/BV[jMin,k::ks]
        # forward sweep
        for j in range(jMin+1,jMax):
            bet = BV[j,k::ks]-AV[j,k::ks]*cvv[j-1,k::ks]
            cvv[j,k::ks] = cvv[j,k::ks]/bet
            vIc[j,k::ks] = (vIc[j,k::ks] - AV[j,k::ks]*vIc[j-1,k::ks])/bet

        # backward sweep
        for j in range(jMax-1,jMin,-1):
            vIc[j-1,k::ks]=vIc[j-1,k::ks]-cvv[j-1,k::ks]*vIc[j,k::ks]

    return vIc

def lsr_solver(uIce, vIce, uVel, vVel, hIceMean, Area,
               press0, forcingU, forcingV,
               SeaIceMassC, SeaIceMassU, SeaIceMassV,
               R_low, nLsr = nLsr, nLin = nLin,
               useAsPreconditioner = False,
               zeta = None, eta = None, cDrag = None, cBotC = None,
               myTime = 0, myIter = 0):

    computeLsrResidual = True
    printLsrResidual   = True
    plotLsrResidual    = False
    if useAsPreconditioner:
        computeLsrResidual = False
        printLsrResidual   = False
        plotLsrResidual    = False

    recip_deltaT = 1./deltaTdyn
    bdfAlpha = 1.
    sinWat = np.sin(np.deg2rad(waterTurnAngle))
    cosWat = np.cos(np.deg2rad(waterTurnAngle))
    areaW = 0.5 * (Area + np.roll(Area,1,1))
    areaS = 0.5 * (Area + np.roll(Area,1,0))

    # copy previous time step (n-1) of uIce, vIce
    uIceNm1 = uIce.copy()
    vIceNm1 = vIce.copy()

    # this does not change in Picard iteration
    # mass*(uIceNm1)/deltaT
    uIceRHSfix = forcingU + SeaIceMassU*uIceNm1*recip_deltaT
    # mass*(vIceNm1)/deltaT
    vIceRHSfix = forcingV + SeaIceMassV*vIceNm1*recip_deltaT

    residual = np.array([None]*(nLsr+1))

    iLsr = -1
    resNonLin = nonLinTol*2
    while resNonLin > nonLinTol and iLsr < nLsr:
        iLsr = iLsr+1
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
        else:
            # This is the case for nLsr > 2, and here we use
            # a different iterative scheme. u/vIceC = u/vIce is unstable, and
            # different stabilisation methods are possible.

            # This is stable but slow to converge.
            # uIceC = 0.5*( uIce+uIceNm1 )
            # vIceC = 0.5*( vIce+vIceNm1 )
            # This converges slightly faster than the previous lines.
            wght = .5
            uIceC = wght*uIce+(1.-wght)*uIceC
            vIceC = wght*vIce+(1.-wght)*vIceC

        if not useAsPreconditioner:
            # drag coefficients for ice-ocean and basal drag
            cDrag = ocean_drag_coeffs(uIceC, vIceC, uVel, vVel)
            cBotC = bottomdrag_coeffs(uIceC, vIceC, hIceMean, Area, R_low)
            #
            e11, e22, e12    = strainrates(uIceC, vIceC, secondOrderBC)
            zeta, eta, press = viscosities(
                e11, e22, e12, press0, iLsr, myTime, myIter)
        else:
            press = press0.copy()

        AU, BU, CU, AV, BV, CV, uRt1, uRt2, vRt1, vRt2 = lsr_coefficents(
            zeta, eta, cDrag+cBotC, SeaIceMassU, SeaIceMassV,
            areaW, areaS,
            iLsr, myTime, myIter)

        if useAsPreconditioner:
            uIceRHS, vIceRHS = calc_rhs_lsr(
                uIce, vIce, areaW, areaS,
                uIceC, vIceC, uVel, vVel, cDrag, zeta, eta, press,
                SeaIceMassC, iLsr, myTime, myIter)
        else:
            uIceRHS, vIceRHS = calc_rhs_lsr(
                uIceRHSfix, vIceRHSfix, areaW, areaS,
                uIceC, vIceC, uVel, vVel, cDrag, zeta, eta, press,
                SeaIceMassC, iLsr, myTime, myIter)

        if computeLsrResidual:
            residUpre, residVpre, uRes, vRes = lsr_residual(
                uIceRHS, vIceRHS, uRt1, uRt2, vRt1, vRt2,
                AU, BU, CU, AV, BV, CV, uIce, vIce,
                True, myTime, myIter )
            if printLsrResidual:
                print ( 'pre  lin: %i      %e %e'%(
                    iLsr, residUpre, residVpre) )

        iLin = 0.
        doIterU = True
        doIterV = True
        while iLin < nLin and (doIterV or doIterV):
            iLin = iLin+1

            if doIterU: uNm1 = uIce.copy()
            if doIterV: vNm1 = vIce.copy()
            isEven = np.mod(iLin,2)==0 and False
            if not isEven:
                if doIterU:
                    uTmp = lsr_tridiagu( AU, BU, CU, uRt1, uRt2, uIceRHS,
                                         uIce )
                if doIterV:
                    vTmp = lsr_tridiagv( AV, BV, CV, vRt1, vRt2, vIceRHS,
                                         vIce )
                    # vTmp = lsr_tridiagu( AV.transpose(),
                    #                      BV.transpose(),
                    #                      CV.transpose(),
                    #                      vRt1.transpose(),
                    #                      vRt2.transpose(),
                    #                      vIceRHS.transpose(),
                    #                      vIce.transpose() ).transpose()
            if isEven:
                if doIterU:
                    uTmp = lsr_tridiagu( AU, BU, CU, uRt1, uRt2, uIceRHS,
                                         uIce )

            # over relaxation step
            # lsrRelaxU = 1.05
            # lsrRelaxV = 1.05
            if doIterU: uIce = uIce + lsrRelaxU * ( uTmp - uIce )
            if doIterV: vIce = vIce + lsrRelaxV * ( vTmp - vIce )

            uIce, vIce = fill_overlap_uv( uIce, vIce )

            if iLin>1:
                doIterU = np.sqrt((uIce-uNm1)**2
                                  )[OLy:-OLy,OLx:-OLx].max() > linTol
                doIterV = np.sqrt((vIce-vNm1)**2
                                  )[OLy:-OLy,OLx:-OLx].max() > linTol

            # resU = np.sqrt((uIce-uNm1)**2).max()
            # resV = np.sqrt((vIce-vNm1)**2).max()
            # print(resU,resV,doIterU,doIterV)

            # residU, residV, uRes, vRes = lsr_residual(
            #     uIceRHS, vIceRHS, uRt1, uRt2, vRt1, vRt2,
            #     AU, BU, CU, AV, BV, CV, uIce, vIce,
            #     True, myTime, myIter )
            # print ( 'in lin: %i %i %e %e'%(iLsr, iLin, residU, residV ) )
            # doIterU = residU > linTol
            # doIterV = residV > linTol

        if computeLsrResidual and (doIterU or doIterV):
            residUpost, residVpost, uRes, vRes = lsr_residual(
                uIceRHS, vIceRHS, uRt1, uRt2, vRt1, vRt2,
                AU, BU, CU, AV, BV, CV, uIce, vIce,
                True, myTime, myIter )
            if printLsrResidual:
                print ( 'post lin: %i %4i %e %e'%(
                    iLsr, iLin, residUpost, residVpost ) )


        resNonLin = np.sqrt(residUpre**2 + residVpre**2)
        residual[iLsr] = resNonLin
        if iLsr==0: resNonLin0 = resNonLin

        resNonLin = resNonLin/resNonLin0

    if plotLsrResidual:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(nrows=2,ncols=1,sharex=True)
        ax[0].semilogy(residual[:iLsr-1]/residual[0],'x-')
        ax[0].set_title('residual')
        plt.show()


    return uIce, vIce

        # n = iMax-iMin
        # for j in range(OLy,sNy+OLy):
        #     b = uIc[j,iMin:iMax]
        #     A = sp.spdiags([AU[j,iMin:iMax],
        #                     BU[j,iMin:iMax],
        #                     CU[j,iMin:iMax]],(-1,0,1), n, n).toarray()
        #     uIc[j,iMin:iMax] = solve(A,b,overwrite_b=True)

        # n = jMax-jMin
        # for i in range(OLx,sNx+OLx):
            # b = vIc[jMin:jMax,i]
            # A = sp.spdiags([AV[jMin:jMax,i],
            #                 BV[jMin:jMax,i],
            #                 CV[jMin:jMax,i]],(-1,0,1), n, n).toarray()
            # vIc[jMin:jMax,i] = solve(A,b,overwrite_b=True)
# def TDMA(a,b,c,d):
#     n = len(d)
#     w= np.zeros(n-1,float)
#     g= np.zeros(n, float)
#     p = np.zeros(n,float)

#     w[0] = c[0]/b[0]
#     g[0] = d[0]/b[0]

#     for i in range(1,n-1):
#         w[i] = c[i]/(b[i] - a[i-1]*w[i-1])
#     for i in range(1,n):
#         g[i] = (d[i] - a[i-1]*g[i-1])/(b[i] - a[i-1]*w[i-1])
#     p[n-1] = g[n-1]
#     for i in range(n-1,0,-1):
#         p[i-1] = g[i-1] - w[i-1]*p[i]
#     return p
