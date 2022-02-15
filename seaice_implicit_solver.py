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
from seaice_lsr import lsr_solver

secondOrderBC = False
nLinear = 50
nonLinTol = 1e-5

nPicard = 10
computePicardResidual = True
printPicardResidual = True
plotPicardResidual = True

nJFNK = 10
computeJFNKResidual = True
printJFNKResidual = True
plotJFNKResidual = True


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
    e11, e22, e12          = strainrates(uIce, vIce, secondOrderBC)
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

def _vecTo2d(vec):
    n = vec.shape[0]//2
    u = np.zeros((sNy+2*OLy,sNx+2*OLx))
    v = np.zeros((sNy+2*OLy,sNx+2*OLx))
    u[OLy:-OLy,OLx:-OLx]=vec[:n].reshape((sNy,sNx))
    v[OLy:-OLy,OLx:-OLx]=vec[n:].reshape((sNy,sNx))
    u,v=fill_overlap_uv(u,v)
    return u,v

def _2dToVec(u,v):
    vec = np.hstack((u[OLy:-OLy,OLx:-OLx].ravel(),
                     v[OLy:-OLy,OLx:-OLx].ravel()))
    return vec

def linOp_lhs(x, zeta, eta, press,
              hIceMean, Area, areaW, areaS,
              SeaIceMassC, SeaIceMassU, SeaIceMassV,
              cDrag, cBotC, R_low,
              iStep, myTime, myIter):
    # get size for convenience
    nn = x.shape[0]
    # unpack the vector
    def matvec(x):
        u,v = _vecTo2d(x)
        Au, Av = calc_lhs(u, v, zeta, eta, press,
                          hIceMean, Area, areaW, areaS,
                          SeaIceMassC, SeaIceMassU, SeaIceMassV,
                          cDrag, cBotC, R_low,
                          iStep, myIter, myTime)
        return _2dToVec(Au,Av)

    return LinearOperator((nn,nn), matvec=matvec) #, dtype = 'float64')

def jacVecOp(x, uIce, vIce, Fu, Fv, hIceMean, Area,
             zeta, eta, press, cDrag, cBotC,
             SeaIceMassC, SeaIceMassU, SeaIceMassV,
             uIceRHSfix, vIceRHSfix, uVel, vVel, R_low,
             iStep, myTime, myIter):
    # get size for convenience
    n = x.shape[0]
    # unpack the vector
    def matvec(x):
        du,dv = _vecTo2d(x)
        epsjfnk = 1e-9
        Fup, Fvp = calc_residual(uIce+epsjfnk*du, vIce+epsjfnk*dv,
                                 hIceMean, Area,
                                 zeta, eta, press, cDrag, cBotC,
                                 SeaIceMassC, SeaIceMassU, SeaIceMassV,
                                 uIceRHSfix, vIceRHSfix, uVel, vVel, R_low,
                                 iStep, myTime, myIter)

        return _2dToVec( (Fup-Fu)/epsjfnk, (Fvp-Fv)/epsjfnk )

    return LinearOperator((n,n), matvec = matvec) #, dtype = 'float64')

def preconditionerLSR(x, uVel, vVel, hIceMean, Area,
                      press, zeta, eta, cDrag, cBotC, forcingU, forcingV,
                      SeaIceMassC, SeaIceMassU, SeaIceMassV,
                      R_low):
    # get size for convenience
    n = x.shape[0]
    # unpack the vector
    def M_x(x):
        u,v = _vecTo2d(x)

        Au, Av = lsr_solver(u, v, uVel, vVel, hIceMean, Area,
                            press, forcingU, forcingV,
                            SeaIceMassC, SeaIceMassU, SeaIceMassV,
                            R_low, nLsr = -1, nLin = 10,
                            useAsPreconditioner = True,
                            zeta = zeta, eta = eta,
                            cDrag = cDrag, cBotC = cBotC,
                            myTime = 0, myIter = 0)

        return _2dToVec(Au,Av)

    return spla.LinearOperator((n, n), matvec = M_x)

def preconditionerSolve(x, zeta, eta, press,
                        hIceMean, Area, areaW, areaS,
                        SeaIceMassC, SeaIceMassU, SeaIceMassV,
                        cDrag, cBotC, R_low,
                        iStep, myTime, myIter):
    # get size for convenience
    n = x.shape[0]
    # unpack the vector
    def lhs(x):
        u,v = _vecTo2d(x)
        Au, Av = calc_lhs(u, v, zeta, eta, press,
                          hIceMean, Area, areaW, areaS,
                          SeaIceMassC, SeaIceMassU, SeaIceMassV,
                          cDrag, cBotC, R_low,
                          iStep, myIter, myTime)
        return _2dToVec(Au,Av)

    def M_x(x):
        P  = spla.LinearOperator((n, n), matvec = lhs)
        xx, exitCode = spla.lgmres(P, x, maxiter = 5)
        return xx

    return spla.LinearOperator((n, n), matvec = M_x)

def preconGmres(x, P):
    import scipy.sparse.linalg as spla
    n = x.shape[0]
    # taken from https://docs.scipy.org/doc/scipy/reference/generated/ ...
    #                                  scipy.sparse.linalg.gmres.html
    # does not work for some reason
    # M_x = lambda x: spla.gmres(P, x, maxiter = 2)
    # return spla.LinearOperator((n, n), M_x)
    def M_x(x):
        xx, exitCode = spla.lgmres(P, x, maxiter = 5) #, atol = 1e-1)
        return xx
    return spla.LinearOperator((n, n), matvec = M_x)

def picard_solver(uIce, vIce, uVel, vVel, hIceMean, Area,
                  press0, forcingU, forcingV,
                  SeaIceMassC, SeaIceMassU, SeaIceMassV,
                  R_low, myTime, myIter):

    recip_deltaT = 1./deltaTdyn
    bdfAlpha = 1.

    # copy previous time step (n-1) of uIce, vIce
    uIceLin = uIce.copy()
    vIceLin = vIce.copy()
    uIceNm1 = uIce.copy()
    vIceNm1 = vIce.copy()

    # this does not change
    # mass*(uIceNm1)/deltaT
    uIceRHSfix = forcingU + SeaIceMassU*uIceNm1*recip_deltaT
    # mass*(vIceNm1)/deltaT
    vIceRHSfix = forcingV + SeaIceMassV*vIceNm1*recip_deltaT

    residual = np.array([None]*(nPicard+1))
    areaW = 0.5 * (Area + np.roll(Area,1,1))
    areaS = 0.5 * (Area + np.roll(Area,1,0))
    exitCode = 1
    iPicard = -1
    resNonLin = nonLinTol*2
    linTol = 1e-1
    while resNonLin > nonLinTol and iPicard < nPicard:
        iPicard = iPicard+1
        # smoothing
        wght=.8
        uIceLin = wght*uIce+(1.-wght)*uIceLin
        vIceLin = wght*vIce+(1.-wght)*vIceLin
        #
        cDrag = ocean_drag_coeffs(uIceLin, vIceLin, uVel, vVel)
        cBotC = bottomdrag_coeffs(uIceLin, vIceLin, hIceMean, Area, R_low)
        e11, e22, e12    = strainrates(uIceLin, vIceLin, secondOrderBC)
        zeta, eta, press = viscosities(
            e11, e22, e12, press0, iPicard, myIter, myTime)
        bu, bv = calc_rhs(uIceRHSfix, vIceRHSfix, areaW, areaS,
                          uVel, vVel, cDrag,
                          iPicard, myIter, myTime)
        # transform to vectors without overlaps
        b = _2dToVec(bu,bv)
        u = _2dToVec(uIce,vIce)
        # set up linear operator
        A =  linOp_lhs(u, zeta, eta, press,
                       hIceMean, Area, areaW, areaS,
                       SeaIceMassC, SeaIceMassU, SeaIceMassV,
                       cDrag, cBotC, R_low,
                       iPicard, myTime, myIter)
        # M = preconditionerLSR( u, uVel, vVel, hIceMean, Area,
        #                        press, forcingU, forcingV,
        #                        SeaIceMassC, SeaIceMassU, SeaIceMassV,
        #                        R_low, zeta, eta, cDrag, cBotC )
        # M = preconditionerSolve(u, zeta, eta, press,
        #                         hIceMean, Area, areaW, areaS,
        #                         SeaIceMassC, SeaIceMassU, SeaIceMassV,
        #                         cDrag, cBotC, R_low,
        #                         iPicard, myTime, myIter)
        M = preconGmres(u, A)

        if computePicardResidual:
            # print(np.allclose(A.dot(u1), b))
            resNonLin = np.sqrt( ( (A.matvec(u)-b)**2 ).sum() )
            if printPicardResidual or exitCode>0: print(
                'iPicard = %3i, pre-gmres non-linear residual       = %e'%(
                    iPicard, resNonLin) )

        if exitCode == 0:
            linTol = linTol*(1-.7)
        else:
            # reset
            linTol = 1.e-1

        # matrix free solver that calls calc_lhs
        #u1, exitCode = spla.bicgstab(A,b,x0=u,maxiter=nLinear,tol=linTol)
        u1, exitCode = spla.gmres(A, b, x0 = u, M = M,
                                  maxiter = nLinear, tol = linTol)

        uIce, vIce = _vecTo2d(u1)
        if computePicardResidual:
            # print(np.allclose(A.dot(u1), b))
            residual[iPicard] = resNonLin
            if iPicard==0: resNonLin0 = resNonLin
            resNonLin = resNonLin/resNonLin0

            resNonLinPost = np.sqrt( ( (A.matvec(u1)-b)**2 ).sum() )

            if printPicardResidual or exitCode>0: print(
                    'iPicard = %3i, exitCode = %3i, non-linear residual = %e'%(
                        iPicard, exitCode, resNonLinPost )) #residual[iPicard]))
            # Au, Av = calc_lhs(uIce, vIce, zeta, eta, press,
            #                   hIceMean, Area, areaW, areaS,
            #                   SeaIceMassC, SeaIceMassU, SeaIceMassV,
            #                   cDrag, cBotC, R_low,
            #                   iPicard, myIter, myTime)
            # print(np.sqrt(((Au-bu)**2+(Av-bv)**2)[OLy:-OLy,OLx:-OLx].sum()),
            #       residual[iPicard])


    if computePicardResidual and plotPicardResidual:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(nrows=2,ncols=1,sharex=True)
        ax[0].semilogy(residual[:iPicard]/residual[0],'x-')
        ax[0].set_title('residual')
        plt.show()

    return uIce, vIce

def jfnk_solver(uIce, vIce, uVel, vVel, hIceMean, Area,
                press0, forcingU, forcingV,
                SeaIceMassC, SeaIceMassU, SeaIceMassV,
                R_low, myTime, myIter):

    recip_deltaT = 1./deltaTdyn
    bdfAlpha = 1.

    # copy previous time step (n-1) of uIce, vIce
    uIceLin = uIce.copy()
    vIceLin = vIce.copy()
    uIceNm1 = uIce.copy()
    vIceNm1 = vIce.copy()

    # this does not change
    # mass*(uIceNm1)/deltaT
    uIceRHSfix = forcingU + SeaIceMassU*uIceNm1*recip_deltaT
    # mass*(vIceNm1)/deltaT
    vIceRHSfix = forcingV + SeaIceMassV*vIceNm1*recip_deltaT

    residual = np.array([None]*(nJFNK+1))
    areaW = 0.5 * (Area + np.roll(Area,1,1))
    areaS = 0.5 * (Area + np.roll(Area,1,0))
    exitCode = 1
    iJFNK = -1
    resNonLin = nonLinTol*2
    linTol = 1e-1
    while resNonLin > nonLinTol and iJFNK < nJFNK:
        iJFNK = iJFNK+1
        # smoothing
        wght=.8
        uIceLin = wght*uIce+(1.-wght)*uIceLin
        vIceLin = wght*vIce+(1.-wght)*vIceLin
        #
        cDrag = ocean_drag_coeffs(uIceLin, vIceLin, uVel, vVel)
        cBotC = bottomdrag_coeffs(uIceLin, vIceLin, hIceMean, Area, R_low)
        e11, e22, e12    = strainrates(uIceLin, vIceLin, secondOrderBC)
        zeta, eta, press = viscosities(
            e11, e22, e12, press0, iJFNK, myIter, myTime)
        Fu, Fv = calc_residual(uIce, vIce, hIceMean, Area,
                               zeta, eta, press, cDrag, cBotC,
                               SeaIceMassC, SeaIceMassU, SeaIceMassV,
                               uIceRHSfix, vIceRHSfix, uVel, vVel, R_low,
                               iJFNK, myTime, myIter)
        # transform to vectors without overlaps
        b = _2dToVec(Fu,Fv)
        u = _2dToVec(uIce,vIce)
        # set up linear operator
        J = jacVecOp(u, uIce, vIce, Fu, Fv, hIceMean, Area,
                     zeta, eta, press, cDrag, cBotC,
                     SeaIceMassC, SeaIceMassU, SeaIceMassV,
                     uIceRHSfix, vIceRHSfix, uVel, vVel, R_low,
                     iJFNK, myTime, myIter)
        # preconditioner
        # A =  linOp_lhs(u, zeta, eta, press,
        #                hIceMean, Area, areaW, areaS,
        #                SeaIceMassC, SeaIceMassU, SeaIceMassV,
        #                cDrag, cBotC, R_low,
        #                iJFNK, myTime, myIter)
        # M = preconditionerLSR( u, uVel, vVel, hIceMean, Area,
        #                        press, forcingU, forcingV,
        #                        SeaIceMassC, SeaIceMassU, SeaIceMassV,
        #                        R_low, zeta, eta, cDrag, cBotC )
        # M = preconditionerSolve(u, zeta, eta, press,
        #                         hIceMean, Area, areaW, areaS,
        #                         SeaIceMassC, SeaIceMassU, SeaIceMassV,
        #                         cDrag, cBotC, R_low,
        #                         iJFNK, myTime, myIter)
        M = preconGmres(u, J)

        resNonLin = np.sqrt( (Fu**2+Fv**2).sum() )

        if printJFNKResidual:
            print(
                'iJFNK = %3i, pre-gmres non-linear residual       = %e'%(
                    iJFNK, resNonLin) )

        # compute convergence criterion for linear preconditioned FGMRES
        linTolMax = 0.99
        linTolMin = 0.01
        linTol = linTolMax
        if iJFNK>0 and iJFNK <= 100 and resNonLin<resT:
            print('hallo')
            # Eisenstat and Walker (1996), eq.(2.6)
            linTol = 0.7*( resNonLin/resNonLinKm1 )**1.5
            linTol = min(linTolMax, linTol)
            linTol = max(linTolMin, linTol)
        elif iJFNK==0:
            print('hallo0')
            resT = resNonLin * 0.5
            resNonLinKm1 = 1.

        # matrix free solver that calls jacobian times vector
        du, exitCode = spla.gmres(J, -b,
                                  M = M,
                                  maxiter = nLinear, tol = linTol)

        print('gmres: exitCode = %i, linTol = %f, %f'%(
            exitCode, linTol, resNonLin/resNonLinKm1))
        # save the residual for the next iteration
        resNonLinKm1 = resNonLin

        # Newton step with line search
        doLineSearch = True
        iLineSearch = -1
        lsFac = 1.
        while iLineSearch < 20 and doLineSearch:
            iLineSearch = iLineSearch + 1
            lsFac = lsFac*(1-.7)
            uIce, vIce = _vecTo2d(u + du*lsFac)
            Fu0, Fv0 = calc_residual(uIce, vIce, hIceMean, Area,
                                     zeta, eta, press, cDrag, cBotC,
                                     SeaIceMassC, SeaIceMassU, SeaIceMassV,
                                     uIceRHSfix, vIceRHSfix, uVel, vVel, R_low,
                                     iJFNK, myTime, myIter)
            resNonLinPost = np.sqrt( (Fu0**2+Fv0**2).sum() )
            if resNonLinPost < resNonLin:
                doLineSearch = False
                print('line search: %04i pre-gmres: %e update = %e'%(
                    iLineSearch,resNonLin,resNonLinPost))


        if computeJFNKResidual:
            # print(np.allclose(A.dot(u1), b))
            residual[iJFNK] = resNonLin
            if iJFNK==0: resNonLin0 = resNonLin

            # Fu0, Fv0 = calc_residual(uIce, vIce, hIceMean, Area,
            #                          zeta, eta, press, cDrag, cBotC,
            #                          SeaIceMassC, SeaIceMassU, SeaIceMassV,
            #                          uIceRHSfix, vIceRHSfix, uVel, vVel, R_low,
            #                          iJFNK, myTime, myIter)
            # resNonLinPost = np.sqrt( (Fu0**2+Fv0**2).sum() )

            # if printJFNKResidual or exitCode>0: print(
            #         'iJFNK = %3i, exitCode = %3i, non-linear residual = %e'%(
            #             iJFNK, exitCode, resNonLinPost )) #residual[iJFNK]))
            # Au, Av = calc_lhs(uIce, vIce, zeta, eta, press,
            #                   hIceMean, Area, areaW, areaS,
            #                   SeaIceMassC, SeaIceMassU, SeaIceMassV,
            #                   cDrag, cBotC, R_low,
            #                   iJFNK, myIter, myTime)
            # print(np.sqrt(((Au-bu)**2+(Av-bv)**2)[OLy:-OLy,OLx:-OLx].sum()),
            #       residual[iJFNK])


    if computeJFNKResidual and plotJFNKResidual:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(nrows=2,ncols=1,sharex=True)
        ax[0].semilogy(residual[:iJFNK]/residual[0],'x-')
        ax[0].set_title('residual')
        plt.show()

    return uIce, vIce
