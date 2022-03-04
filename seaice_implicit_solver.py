import numpy as np

from scipy.sparse import linalg as spla
from scipy.sparse.linalg import LinearOperator

from seaice_params import *
from seaice_size import *

from dynamics_routines import strainrates, \
    calc_ice_strength, viscosities, \
    ocean_drag_coeffs, bottomdrag_coeffs, \
    calc_stressdiv, calc_stress, calc_lhs, calc_rhs, calc_residual

from seaice_global_sum import global_sum
from seaice_fill_overlap import fill_overlap_uv
from seaice_lsr import lsr_solver

nLinear = 50
nonLinTol = 1e-5

useLsrAsPreconditioner = True

nPicard = 10
computePicardResidual = True
printPicardResidual = True
plotPicardResidual = True

nJFNK = 100
computeJFNKResidual = True
printJFNKResidual = True
plotJFNKResidual = True

def _vecTo2d(vec):
    n = vec.shape[0]//2
    u = np.zeros((ny+2*oly,nx+2*olx))
    v = np.zeros((ny+2*oly,nx+2*olx))
    u[oly:-oly,olx:-olx]=vec[:n].reshape((ny,nx))
    v[oly:-oly,olx:-olx]=vec[n:].reshape((ny,nx))
    u,v=fill_overlap_uv(u,v)
    return u,v

def _2dToVec(u,v):
    vec = np.hstack((u[oly:-oly,olx:-olx].ravel(),
                     v[oly:-oly,olx:-olx].ravel()))
    return vec

def calc_nonlinear_residual( Fu, Fv ):
    return np.sqrt( _2dToVec(Fu**2*rAw,Fv**2*rAs).sum()/globalArea )

def matVecOp(x, zeta, eta, press,
             hIceMean, Area, areaW, areaS,
             SeaIceMassC, SeaIceMassU, SeaIceMassV,
             cDrag, cBotC, R_low,
             iStep, myTime, myIter):
    # get size for convenience
    n = x.shape[0]
    # unpack the vector
    def matvec(x):
        u,v = _vecTo2d(x)
        Au, Av = calc_lhs(u, v, zeta, eta, press,
                          hIceMean, Area, areaW, areaS,
                          SeaIceMassC, SeaIceMassU, SeaIceMassV,
                          cDrag, cBotC, R_low,
                          iStep, myIter, myTime)
        return _2dToVec(Au,Av)

    return LinearOperator((n,n), matvec = matvec, dtype = 'float64')

def jacVecOp(x, uIce, vIce, Fu, Fv, hIceMean, hSnowMean, Area,
             zeta, eta, press, cDrag, cBotC,
             SeaIceMassC, SeaIceMassU, SeaIceMassV,
             uIceRHSfix, vIceRHSfix, uVel, vVel, R_low,
             iStep, myTime, myIter):
    # get size for convenience
    n = x.shape[0]
    # unpack the vector
    def matvec(x):
        du,dv = _vecTo2d(x)
        epsjfnk = 1e-12
        Fup, Fvp = calc_residual(uIce+epsjfnk*du, vIce+epsjfnk*dv,
                                 hIceMean, hSnowMean, Area,
                                 SeaIceMassC, SeaIceMassU, SeaIceMassV,
                                 uIceRHSfix, vIceRHSfix, uVel, vVel, R_low,
                                 iStep, myTime, myIter)

        return _2dToVec( (Fup-Fu)/epsjfnk, (Fvp-Fv)/epsjfnk )

    return LinearOperator((n,n), matvec = matvec, dtype = 'float64')

def preconditionerLSR(x, uVel, vVel, hIceMean, hSnowMean, Area,
                      zeta, eta, cDrag, cBotC, forcingU, forcingV,
                      SeaIceMassC, SeaIceMassU, SeaIceMassV,
                      R_low):
    # get size for convenience
    n = x.shape[0]
    # unpack the vector
    def M_x(x):
        u,v = _vecTo2d(x)

        Au, Av = lsr_solver(u, v, hIceMean*0, hSnowMean*0, Area,
                            uVel, vVel, forcingU*0, forcingV*0,
                            SeaIceMassC, SeaIceMassU, SeaIceMassV,
                            R_low, nLsr = 1, nLin = 10,
                            useAsPreconditioner = True,
                            zeta = zeta, eta = eta,
                            cDrag = cDrag, cBotC = cBotC,
                            myTime = -1., myIter = -1)

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

def picard_solver(uIce, vIce, hIceMean, hSnowMean, Area,
                  uVel, vVel, forcingU, forcingV,
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
    # calculate ice strength
    press0 = calc_ice_strength(hIceMean, iceMask)


    residual = []
    areaW = 0.5 * (Area + np.roll(Area,1,1))
    areaS = 0.5 * (Area + np.roll(Area,1,0))
    exitCode = 1
    iPicard = -1
    resNonLin = nonLinTol*2
    linTolMax = 1e-1
    linTol = linTolMax
    while resNonLin > nonLinTol and iPicard < nPicard:
        iPicard = iPicard+1
        # smoothing
        wght=.8
        uIceLin = wght*uIce+(1.-wght)*uIceLin
        vIceLin = wght*vIce+(1.-wght)*vIceLin
        #
        cDrag = ocean_drag_coeffs(uIceLin, vIceLin, uVel, vVel)
        cBotC = bottomdrag_coeffs(uIceLin, vIceLin, hIceMean, Area, R_low)
        e11, e22, e12    = strainrates(uIceLin, vIceLin)
        zeta, eta, press = viscosities(
            e11, e22, e12, press0, iPicard, myIter, myTime)
        bu, bv = calc_rhs(uIceRHSfix, vIceRHSfix, areaW, areaS,
                          uVel, vVel, cDrag,
                          iPicard, myIter, myTime)
        # transform to vectors without overlaps
        b = _2dToVec(bu,bv)
        u = _2dToVec(uIce,vIce)
        # set up linear operator
        A =  matVecOp(u, zeta, eta, press,
                      hIceMean, Area, areaW, areaS,
                      SeaIceMassC, SeaIceMassU, SeaIceMassV,
                      cDrag, cBotC, R_low,
                      iPicard, myTime, myIter)
        if useLsrAsPreconditioner:
            M = preconditionerLSR(u, uVel, vVel, hIceMean, hSnowMean, Area,
                                  zeta, eta, cDrag, cBotC, forcingU, forcingV,
                                  SeaIceMassC, SeaIceMassU, SeaIceMassV,
                                  R_low)
        else:
            M = preconGmres(u, A)

        if exitCode == 0:
            linTolMin = 1e-2
            linTol = max(linTol*(1-.1),linTolMin)
        else:
            # reset
            linTol = linTolMax

        if computePicardResidual:
            # print(np.allclose(A.dot(u1), b))
            resNonLin = np.sqrt( ( (A.matvec(u)-b)**2 ).sum() )
            if printPicardResidual or exitCode>0: print(
                'iPicard = %3i, linTol %f non-linear residual = %e'%(
                    iPicard, linTol, resNonLin) )

        # matrix free solver that calls calc_lhs
        #u1, exitCode = spla.bicgstab(A,b,x0=u,maxiter=nLinear,tol=linTol)
        u1, exitCode = spla.gmres(A, b, x0 = u, M = M,
                                  maxiter = nLinear, atol = linTol)

        uIce, vIce = _vecTo2d(u1)
        if computePicardResidual:
            # print(np.allclose(A.dot(u1), b))
            residual.append(resNonLin)
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
            # print(np.sqrt(((Au-bu)**2+(Av-bv)**2)[oly:-oly,olx:-olx].sum()),
            #       residual[iPicard])


    if computePicardResidual and plotPicardResidual:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(nrows=2,ncols=1,sharex=True)
        ax[0].semilogy(residual/residual[0],'x-')
        ax[0].set_title('residual')
        plt.show()

    return uIce, vIce

def jfnk_solver(uIce, vIce, hIceMean, hSnowMean, Area,
                uVel, vVel, forcingU, forcingV,
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
    # calculate ice strength
    press0 = calc_ice_strength(hIceMean, iceMask)

    residual = []
    areaW = 0.5 * (Area + np.roll(Area,1,1))
    areaS = 0.5 * (Area + np.roll(Area,1,0))
    exitCode = 1
    iJFNK = -1
    resNonLin = 1.
    JFNKtol = 0.
    while resNonLin > JFNKtol and iJFNK < nJFNK:
        iJFNK = iJFNK+1

        # smoothing
        wght=1.
        uIceLin = wght*uIce+(1.-wght)*uIceLin
        vIceLin = wght*vIce+(1.-wght)*vIceLin

        # these are just for the lsr-preconditioner
        cDrag = ocean_drag_coeffs(uIceLin, vIceLin, uVel, vVel)
        cBotC = bottomdrag_coeffs(uIceLin, vIceLin, hIceMean, Area, R_low)
        e11, e22, e12    = strainrates(uIceLin, vIceLin)
        zeta, eta, press = viscosities(
            e11, e22, e12, press0, iJFNK, myIter, myTime)

        # first residual
        Fu, Fv = calc_residual(uIce, vIce, hIceMean, hSnowMean, Area,
                               SeaIceMassC, SeaIceMassU, SeaIceMassV,
                               uIceRHSfix, vIceRHSfix, uVel, vVel, R_low,
                               iJFNK, myTime, myIter)

        resNonLin = calc_nonlinear_residual(Fu, Fv)
        residual.append(resNonLin)
        # compute non-linear convergence criterion
        if iJFNK==0:
            JFNKtol = resNonLin*nonLinTol
            resT    = resNonLin * 0.5
            print('JFNKtol = %e'%JFNKtol)

        # compute convergence criterion for linear solver
        linTolMax = 0.99
        linTolMin = 0.01
        linTol = linTolMax
        if iJFNK > 0 and iJFNK < 100 and resNonLin < resT:
            # Eisenstat and Walker (1996), eq.(2.6)
            linTol = 0.7*( resNonLin/resNonLinKm1 )**1.5
            linTol = min(linTolMax, linTol)
            linTol = max(linTolMin, linTol)
        elif iJFNK==0:
            resNonLinKm1 = resNonLin

        if printJFNKResidual:
            print(
                'iJFNK = %3i, linTol = %f,   non-lin residual = %e'%(
                    iJFNK, linTol, resNonLin) )

        # transform 2D fields to vectors without overlaps
        b = _2dToVec(Fu,Fv)
        u = _2dToVec(uIce,vIce)
        # set up Jacobian times vector operator
        J = jacVecOp(u, uIce, vIce, Fu, Fv, hIceMean, hSnowMean, Area,
                     zeta, eta, press, cDrag, cBotC,
                     SeaIceMassC, SeaIceMassU, SeaIceMassV,
                     uIceRHSfix, vIceRHSfix, uVel, vVel, R_low,
                     iJFNK, myTime, myIter)
        # preconditioner
        if useLsrAsPreconditioner:
            M = preconditionerLSR(u, uVel, vVel, hIceMean, hSnowMean, Area,
                                  zeta, eta, cDrag, cBotC, forcingU, forcingV,
                                  SeaIceMassC, SeaIceMassU, SeaIceMassV,
                                  R_low)
        else:
            # matrix free solver that calls jacobian times vector
            M = preconGmres(u, J)

        if ( iJFNK < 0 ):
            du, exitCode = spla.gmres(J, -b,
                                  maxiter = nLinear, atol = linTol)
        else:
            du, exitCode = spla.gmres(J, -b,
                                  M = M,
                                  maxiter = nLinear, atol = linTol)

        print('gmres: exitCode = %i, linTol = %f, %f, %f'%(
            exitCode, linTol, du.min(), du.max() ) )
        if np.abs(du).max() == 0:
            print('Newton innovation vector = 0, stopping')
            break

        # Newton step with line search
        iLineSearch = -1
        lsGamma = 0.7
        resNonLinLS = 2*resNonLin
        while iLineSearch < 20 and resNonLinLS > resNonLin:
            iLineSearch = iLineSearch + 1
            lsFac = (1.-lsGamma)**iLineSearch
            uIce, vIce = _vecTo2d(u + du*lsFac)
            Fu, Fv = calc_residual(uIce, vIce, hIceMean, hSnowMean, Area,
                                   SeaIceMassC, SeaIceMassU, SeaIceMassV,
                                   uIceRHSfix, vIceRHSfix, uVel, vVel, R_low,
                                   iJFNK, myTime, myIter)
            resNonLinLS =  calc_nonlinear_residual(Fu, Fv)
            if resNonLinLS < resNonLin:
                print('line search: %04i resKm1 = %e, %i updates = %e'%(
                    iLineSearch+1,resNonLin,iLineSearch,resNonLinLS))

        # save the residual for the next iteration
        resNonLinKm1 = resNonLin
        resNonLin    = resNonLinLS

    # after the outer nonlinear iteration is done plot residual
    if computeJFNKResidual and plotJFNKResidual:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(nrows=2,ncols=1,sharex=True)
        ax[0].semilogy(residual/residual[0],'x-')
        ax[0].set_title('residual')
        plt.show()

    return uIce, vIce
