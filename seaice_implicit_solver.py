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

secondOrderBC = False

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
             hIceMean, Area, areaW, areaS, press0,
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

def calc_rhs(uIceRHS, vIceRHS, areaW, areaS,
             uVel, vVel, cDrag,
             iStep, myTime, myIter):

    sinWat = np.sin(np.deg2rad(waterTurnAngle))
    cosWat = np.cos(np.deg2rad(waterTurnAngle))

    # ice-velocity independent contribution to drag terms
    # - Cdrag*(uVel*cosWat - vVel*sinWat)/(vVel*cosWat + uVel*sinWat)
    # ( remember to average to correct velocity points )
    uAtC = 0.5 * ( uVel + np.roll(uVel,-1,1) )
    vAtC = 0.5 * ( vVel + np.roll(vVel,-1,0) )
    uIceRHS = uIceRHS + (
        0.5 * ( cDrag + np.roll(cDrag,1,1) ) * cosWat *  uVel
        - np.sign(fCori) * sinWat * 0.5 * (
            cDrag * vAtC + np.roll(cDrag * vAtC,1,1)
        )
    ) * areaW
    vIceRHS = vIceRHS + (
        0.5 * ( cDrag + np.roll(cDrag,1,0) ) * cosWat * vVel
        + np.sign(fCori) * sinWat * 0.5 * (
            cDrag * uAtC  + np.roll(cDrag * uAtC,1,0)
        )
    ) * areaS

    return uIceRHS*maskInW, vIceRHS*maskInS

def calc_residual(uIce, vIce, hIceMean, Area,
                  SeaIceMassC, SeaIceMassU, SeaIceMassV,
                  uVel, vVel, R_low,
                  iStep, myTime, myIter):

    # initialize fractional areas at velocity points
    areaW = 0.5 * (Area + np.roll(Area,1,1))
    areaS = 0.5 * (Area + np.roll(Area,1,0))
    cDrag = ocean_drag_coeffs(uIce, vIce, uVel, vVel)

    Au, Av = calc_lhs(uIce, vIce, zeta, eta, press,
                      hIceMean, Area, areaW, areaS, press0,
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

def linOp_lhs(xIn, zeta, eta, press,
              hIceMean, Area, areaW, areaS, press0,
              SeaIceMassC, SeaIceMassU, SeaIceMassV,
              cDrag, cBotC, R_low,
              iStep, myTime, myIter):
    # get size for convenience
    nn = xIn.shape[0]
    # unpack the vector
    def matvec(xIn):
        u,v = _vecTo2d(xIn)
        Au, Av = calc_lhs(u, v, zeta, eta, press,
                          hIceMean, Area, areaW, areaS,
                          press0,
                          SeaIceMassC, SeaIceMassU, SeaIceMassV,
                          cDrag, cBotC, R_low,
                          iStep, myIter, myTime)
        return _2dToVec(Au,Av)

    return LinearOperator((nn,nn), matvec=matvec) #, dtype = 'float64')

def preconditionerMatrix(x, P):
    n = x.shape[0]
    M_x = lambda x: spla.spsolve(P, x)
    return spla.LinearOperator((n, n), M_x)

def picard_solver(uIce, vIce, uVel, vVel, hIceMean, Area,
                  press0, forcingU, forcingV,
                  SeaIceMassC, SeaIceMassU, SeaIceMassV,
                  R_low, myTime, myIter):

    computePicardResidual = True

    recip_deltaT = 1./deltaTdyn
    bdfAlpha = 1.


    # copy previous time step (n-1) of uIce, vIce
    uIceLin = uIce.copy()
    vIceLin = vIce.copy()

    # this does not change
    # mass*(uIceNm1)/deltaT
    uIceRHS = forcingU + SeaIceMassU*uIce*recip_deltaT
    # mass*(vIceNm1)/deltaT
    vIceRHS = forcingV + SeaIceMassV*vIce*recip_deltaT

    nPicard = 100
    nLinear = 20
    residual = np.array([None]*nPicard)
    areaW = 0.5 * (Area + np.roll(Area,1,1))
    areaS = 0.5 * (Area + np.roll(Area,1,0))
    exitCode = 1
    linTol = 1e-1
    for iPicard in range(nPicard):
        # smoothing
        wght=0.5
        uIceLin = wght*uIce+(1.-wght)*uIceLin
        vIceLin = wght*vIce+(1.-wght)*vIceLin
        #
        cDrag = ocean_drag_coeffs(uIceLin, vIceLin, uVel, vVel)
        cBotC = bottomdrag_coeffs(uIceLin, vIceLin, hIceMean, Area, R_low)
        e11, e22, e12    = strainrates(uIceLin, vIceLin, secondOrderBC)
        zeta, eta, press = viscosities(
            e11, e22, e12, press0, iPicard, myIter, myTime)
        bu, bv = calc_rhs(uIceRHS, vIceRHS, areaW, areaS,
                          uVel, vVel, cDrag,
                          iPicard, myIter, myTime)
        # transform to vectors without overlaps
        b = _2dToVec(bu,bv)
        u = _2dToVec(uIce,vIce)
        # set up linear operator
        A =  linOp_lhs(u, zeta, eta, press,
                       hIceMean, Area, areaW, areaS, press0,
                       SeaIceMassC, SeaIceMassU, SeaIceMassV,
                       cDrag, cBotC, R_low,
                       iPicard, myTime, myIter)
        # M = preconditionerMatrix( u, A )
        # matrix free solver that calls calc_lhs
        if exitCode == 0: linTol = linTol*(1-.7)
        #u1, exitCode = spla.bicgstab(A,b,x0=u,maxiter=nLinear,tol=linTol)
        u1, exitCode = spla.gmres(A,b,x0=u,maxiter=nLinear,tol=linTol)
        # for iLin in range(nLinear):
        uIce, vIce = _vecTo2d(u1)
        if computePicardResidual:
            # print(np.allclose(A.dot(u1), b))
            residual[iPicard] = np.sqrt( ( (A.matvec(u1)-b)**2 ).sum() )
            # if exitCode>0: print(
            if True: print(
                    'iPicard = %3i, exitCode = %3i,linear residual = %e'%(
                        iPicard, exitCode, residual[iPicard]))
            # Au, Av = calc_lhs(uIce, vIce, zeta, eta, press,
            #                   hIceMean, Area, areaW, areaS,
            #                   press0,
            #                   SeaIceMassC, SeaIceMassU, SeaIceMassV,
            #                   cDrag, cBotC, R_low,
            #                   iPicard, myIter, myTime)
            # print(np.sqrt(((Au-bu)**2+(Av-bv)**2)[OLy:-OLy,OLx:-OLx].sum()),
            #       residual[iPicard])

    if computePicardResidual:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(nrows=2,ncols=1,sharex=True)
        ax[0].semilogy(residual[:],'x-')
        ax[0].set_title('residual')
        plt.show()

    return uIce, vIce
