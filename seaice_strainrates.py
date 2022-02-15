import numpy as np

from seaice_params import *
from seaice_size import *

### input
# uFld: zonal ice field velocity
# vFld: meridional ice Field velocity
# secondOrderBC: flag

### output
# e11: 1,1 component of strain rate tensor
# e22: 2,2 component of strain rate tensor
# e12: 1,2 component of strain rate tensor


def strainrates(uFld, vFld, secondOrderBC):

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
