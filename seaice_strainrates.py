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

    # initializations
    noSlipFac = 1

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
    hFacU = SeaIceMaskU - np.roll(SeaIceMaskU,1,axis=0)
    hFacV = SeaIceMaskV - np.roll(SeaIceMaskV,1,axis=1)
    mskZ = iceMask*np.roll(iceMask,1,axis=1)
    mskZ =    mskZ*np.roll(   mskZ,1,axis=0)
    e12 = 0.5 * (dudy + dvdx - k1AtZ * vave - k2AtZ * uave ) * mskZ \
        + noSlipFac * (
              2.0 * uave * recip_dyU * hFacU
            + 2.0 * vave * recip_dxV * hFacV
        )

    if secondOrderBC: print('not implemented yet')
    # if secondOrderBC:
    #     hFacU = (SeaIceMaskU[2:-1,2:-1] - SeaIceMaskU[1:-2,2:-1]) / 3.
    #     hFacV = (SeaIceMaskV[2:-1,2:-1] - SeaIceMaskV[2:-1,1:-2]) / 3.
    #     hFacU = hFacU * (SeaIceMaskU[:-3,2:-1] * SeaIceMaskU[1:-2,2:-1]
    #                      + SeaIceMaskU[3:,2:-1] * SeaIceMaskU[2:-1,2:-1])
    #     hFacV = hFacV * (SeaIceMaskV[2:-1,:-3] * SeaIceMaskV[2:-1,1:-2]
    #                      + SeaIceMaskV[2:-1,3:] * SeaIceMaskV[2:-1,2:-1])

    #     e12[2:-1,2:-1] = e12[2:-1,2:-1] + 0.5 * (
    #         recip_dyU[2:-1,2:-1] * (6 * uave[2:-1,2:-1]
    #                                 - uFld[:-3,2:-1] * SeaIceMaskU[1:-2,2:-1]
    #                                 - uFld[3:,2:-1] * SeaIceMaskU[2:-1,2:-1]
    #                                 ) * hFacU \
    #         + recip_dxV[2:-1,2:-1] * (
    #             6 * vave[2:-1,2:-1]
    #             - vFld[2:-1,:-3] * SeaIceMaskV[2:-1,1:-2]
    #             - vFld[2:-1,3:] * SeaIceMaskV[2:-1,2:-1]) * hFacV)

    return e11, e22, e12
