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

    noSlipFac = 1

    # abbreviations at c points
    dudx = (uFld[:-1,1:] - uFld[:-1,:-1]) * recip_dxF[:-1,:-1]
    uave = 0.5 * (uFld[:-1,:-1] + uFld[:-1,1:])
    dvdy = (vFld[1:,:-1] - vFld[:-1,:-1]) * recip_dyF[:-1,:-1]
    vave = 0.5 * (vFld[:-1,:-1] + vFld[1:,:-1])

    # evaluate strain rates at c points
    e11 = np.ones((sNy+2*OLy,sNx+2*OLx))
    e22 = np.ones((sNy+2*OLy,sNx+2*OLx))
    e11[:-1,:-1] = (dudx + vave * k2AtC[:-1,:-1]) * maskInC
    e22[:-1,:-1] = (dvdy + uave * k1AtC[:-1,:-1]) * maskInC

    # abbreviations at z points - what are z points?
    dudy = (uFld[1:,1:] - uFld[:-1,1:]) * recip_dyU[1:,1:]
    uave = 0.5 * (uFld[1:,1:] + uFld[:-1,1:])
    dvdx = (vFld[1:,1:] - vFld[1:,:-1]) * recip_dxV[1:,1:]
    vave = 0.5 * (vFld[1:,1:] + vFld[1:,:-1])

    # evaluate strain rate at z points
    e12 = np.ones((sNy+2*OLy,sNx+2*OLx))
    hFacU = SeaIceMaskU[1:,1:] - SeaIceMaskU[:-1,1:]
    hFacV = SeaIceMaskV[1:,1:] - SeaIceMaskV[1:,:-1]
    e12[1:,1:] = 0.5 * (dudy + dvdx - k1AtZ[1:,1:] * vave - k2AtZ[1:,1:] * uave) * hIceMeanMask[1:,1:] * hIceMeanMask[1:,:-1] * hIceMeanMask[:-1,1:] * hIceMeanMask[:-1,:-1] + noSlipFac * (2 * uave[1:,1:] * recip_dyU[1:,1:] * hFacU + 2 * vave[1:,1:] * recip_dxV[1:,1:] * hFacV)

    if secondOrderBC:
        hFacU = (SeaIceMaskU[2:-1,2:-1] - SeaIceMaskU[1:-2,2:-1]) / 3
        hFacV = (SeaIceMaskV[2:-1,2:-1] - SeaIceMaskV[2:-1,1:-2]) / 3
        hFacU = hFacU * (SeaIceMaskU[:-3,2:-1] * SeaIceMaskU[1:-2,2:-1] + SeaIceMaskU[3:,2:-1] * SeaIceMaskU[2:-1,2:-1])
        hFacV = hFacV * (SeaIceMaskV[2:-1,:-3] * SeaIceMaskV[2:-1,1:-2] + SeaIceMaskV[2:-1,3:] * SeaIceMaskV[2:-1,2:-1])

        e12[2:-1,2:-1] = e12[2:-1,2:-1] + 0.5 * (recip_dyU[2:-1,2:-1] * (6 * uave - uFld[:-3,2:-1] * SeaIceMaskU[1:-2,2:-1] - uFld[3:,2:-1] * SeaIceMaskU[2:-1,2:-1]) * hFacU + recip_dxV[2:-1,2:-1] * (6 * vave - vFld[2:-1,:-3] * SeaIceMaskV[2:-1,1:-2] - vFld[2:-1,3:] * SeaIceMaskV[2:-1,2:-1]) * hFacV)


    return e11, e22, e12