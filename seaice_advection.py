import numpy as np

from seaice_size import *
from seaice_params import *

from seaice_fluxlimit_adv_x import fluxlimit_adv_x
from seaice_fluxlimit_adv_y import fluxlimit_adv_y
from seaice_fill_overlap import fill_overlap

# calculates the tendency of a sea ice field due to advection

### input
# uFld: zonal velocity component
# vFld: meridional velocity component
# uTrans: volume transport at U points
# vTrans: volume transport at V points
# iceFld: sea ice field (mean thickness of ice or snow, or area)
# r_hFld: reciprocal of ice thickness
# extensiveFld: indicates to advect an "extensive" type of ice field

### output
# gFld: advection tendency


def advection(uFld, vFld, uTrans, vTrans, iceFld, r_hFld, extensiveFld):

    # make local copy of sea-ice field
    localTij = iceFld.copy()

    # mask West & South
    # ifdef ALLOW_OBCS
    maskLocW = SeaIceMaskU * maskInW
    maskLocS = SeaIceMaskV * maskInS


    ##### calculate advective flux in x direction #####

    # advective flux in x direction
    afx = fluxlimit_adv_x(uFld, localTij, uTrans, deltaTtherm, maskLocW)

    # update the local seaice field
    if extensiveFld:
        localTij = localTij - deltaTtherm * maskInC * recip_rA \
            * ( np.roll(afx,-1,1) - afx )
    else:
        localTij= localTij- deltaTtherm * maskInC * recip_rA * r_hFld * (
            ( np.roll(afx,-1,1) - afx )
            - ( np.roll(uTrans,-1,1) - uTrans ) * iceFld
        )

    ##### calculate advective flux in y direction #####

    # advective flux in y direction
    afy = fluxlimit_adv_y(vFld, localTij, vTrans, deltaTtherm, maskLocS)

    # update the local seaice field
    if extensiveFld:
        localTij = localTij - deltaTtherm * maskInC * recip_rA \
            * ( np.roll(afy,-1,0) - afy )
    else:
        localTij = localTij - deltaTtherm * maskInC * recip_rA * r_hFld * (
            ( np.roll(afy,-1,0) - afy )
            - ( np.roll(vTrans,-1,0) - vTrans) * iceFld
        )

    # explicit advection is done, store tendency in gFld
    return (localTij - iceFld) / deltaTtherm
