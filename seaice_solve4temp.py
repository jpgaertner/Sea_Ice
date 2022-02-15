# Calculates heat fluxes and temperature of the ice surface
# (see Hibler, MWR, 108, 1943-1973, 1980)

import numpy as np

from seaice_size import *
from seaice_params import *

sNx = 1
sNy = 1
OLx = 2
OLy = 2

### input
# hIceActual: actual ice thickness [m]
# hSnowActual: actual snow thickness [m]
# TSurfIn: surface temperature of ice/snow [K]
# TempFrz: freezing temperature [K]
# ug: atmospheric wind speed [m/s]
# SWDown: shortwave radiative downward flux [W/m2]
# LWDown: longwave radiative downward flux [W/m2]
# ATemp: atmospheric temperature [K]
# aqh: atmospheric specific humidity [g/kg]

### output
# TSurfOut: updated surface temperature of ice/snow [K]
# F_io_net: upward conductive heat flux through sea ice and snow [W/m2]
# F_ia_net: net heat flux divergence at the sea ice/snow surface
#   including ice conductive fluxes and atmospheric fluxes [W/m2]
# F_ia: upward seaice/snow surface heat flux to atmosphere [W/m^2]
# IcePenetSW: short wave heat flux arriving at the ocean-ice interface (+ = upward) [W/m^2]
# FWsublim: fresh water (mass) flux due to sublimation (+ = upward) [kg/sm2]


def solve4temp(hIceActual, hSnowActual, TSurfIn, TempFrz, ug,
    SWDown, LWDown, ATemp, aqh):

    ##### define local constants used for calculations #####

    # coefficients for the saturation vapor pressure equation
    aa1 = 2663.5
    aa2 = 12.537
    bb1 = 0.622
    bb2 = 1 - bb1
    Ppascals = 100000
    lnTen = np.log(10)
    cc0 = 10**aa2
    cc1 = cc0 * aa1 * bb1 * Ppascals * lnTen
    cc2 = cc0 * bb2
    
    # sensible heat constant
    d1 = seaice_dalton * cpAir * rhoAir
    # ice latent heat constant
    d1i = seaice_dalton * lhSublim * rhoAir

    # melting temperature of ice
    Tmelt = celsius2K

    # temperature threshold for when to use wet albedo
    SurfMeltTemp = Tmelt + wetAlbTemp 


    ##### initializations #####

    TSurfOut = TSurfIn.copy()
    F_ia = np.zeros(iceMask.shape)
    #F_ia_net
    F_io_net = np.zeros(iceMask.shape)

    # the shortwave radiative flux at the ocean-ice interface (+ = upwards)
    IcePenetSW = np.zeros(iceMask.shape)

    # reshwater flux due to sublimation [kg/m2] (+ = upward)
    FWsublim = np.zeros(iceMask.shape)
        
    # effective conductivity of ice and snow combined
    effConduct = np.zeros(iceMask.shape)

    # derivative of F_ia w.r.t snow/ice surf. temp
    dFia_dTs = np.zeros(iceMask.shape)

    # shortwave radiative flux convergence in the sea ice
    absorbedSW = np.zeros(iceMask.shape)

    # saturation vapor pressure of snow/ice surface
    qhice = np.zeros(iceMask.shape)

    # derivative of qhice w.r.t snow/ice surf. temp
    dqh_dTs = np.zeros(iceMask.shape)

    # latent heat flux (sublimation) (+ = upward)
    F_lh = np.zeros(iceMask.shape)

    # upward long-wave surface heat flux (+ = upward)
    F_lwu = np.zeros(iceMask.shape)

    # sensible surface heat flux (+ = upward)
    F_sens = np.zeros(iceMask.shape)

    # conductive heat flux through ice and snow (+ = upward)
    F_c = np.zeros(iceMask.shape)

    # make local copies of downward longwave radiation, surface
    # and atmospheric temperatures
    TSurfLoc = TSurfIn.copy()
    # TSurfLoc = np.minimum(celsius2k + maxTIce, TSurfIn)
    LWDownLocBound = np.maximum(minLwDown, LWDown)
    ATempLoc = np.maximum(celsius2K + minTAir, ATemp)


    ##### determine fixed (relative to surface temperature) forcing term in heat budget #####

    isIce = np.where(hIceActual > 0)
    isSnow = np.where(hSnowActual > 0)

    d3 = np.ones((sNy+2*OLy,sNx+2*OLx)) * iceEmiss * stefBoltz
    d3[isSnow] = snowEmiss * stefBoltz

    LWDownLoc = iceEmiss * LWDownLocBound
    LWDownLoc[isSnow] = snowEmiss * LWDownLocBound[isSnow]


    ##### determine albedo #####

    albIce = np.zeros_like(iceMask)
    albSnow = np.zeros_like(iceMask)
    alb = np.zeros_like(iceMask)

    # use albedo of dry surface (if ice is present)
    albIce[isIce] = dryIceAlb
    albSnow[isIce] = drySnowAlb

    # use albedo of wet surface if surface is thawing
    useWetAlb = np.where(TSurfLoc[isIce] >= SurfMeltTemp)
    albIce[isIce][useWetAlb] = dryIceAlb
    albSnow[isIce][useWetAlb] = drySnowAlb

    # same for southern hermisphere
    south = np.where(fCori[isIce] < 0)
    albIce[isIce][south] = dryIceAlb_south
    albSnow[isIce][south] = drySnowAlb_south
    useWetAlb_south = np.where(TSurfLoc[isIce][south] >= SurfMeltTemp)
    albIce[isIce][south][useWetAlb_south] = dryIceAlb
    albSnow[isIce][south][useWetAlb_south] = drySnowAlb

    # if the snow thickness is smaller than hCut, use linear transition
    # between ice and snow albedo
    alb[isIce] = albIce[isIce] + hSnowActual[isIce] / hCut * (
        albSnow[isIce] - albIce[isIce])

    # is the snow thickness is larger than hCut, the snow is opaque for
    # shortwave radiation -> use snow albedo
    alb[isIce] = np.where(hSnowActual[isIce] > hCut, albSnow[isIce], alb[isIce])

    # if no snow is present, use ice albedo
    alb[isIce] = np.where(hSnowActual[isIce] == 0, albIce[isIce], alb[isIce])


    ##### determine the shortwave radiative flux arriving at the     #####
    #####  ice-ocean interface after scattering through snow and ice #####

    # if snow is present, all radiation is absorbed in the ice
    penetSWFrac = np.zeros_like(iceMask)
    penetSWFrac[isIce] = shortwave * np.exp(-1.5 * hIceActual[isIce])
    penetSWFrac[isSnow] = 0

    # shortwave radiative flux at the ocean-ice interface (+ = upward)
    IcePenetSW[isIce] = -(1 - alb[isIce]) * penetSWFrac[isIce] * SWDown[isIce]

    # shortwave radiative flux convergence in the ice
    absorbedSW[isIce] = (1 - alb[isIce]) * (1 - penetSWFrac[isIce]) * SWDown[isIce]
    
    # effective conductivity of the snow-ice system
    effConduct[isIce] = iceConduct * snowConduct / (
        snowConduct * hIceActual[isIce] + iceConduct * hSnowActual[isIce])


    ##### calculate the heat fluxes #####

    def fluxes(t1):

        t2 = t1 * t1
        t3 = t2 * t1
        t4 = t2 * t2

        # calculate the saturation vapor pressure in the snow/ ice-atmosphere boundary layer
        mm_log10pi = - aa1 / t1 + aa2
        mm_pi = 10**mm_log10pi

        qhice[isIce] = bb1 * mm_pi / (Ppascals - (1 - bb1) * mm_pi)
        cc3t = 10**(aa1 / t1)
        dqh_dTs[isIce] = cc1 * cc3t / ((cc2 - cc3t * Ppascals)**2 * t2)

        # calculate the fluxes based on the surface temperature
        F_c[isIce] = effConduct[isIce] * (TempFrz[isIce] + celsius2K - t1)
        F_lh[isIce] = d1i * ug[isIce] * (qhice[isIce] - aqh[isIce])
        F_lwu[isIce] = t4 * d3[isIce]
        F_sens[isIce] = d1 * ug[isIce] * (t1 - ATempLoc[isIce])
        F_ia[isIce] = - LWDownLoc[isIce] - absorbedSW[isIce] + F_lwu[isIce] + F_sens[isIce] + F_lh[isIce]
        dFia_dTs[isIce] = 4 * d3[isIce] * t3 + d1 * ug[isIce] + d1i * ug[isIce] * dqh_dTs[isIce]

        return F_c, F_lh, F_ia, dFia_dTs

    # do a loop for the fluxes to cenverge
    for i in range(6):

        t1 = TSurfLoc[isIce]
        F_c, F_lh, F_ia, dFia_dTs = fluxes(t1)

        # update surface temperature as solution of
        # F_c = F_ia + d/dT (F_c - F_ia) * delta T
        TSurfLoc[isIce] = TSurfLoc[isIce] + (F_c[isIce] - F_ia[isIce]) \
            / (effConduct[isIce] + dFia_dTs[isIce])

        # add upper and lower boundary
        TSurfLoc = np.minimum(TSurfLoc, Tmelt)
        TSurfLoc = np.maximum(TSurfLoc, celsius2K + minTIce)

    # recalculate the fluxes based on the adjusted surface temperature
    t1 = TSurfLoc[isIce]
    F_c, F_lh, F_ia, dFia_dTs = fluxes(t1)

    # case 1: F_c <= 0
    # F_io_net is already set up as zero everywhere
    F_ia_net = F_ia.copy()
    # case 2: F_c > 0
    upCondflux = np.where(F_c > 0)
    F_io_net[upCondflux] = F_c[upCondflux]
    F_ia_net[upCondflux] = 0

    # save updated surface temperature as output and finalize the flux terms
    TSurfOut[isIce] = TSurfLoc[isIce]
    FWsublim[isIce] = F_lh[isIce] / lhSublim


    return TSurfOut, F_io_net, F_ia_net, F_ia, IcePenetSW, FWsublim