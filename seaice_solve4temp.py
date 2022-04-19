# Calculates heat fluxes and temperature of the ice surface
# (see Hibler, MWR, 108, 1943-1973, 1980)

from veros.core.operators import numpy as npx
from veros import veros_kernel

from seaice_size import *
from seaice_params import *

# # for the 1d test
# nx = 1
# ny = 1
# olx = 2
# oly = 2

### input
# hIceActual: actual ice thickness [m]
# hSnowActual: actual snow thickness [m]
# TSurfIn: surface temperature of ice/snow [K]
# TempFrz: freezing temperature [K]
# ug: atmospheric geostrophic wind speed [m/s]
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


@veros_kernel
def solve4temp(state, hIceActual, hSnowActual, TSurfIn, TempFrz):

    ##### define local constants used for calculations #####

    # coefficients for the saturation vapor pressure equation
    aa1 = 2663.5
    aa2 = 12.537
    bb1 = 0.622
    bb2 = 1 - bb1
    Ppascals = 100000
    cc0 = 10**aa2
    cc1 = cc0 * aa1 * bb1 * Ppascals * npx.log(10)
    cc2 = cc0 * bb2
    
    # sensible heat constant
    d1 = seaice_dalton * cpAir * rhoAir
    # latent heat constant
    d1i = seaice_dalton * lhSublim * rhoAir

    # melting temperature of ice
    Tmelt = celsius2K

    # temperature threshold for when to use wet albedo
    SurfMeltTemp = Tmelt + wetAlbTemp 

    # make local copies of downward longwave radiation, surface
    # and atmospheric temperatures
    TSurfLoc = TSurfIn
    LWDownLocCapped = npx.maximum(minLwDown, state.variables.LWDown)
    ATempLoc = npx.maximum(celsius2K + minTAir, state.variables.ATemp)

    # set geostrophic wind speed
    ug = npx.maximum(eps, state.variables.wSpeed)


    ##### determine forcing term in heat budget #####

    # returns arrays with the size of hIceActual with True or False values
    isIce = (hIceActual > 0)
    isSnow = (hSnowActual > 0)

    d3 = npx.where(isSnow, snowEmiss, iceEmiss) * stefBoltz

    LWDownLoc = npx.where(isSnow, snowEmiss, iceEmiss) * LWDownLocCapped


    ##### determine albedo #####

    # use albedo of dry surface (if ice is present)
    albIce = npx.where(isIce, dryIceAlb, 0)
    albSnow = npx.where(isIce, drySnowAlb, 0)

    # use albedo of wet surface if surface is thawing
    useWetAlb = ((hIceActual > 0) & (TSurfLoc >= SurfMeltTemp))
    albIce = npx.where(useWetAlb, wetIceAlb, albIce)
    albSnow = npx.where(useWetAlb, wetSnowAlb, albSnow)

    # same for southern hermisphere
    south = ((hIceActual > 0) & (fCori < 0))
    albIce = npx.where(south, dryIceAlb_south, albIce)
    albSnow = npx.where(south, drySnowAlb_south, albSnow)
    useWetAlb_south = ((hIceActual > 0) & (fCori < 0) & (TSurfLoc >= SurfMeltTemp))
    albIce = npx.where(useWetAlb_south, wetIceAlb_south, albIce)
    albSnow = npx.where(useWetAlb_south, wetSnowAlb_south, albSnow)

    # if the snow thickness is smaller than hCut, use linear transition
    # between ice and snow albedo
    alb = npx.where(isIce, albIce + hSnowActual / hCut * (albSnow - albIce), 0)

    # if the snow thickness is larger than hCut, the snow is opaque for
    # shortwave radiation -> use snow albedo
    alb = npx.where(hSnowActual > hCut, albSnow, alb)

    # if no snow is present, use ice albedo
    alb = npx.where(hSnowActual == 0, albIce, alb)


    ##### determine the shortwave radiative flux arriving at the     #####
    #####  ice-ocean interface after scattering through snow and ice #####

    # the fraction of shortwave radiative flux that arrives at the ocean
    # surface after passing the ice
    penetSWFrac = npx.where(isIce, shortwave * npx.exp(-1.5 * hIceActual), 0)

    # if snow is present, all radiation is absorbed
    penetSWFrac = npx.where(isSnow, 0, penetSWFrac)

    # shortwave radiative flux at the ocean-ice interface (+ = upward)
    IcePenetSW = npx.where(isIce, -(1 - alb) * penetSWFrac * state.variables.SWDown, 0)

    # shortwave radiative flux convergence in the ice
    absorbedSW = npx.where(isIce, (1 - alb) * (1 - penetSWFrac) * state.variables.SWDown, 0)
    
    # effective conductivity of the snow-ice system
    effConduct = npx.where(isIce, iceConduct * snowConduct / (
                    snowConduct * hIceActual + iceConduct * hSnowActual), 0)


    ##### calculate the heat fluxes #####

    def fluxes(t1):

        t2 = t1 * t1
        t3 = t2 * t1
        t4 = t2 * t2


        # saturation vapor pressure of snow/ice surface
        svp = 10**(- aa1 / t1 + aa2)

        # specific humidity at the surface
        q_s = npx.where(isIce, bb1 * svp / (Ppascals - (1 - bb1) * svp), 0)

        # derivative of q_s w.r.t snow/ice surface temperature
        cc3t = 10**(aa1 / t1)
        dqs_dTs = npx.where(isIce, cc1 * cc3t / ((cc2 - cc3t * Ppascals)**2 * t2), 0)

        # calculate the fluxes based on the surface temperature

        # conductive heat flux through ice and snow (+ = upward)
        F_c  = npx.where(isIce, effConduct * (TempFrz - t1), 0)

        # latent heat flux (sublimation) (+ = upward)
        F_lh = npx.where(isIce, d1i * ug * (q_s - state.variables.aqh), 0)

        # long-wave surface heat flux (+ = upward)
        F_lwu = npx.where(isIce, t4 * d3, 0)

        # sensible surface heat flux (+ = upward)
        F_sens = npx.where(isIce, d1 * ug * (t1 - ATempLoc), 0)

        # upward seaice/snow surface heat flux to atmosphere
        F_ia = npx.where(isIce, (- LWDownLoc - absorbedSW + F_lwu
                                + F_sens + F_lh), 0)

        # derivative of F_ia w.r.t snow/ice surf. temp
        dFia_dTs = npx.where(isIce, 4 * d3 * t3 + d1 * ug
                                    + d1i * ug * dqs_dTs, 0)

        return F_c, F_lh, F_ia, dFia_dTs

    # iterate for the temperatue to converge (Newton-Raphson method)
    for i in range(6):

        F_c, F_lh, F_ia, dFia_dTs = fluxes(TSurfLoc)

        # update surface temperature as solution of
        # F_c = F_ia + d/dT (F_c - F_ia) * delta T
        TSurfLoc = npx.where(isIce, TSurfLoc + (F_c - F_ia)
                                    / (effConduct + dFia_dTs), 0)

        # add upper and lower boundary
        TSurfLoc = npx.minimum(TSurfLoc, Tmelt)
        TSurfLoc = npx.maximum(TSurfLoc, celsius2K + minTIce)

    # recalculate the fluxes based on the adjusted surface temperature
    F_c, F_lh, F_ia, dFia_dTs = fluxes(TSurfLoc)

    # set net ocean-ice flux and surface heat flux divergence based on
    # the direction of the conductive heat flux
    upCondFlux = (F_c > 0)
    F_io_net = npx.where(upCondFlux, F_c, 0)
    F_ia_net = npx.where(upCondFlux, 0, F_ia)

    # save updated surface temperature as output
    TSurfOut = npx.where(isIce, TSurfLoc, TSurfIn)

    # freshwater flux due to sublimation [kg/m2] (+ = upward)
    FWsublim = npx.where(isIce, F_lh / lhSublim, 0)


    return TSurfOut, F_io_net, F_ia_net, F_ia, IcePenetSW, FWsublim