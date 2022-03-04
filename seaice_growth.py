from veros.core.operators import numpy as npx
from veros.core.operators import update, at

from seaice_size import *
from seaice_params import *

from seaice_solve4temp import solve4temp

# # for the 1d test
# sNx = 1
# sNy = 1
# OLx = 2
# OLy = 2

### input
# hIceMean: mean ice thickness [m3/m2] (= hIceActual * Area with Area the
#   sea ice cover fraction and hIceActual = Vol_ice / x_len y_len)
# iceMask: contains geometry of the set up
# hSnowMean: mean snow thickness [m3/m2]
# Area: sea ice cover fraction (0 <= Area <= 1)
# salt: surface salinity of the ocean [g/kg]
# TIceSnow: ice/ snow surface temp for each category [K]
# precip: precipition [m/s] (order 1e-6)
# snowPrecip: snowfall [m/s]
# evap: evaporation over ocean [m/s] (same order, =0 for now)
# runoff: runoff into ocean [m/s]   = 0 for now
# wspeed: wind speed [m/s]
# theta: potential temperature of the ocean surface [K]
# Qnet: net heat flux out of the ocean (open water or under ice) [W/m2]
# Qsw: shortwave heat flux out of the ocean (open water or under ice) [W/m2]
# SWDown: shortwave downward radiation [W/m2]
# LWDown: longwave downward radiation [W/m2]
# ATemp: atmospheric temperature [K]
# aqh: atmospheric specific humidity [g/kg]

### output
# hIceMean
# hSnowMean
# Area
# TIceSnow
# saltflux: salt flux into the ocean
# EmPmR: evaporation minus precipitation minus runoff (freshwater flux to ocean) - direction?
# Qnet
# Qsw
# seaIceLoad: sea ice + snow load on the sea surface


def growth(hIceMean, hSnowMean, Area, os_hIceMean, os_hSnowMean, salt,
        TIceSnow, precip, snowPrecip, evap, runoff, wspeed, theta, Qnet, 
        Qsw, SWDown, LWDown, ATemp, aqh):

    ##### constants and initializations #####

    # XXX: salt
    # ifdef ALLOW_SALT_PLUME ...

    # heat fluxes in [W/m2]

    # sea ice/snow surface heat flux to atmosphere (+ = upward)
    # F_ia

    # net heat flux divergence at the
    # sea ice/snow surface including sea ice conductive fluxes and
    # atmospheric fluxes
    # = 0: surface heat loss is balanced by upward conductive fluxes
    # < 0: net heat flux convergence at the ice/ snow surface
    # F_ia_net

    # the net heat flux divergence at the sea ice/snow
    # surface before snow is melted with any convergence
    # < 0: some snow (if present) will melt
    # F_ia_net_before_snow

    # net upward conductive heat flux
    # through sea ice and snow realized at the sea ice/snow surface (+ = upward)
    # F_io_net

    # heat flux from atmosphere to ocean (+ = upward)
    # F_ao

    # heat flux from ocean to the ice (change of mixed layer temperature) (+ = upward)
    # F_oi

    # freshwater flux due to sublimation [kg/m2] (+ = upward)
    # FWsublim

    hIceActual_mult = npx.zeros((*iceMask.shape,nITC))
    hSnowActual_mult = npx.zeros((*iceMask.shape,nITC))
    F_io_net_mult = npx.zeros((*iceMask.shape,nITC))
    F_ia_net_mult = npx.zeros((*iceMask.shape,nITC))
    F_ia_mult = npx.zeros((*iceMask.shape,nITC))
    qswi_mult = npx.zeros((*iceMask.shape,nITC))
    FWsublim_mult = npx.zeros((*iceMask.shape,nITC))

    # constants for converting heat fluxes into growth rates
    qi = 1 / (rhoIce * lhFusion)
    qs = 1 / (rhoSnow * lhFusion)


    ##### save ice, snow thicknesses and area prior to thermodynamic #####
    #####   changes and regularize thicknesses                       #####

    # set lower boundaries
    hIceMean = npx.maximum(hIceMean, 0)
    Area = npx.maximum(Area, 0)

    # the fields Area, hSnowMean, hIceMean are set to zero where noIce is True
    noIce = ((hIceMean == 0) | (Area == 0))
    Area *= ~noIce
    hSnowMean *= ~noIce
    hIceMean *= ~noIce

    # store mean ice and snow thickness and sea ice cover fraction
    # (prior to any thermodynamical changes)
    hIceMeanpreTH = hIceMean.copy()
    hSnowMeanpreTH = hSnowMean.copy()
    AreapreTH = Area.copy()

    # compute actual ice and snow thickness using the regularized area.
    # ice or snow thickness divided by Area does not work if Area -> 0,
    # therefore the regularization

    isIce = (hIceMeanpreTH > 0)
    regAreaSqrt =  npx.sqrt(AreapreTH**2 + area_reg_sq)
    recip_regAreaSqrt = 1 / regAreaSqrt

    hIceActual = npx.where(isIce, hIceMeanpreTH * recip_regAreaSqrt, 0)
    recip_hIceActual = AreapreTH / npx.sqrt(hIceMeanpreTH**2 + hice_reg_sq)
    hSnowActual = npx.where(isIce, hSnowMeanpreTH * recip_regAreaSqrt, 0)

    # add lower boundary
    hIceActual = npx.maximum(hIceActual, 0.05)

    ##### retrieve the air-sea heat and shortwave radiative fluxes and #####
    #####   calculate the corresponding ice growth rate for open water #####

    # set wind speed
    ug = npx.maximum(eps, wspeed)

    # set fluxed in (qswo) and out (F_ao) of the ocean
    F_ao = Qnet.copy()
    qswo = Qsw.copy()

    # the fraction of shortwave radiation that passes the ocean surface layer
    swFracPassTopOcean = 0

    # XXX: maybe as output for the ocean model
    # qswo_below_first_layer = qswo * swFracPassTopOcean
    qswo_in_first_layer = qswo * (1 - swFracPassTopOcean)

    # IceGrowthRateOpenWater is also defined for Area > 0 as the area can
    # be only partially covered with ice
    IceGrowthRateOpenWater = qi * (F_ao - qswo + qswo_in_first_layer)


    ##### calculate surface temperature and heat fluxes ##### 

    # record prior ice surface temperatures
    TIce_mult = TIceSnow.copy()

    for l in range(0, nITC):
        # set relative thickness of ice and snow categories
        pFac = (2 * (l + 1) - 1) * recip_nITC
        # find actual snow and ice thickness within each category
        hIceActual_mult = update(hIceActual_mult, at[:,:,l], hIceActual * pFac)
        hSnowActual_mult = update(hSnowActual_mult, at[:,:,l], hSnowActual * pFac)

    # calculate freezing temperature
    TempFrz = tempFrz0 + dTempFrz_dS * salt

    # calculate heat fluxes and ice/ snow surface temperature
    for l in range(nITC):
        output = solve4temp(hIceActual_mult[:,:,l], hSnowActual_mult[:,:,l],
            TIce_mult[:,:,l], TempFrz, ug, SWDown, LWDown, ATemp, aqh)

        TIce_mult = update(TIce_mult, at[:,:,l], output[0])
        F_io_net_mult = update(F_io_net_mult, at[:,:,l], output[1])
        F_ia_net_mult = update(F_ia_net_mult, at[:,:,l], output[2])
        F_ia_mult = update(F_ia_mult, at[:,:,l], output[3])
        qswi_mult = update(qswi_mult, at[:,:,l], output[4])
        FWsublim_mult = update(FWsublim_mult, at[:,:,l], output[5])


    ##### evaluate precipitation as snow or rain #####

    # the precipitation rate over the ice which goes immediately into the
    # ocean (flowing through cracks in the ice). if the temperature is
    # above the freezing point, the precipitation remains wet and runs
    # into the ocean
    PrecipRateOverIceSurfaceToSea = precip.copy()

    # if there is ice and the temperature is below the freezing point,
    # the precipitation falls and accumulates as snow
    tmp = ((AreapreTH > 0) & (TIce_mult[:,:,0] < celsius2K))

    # snow accumulation rate over ice [m/s]
    SnowAccRateOverIce = npx.where(tmp, snowPrecip * rhoFresh2rhoSnow, 0)

    PrecipRateOverIceSurfaceToSea *= ~tmp

    # total snow accumulation over ice [m]
    SnowAccOverIce = SnowAccRateOverIce * AreapreTH * deltaTtherm


    ##### for every thickness category, record the ice surface #####
    #####  temperature and find the average flux across it     #####

    # update surface temperature and fluxes
    # multplying the fluxes with the area changes them from mean fluxes
    # for the ice part of the cell to mean fluxes for the whole cell
    TIceSnow = TIce_mult
    for i in range(nITC):
        # XXX: maybe define a temperature for the case hIceMean = 0
        TIceSnow = update(TIceSnow, at[:,:,i], TIceSnow[:,:,i])
    F_io_net = npx.sum(F_io_net_mult*recip_nITC, axis=2) * AreapreTH
    F_ia_net = npx.sum(F_ia_net_mult*recip_nITC, axis=2) * AreapreTH
    #F_ia = npx.sum(F_ia_mult*recip_nITC, axis=2) * AreapreTH
    qswi = npx.sum(qswi_mult*recip_nITC, axis=2) * AreapreTH
    #FWsublim = npx.sum(FWsublim_mult*recip_nITC, axis=2) * AreapreTH


    ##### calculate growth rates of ice and snow #####

    # the ice growth rate beneath ice is given by the upward conductive
    # flux F_io_net and qi:
    IceGrowthRateUnderExistingIce = F_io_net * qi
    IceGrowthRateUnderExistingIce = npx.where(AreapreTH == 0,
                                        0, IceGrowthRateUnderExistingIce)

    # the potential snow melt rate if all snow surface heat flux
    # convergence (F_ia_net < 0) goes to melting snow [m/s]
    PotSnowMeltRateFromSurf = - F_ia_net * qs

    # the thickness of snow that can be melted in one time step:
    PotSnowMeltFromSurf = PotSnowMeltRateFromSurf * deltaTtherm

    # if the heat flux convergence could melt more snow than is actually
    # there, the excess is used to melt ice

    # case 1: snow will remain after melting, i.e. all of the heat flux
    # convergence will be used up to melt snow
    # case 2: all snow will be melted if the potential snow melt
    # height is larger or equal to the actual snow height. if there is
    # an excess of heat flux convergence after snow melting, it will
    # be used to melt ice

    allSnowMelted = (PotSnowMeltFromSurf >= hSnowMean)

    # the actual thickness of snow to be melted by snow surface
    # heat flux convergence [m]
    SnowMeltFromSurface = npx.where(allSnowMelted, hSnowMean, PotSnowMeltFromSurf)

    # the actual snow melt rate due to snow surface heat flux convergence [m/s]
    SnowMeltRateFromSurface = npx.where(allSnowMelted,
                SnowMeltFromSurface * recip_deltaTtherm,
                PotSnowMeltRateFromSurf)

    # the actual surface heat flux convergence used to melt snow [W/m2]
    SurfHeatFluxConvergToSnowMelt = npx.where(allSnowMelted,
                - hSnowMean * recip_deltaTtherm / qs, F_ia_net)

    # the surface heat flux convergence is reduced by the amount that
    # is used for melting snow:
    F_ia_net = F_ia_net - SurfHeatFluxConvergToSnowMelt

    # the remaining heat flux convergence is used to melt ice:
    IceGrowthRateFromSurface = F_ia_net * qi

    # the total ice growth rate is then:
    NetExistingIceGrowthRate = IceGrowthRateUnderExistingIce + IceGrowthRateFromSurface


    ##### calculate the heat fluxes from the ocean to the sea ice #####

    tmpscal0 = 0.4
    tmpscal1 = 7 / tmpscal0
    tmpscal2 = stantonNr * uStarBase * rhoConst * heatCapacity

    # the ocean temperature cannot be lower than the freezing temperature
    surf_theta = npx.maximum(theta, TempFrz)

    # mltf = mixed layer turbulence factor (determines how much of the temperature
    # difference is used for heat flux)
    mltf = 1 + (McPheeTaperFac - 1) / (1 + npx.exp((AreapreTH - tmpscal0) * tmpscal1))

    F_oi = - tmpscal2 * (surf_theta - TempFrz) * mltf
    IceGrowthRateMixedLayer = F_oi * qi


    ##### calculate change in ice, snow thicknesses and area #####

    # calculate thickness derivatives of ice and snow
    dhIceMean_dt = NetExistingIceGrowthRate * AreapreTH + \
        IceGrowthRateOpenWater * (1 - AreapreTH) + IceGrowthRateMixedLayer
    dhSnowMean_dt = (SnowAccRateOverIce - SnowMeltRateFromSurface) * AreapreTH

    # XXX: salt
    # ifdef allow_salt_plume ...

    tmpscal0 =  0.5 * recip_hIceActual

    # ice growth open water (due to ocean-atmosphere fluxes)
    # reduce ice cover if the open water growth rate is negative
    dArea_oaFlux = tmpscal0 * IceGrowthRateOpenWater * (1 - AreapreTH)

    # increase ice cover if the open water growth rate is positive
    tmp = ((IceGrowthRateOpenWater > 0) & (
            (AreapreTH > 0) | (dhIceMean_dt > 0)))
    dArea_oaFlux = npx.where(tmp, IceGrowthRateOpenWater
                             * (1 - AreapreTH), dArea_oaFlux)

    # multiply with lead closing factor
    dArea_oaFlux = npx.where((tmp & (fCori < 0)),
                                dArea_oaFlux * recip_h0_south, 
                                dArea_oaFlux * recip_h0)

    # ice growth mixed layer (due to ocean-ice fluxes)
    # (if the ocean is warmer than the ice IceGrowthRateMixedLayer > 0.
    # the supercooled state of the ocean is ignored/ does not lead to ice
    # growth. ice growth is only due to fluxes calculated by solve4temp)
    dArea_oiFlux = npx.where(IceGrowthRateMixedLayer <= 0,
        tmpscal0 * IceGrowthRateMixedLayer, 0)

    # ice growth over ice (due to ice-atmosphere fluxes)
    # (NetExistingIceGrowthRate leads to vertical and lateral melting but
    # only to vertical growing. lateral growing is covered by
    # IceGrowthRateOpenWater)
    dArea_iaFlux = npx.where((NetExistingIceGrowthRate <= 0) & (hIceMeanpreTH > 0),
        tmpscal0 * NetExistingIceGrowthRate * AreapreTH, 0)

    # total change in area
    dArea_dt = dArea_oaFlux + dArea_oiFlux + dArea_iaFlux


    ######  update ice, snow thickness and area #####

    Area = AreapreTH + dArea_dt * iceMask * deltaTtherm
    hIceMean = hIceMeanpreTH + dhIceMean_dt * iceMask * deltaTtherm
    hSnowMean = hSnowMeanpreTH + dhSnowMean_dt * iceMask * deltaTtherm

    # set boundaries:
    Area = npx.clip(Area, 0, 1)
    hIceMean = npx.maximum(hIceMean, 0)
    hSnowMean = npx.maximum(hSnowMean, 0)

    noIce = ((hIceMean == 0) | (Area == 0))
    Area *= ~noIce
    hSnowMean *= ~noIce
    hIceMean *= ~noIce

    # change of ice thickness due to conversion of snow to ice if snow
    # is submerged with water
    tmpscal0 = (hSnowMean * rhoSnow + hIceMean * rhoIce) * recip_rhoConst
    d_hIceMeanByFlood = npx.maximum(0, tmpscal0 - hIceMean)
    hIceMean = hIceMean + d_hIceMeanByFlood
    hSnowMean = hSnowMean - d_hIceMeanByFlood * rhoice2rhosnow


    ##### calculate output to ocean #####

    # effective shortwave heating rate
    Qsw = qswi * AreapreTH + qswo * (1 - AreapreTH)

    # the actual ice volume change over the time step [m3/m2]
    ActualNewTotalVolumeChange = hIceMean - hIceMeanpreTH

    # the net melted snow thickness [m3/m2] (positive if the snow
    # thickness decreases/ melting occurs)
    ActualNewTotalSnowMelt = hSnowMeanpreTH + SnowAccOverIce - hSnowMean

    # the energy required to melt or form the new ice volume [J/m2]
    EnergyInNewTotalIceVolume = ActualNewTotalVolumeChange / qi

    # the net energy flux out of the ocean [J/m2]
    NetEnergyFluxOutOfOcean = (AreapreTH * (F_ia_net + F_io_net + qswi)
                + (1 - AreapreTH) * F_ao) * deltaTtherm

    # energy taken out of the ocean which is not used for sea ice growth [J].
    # If the net energy flux out of the ocean is balanced by the latent
    # heat of fusion, the temperature of the mixed layer will not change
    ResidualEnergyOutOfOcean = NetEnergyFluxOutOfOcean - EnergyInNewTotalIceVolume

    # total heat flux out of the ocean [W/m2]
    Qnet = ResidualEnergyOutOfOcean * recip_deltaTtherm

    # the freshwater contribution to (from) the ocean from melting (growing)
    # ice [m3/m2] (positive for melting)
    FreshwaterContribFromIce = - ActualNewTotalVolumeChange * rhoIce2rhoFresh

    # in the case of non-zero ice salinity, the freshwater contribution
    # is reduced by the salinity ration of ice to water
    FreshwaterContribFromIce = npx.where(((salt > 0) & (salt > saltIce)),
                - ActualNewTotalVolumeChange * rhoIce2rhoFresh * (1 - saltIce/salt), 
                FreshwaterContribFromIce)

    # if the liquid cell has a lower salinity than the specified salinity
    # of sea ice, then assume the sea ice is completely fresh
    # (if the water is fresh, no salty sea ice can form)

    tmpscal0 = npx.minimum(saltIce, salt)
    saltflux = (ActualNewTotalVolumeChange + os_hIceMean) * tmpscal0 \
        * iceMask * rhoIce * recip_deltaTtherm

    # the freshwater contribution to the ocean from melting snow [m]
    FreshwaterContribFromSnowMelt = ActualNewTotalSnowMelt / rhoFresh2rhoSnow

    # evaporation minus precipitation minus runoff (freshwater flux to ocean)
    EmPmR = iceMask *  ((evap - precip) * (1 - AreapreTH) \
        - PrecipRateOverIceSurfaceToSea * AreapreTH - runoff - (
        FreshwaterContribFromIce + FreshwaterContribFromSnowMelt) \
            / deltaTtherm) * rhoFresh + iceMask * (
        os_hIceMean * rhoIce + os_hSnowMean * rhoSnow) \
            * recip_deltaTtherm

    # sea ice + snow load on the sea surface
    seaIceLoad = hIceMean * rhoIce + hSnowMean * rhoSnow
    # XXX: maybe introduce a cap if needed for ocean model


    return hIceMean, hSnowMean, Area, TIceSnow, saltflux, EmPmR, \
        Qsw, Qnet, seaIceLoad