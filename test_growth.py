
# hIceMean: mean ice thickness [m3/m2] (= hIceActual * Area with hIceActual = Vol_ice / x_len y_len)
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
# Qnet: surface net heat flux out of the ocean (open water or under ice) [W/m2]
# Qsw: surface shortwave heat flux out of the ocean (open water or under ice) [W/m2]
# SWDown: shortwave downward radiation [W/m2]
# LWDown: longwave downward radiation [W/m2]
# ATemp: atmospheric temperature [K]
# aqh: atmospheric specific humidity [g/kg]

from veros import runtime_settings
backend = 'jax' # flag which backend to use (numpy or jax)
runtime_settings.backend = backend

from veros import veros_routine
from veros.core.operators import numpy as npx

from seaice_size import *
from seaice_params import *

from seaice_growth import update_Growth

import matplotlib.pyplot as plt

from initialize import state

ice = npx.array([state.variables.hIceMean[0,0]])
snow = npx.array([state.variables.hSnowMean[0,0]])
iceTemp = npx.array([state.variables.TIceSnow[0,0,0]])
area = npx.array([state.variables.Area[0,0]])
days = npx.array([0])

timesteps = 30

for i in range(timesteps):

    update_Growth(state)

    ice = npx.append(ice, state.variables.hIceMean[0,0])
    snow = npx.append(snow, state.variables.hSnowMean[0,0])
    iceTemp = npx.append(iceTemp, state.variables.TIceSnow[0,0,0])
    area = npx.append(area, state.variables.Area[0,0])
    days = npx.append(days,i)

# fig, axs = plt.subplots(2,2, figsize=(10,6))
# axs[0,0].plot(days, ice)
# axs[0,0].set_ylabel("Ice Thickness")
# axs[0,1].plot(days, snow)
# axs[0,1].set_ylabel("Snow Thickness")
# axs[1,0].plot(days, area)
# axs[1,0].set_ylabel("Area")
# axs[1,1].plot(days, iceTemp)
# axs[1,1].set_ylabel("Ice Temperature")

# fig.tight_layout()
# plt.show()