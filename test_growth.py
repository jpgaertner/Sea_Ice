
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

import numpy as np
from seaice_size import *
from seaice_params import *
from seaice_growth import growth
import matplotlib.pyplot as plt

# # for the 1d test
# nx = 1
# ny = 1
# olx = 2
# oly = 2



hIceMean = np.ones((nx+2*olx,ny+2*oly)) * 1.3 * iceMask
Area = np.ones((nx+2*olx,ny+2*oly))* 0.9 * iceMask
hSnowMean = np.ones((nx+2*olx,ny+2*oly)) * 0.1 * iceMask

salt = np.ones((nx+2*olx,ny+2*oly))*29
TIceSnow = np.ones((nx+2*olx,ny+2*oly,nITC)) * 248.7569580078125
precip = np.ones((nx+2*olx,ny+2*oly))*0 #order 1e-6
snowPrecip = np.ones((nx+2*olx,ny+2*oly))*0
evap = np.ones((nx+2*olx,ny+2*oly))*0
runoff = np.ones((nx+2*olx,ny+2*oly))*0
wspeed = np.ones((nx+2*olx,ny+2*oly))*2
theta = np.zeros((nx+2*olx,ny+2*oly)) -1.66 #-1.96
Qnet = np.ones((nx+2*olx,ny+2*oly))* 173.03212617345582#29.694019940648037
Qsw = np.ones((nx+2*olx,ny+2*oly))*0 #winter condition
SWDown = np.ones((nx+2*olx,ny+2*oly))*0 #winter
LWDown = np.ones((nx+2*olx,ny+2*oly))*180 #20 winter
ATemp = np.ones((nx+2*olx,ny+2*oly))* 253
aqh = np.ones((nx+2*olx,ny+2*oly))*0#1e-4

os_hIceMean = np.zeros_like(iceMask)
os_hSnowMean = np.zeros_like(iceMask)

ice = np.array([hIceMean[0,0]])
snow = np.array([hSnowMean[0,0]])
qnet = np.array([Qnet[0,0]])
iceTemp = np.array([TIceSnow[0,0,:]])
area = np.array([Area[0,0]])
qsw = np.array([Qsw[0,0]])
days = np.array([0])

timesteps = 60*12

for i in range(timesteps):
    hIceMean, hSnowMean, Area, TIceSnow, saltflux, EvPrecRun, Qsw_out, Qnet_out, seaIceLoad = (
        growth(hIceMean, hSnowMean, Area, os_hIceMean, os_hSnowMean, salt, TIceSnow, precip, snowPrecip, evap, 
        runoff, wspeed, theta, Qnet, Qsw, SWDown, LWDown, ATemp, aqh))

    ice = np.append(ice, hIceMean[0,0])
    snow = np.append(snow, hSnowMean[0,0])
    qnet = np.append(qnet, Qnet[0,0])
    iceTemp = np.append(iceTemp, TIceSnow[0,0,:])
    area = np.append(area, Area[0,0])
    qsw = np.append(qsw, Qsw[0,0])
    days = np.append(days,i)

import matplotlib.pyplot as plt
plt.contourf(hIceMean[oly:-oly,olx:-olx])
plt.colorbar()
plt.show()

# fig, axs = plt.subplots(2,2, figsize=(10,6))
# axs[0,0].plot(ice)
# axs[0,0].set_ylabel("Ice Thickness")
# axs[0,1].plot(snow)
# axs[0,1].set_ylabel("Snow Thickness")
# axs[1,0].plot(area)
# axs[1,0].set_ylabel("Area")
# axs[1,1].plot(iceTemp)
# axs[1,1].set_ylabel("Ice Temperature")

# fig.tight_layout()
# plt.show()

# print(ice[timesteps])
# print(area[timesteps])
# print(snow[timesteps])
#print(iceTemp[timesteps])
