
# hIceMean: mean ice thickness [m3/m2] (= hIceActual * Area with hIceActual = Vol_ice / x_len y_len)
# hIceMeanMask: contains geometry of the set up
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


hIceMean = np.ones((sNx+2*OLx,sNy+2*OLy))*1
hSnowMean = np.ones((sNx+2*OLx,sNy+2*OLy))*0
Area = np.ones((sNx+2*OLx,sNy+2*OLy))*1

hIceMeanMask = np.ones((sNx+2*OLx,sNy+2*OLy))*1
salt = np.ones((sNx+2*OLx,sNy+2*OLy))*29
TIceSnow = np.ones((sNx+2*OLx,sNy+2*OLy,nITC))*celsius2K
precip = np.ones((sNx+2*OLx,sNy+2*OLy))*0 #order 1e-6
snowPrecip = np.ones((sNx+2*OLx,sNy+2*OLy))*0
evap = np.ones((sNx+2*OLx,sNy+2*OLy))*0
runoff = np.ones((sNx+2*OLx,sNy+2*OLy))*0
wspeed = np.ones((sNx+2*OLx,sNy+2*OLy))*2
theta = np.ones((sNx+2*OLx,sNy+2*OLy))*celsius2K-1.96
Qnet = np.ones((sNx+2*OLx,sNy+2*OLy))* 153.536072
Qsw = np.ones((sNx+2*OLx,sNy+2*OLy))*0 #winter condition
SWDown = np.ones((sNx+2*OLx,sNy+2*OLy))*0 #winter
LWDown = np.ones((sNx+2*OLx,sNy+2*OLy))*20 #winter
ATemp = np.ones((sNx+2*OLx,sNy+2*OLy))*celsius2K-20.16
aqh = np.ones((sNx+2*OLx,sNy+2*OLy))*0#1e-4


ice = np.array([hIceMean[0,0]])
snow = np.array([hSnowMean[0,0]])
qnet = np.array([Qnet[0,0]])
iceTemp = np.array([TIceSnow[0,0,:]])
area = np.array([Area[0,0]])
qsw = np.array([Qsw[0,0]])
days = np.array([0])

timesteps = 30

# in F, runtime = 360d with deltat = 12h and dump frequency = 10d 
for i in range(timesteps):
    hIceMean, hSnowMean, Area, TIceSnow, saltflux, EvPrecRun, Qsw, Qnet_out, seaIceLoad = (
        growth(hIceMean, hIceMeanMask, hSnowMean, Area, salt, TIceSnow, precip, snowPrecip, evap, 
        runoff, wspeed, theta, Qnet, Qsw, SWDown, LWDown, ATemp, aqh)    )

    ice = np.append(ice, hIceMean[0,0])
    snow = np.append(snow, hSnowMean[0,0])
    qnet = np.append(qnet, Qnet[0,0])
    iceTemp = np.append(iceTemp, TIceSnow[0,0,:])
    area = np.append(area, Area[0,0])
    qsw = np.append(qsw, Qsw[0,0])
    days = np.append(days,(i+1)/2)


# fig, axs = plt.subplots(3,2, figsize=(10,6))
# axs[0,0].plot(ice)
# axs[0,0].set_ylabel("Ice Thickness")
# axs[0,1].plot(snow)
# axs[0,1].set_ylabel("Snow Thickness")
# axs[1,0].plot(area_arr)
# axs[1,0].set_ylabel("Area")
# axs[1,1].plot(qnet)
# axs[1,1].set_ylabel("Qnet")
# axs[2,0].plot(iceTemp)
# axs[2,0].set_ylabel("Ice Temperature")
# axs[2,1].plot(qsw)
# axs[2,1].plot("Qsw")
# fig.tight_layout()
# plt.show()


print(hIceMean)