
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

hIceMean = np.ones((sNx,sNy))*1
hSnowMean = np.ones((sNx,sNy))*0
hIceMeanMask = np.ones((sNx,sNy))*1
Area = np.ones((sNx,sNy))*1
salt = np.ones((sNx,sNy))*35
TIceSnow = np.ones((sNx,sNy,nITC))*celsius2K
precip = np.ones((sNx,sNy))*0 #order 1e-6
snowPrecip = np.ones((sNx,sNy))*0
evap = np.ones((sNx,sNy))*0
runoff = np.ones((sNx,sNy))*0
wspeed = np.ones((sNx,sNy))*2
theta = np.ones((sNx,sNy))*celsius2K-1.7
Qnet = np.ones((sNx,sNy))*50
Qsw = np.ones((sNx,sNy))*0 #winter condition (summer up to 500)
SWDown = np.ones((sNx,sNy))*0 #winter
LWDown = np.ones((sNx,sNy))*50 #winter
ATemp = np.ones((sNx,sNy))*celsius2K-20
aqh = np.ones((sNx,sNy))*1e-4


x = np.array([0])
ice = np.array([hIceMean[0,0]])
snow = np.array([hSnowMean[0,0]])
qnet = np.array([Qnet[0,0]])
iceTemp = np.array([TIceSnow[0,0,:]])
area_arr = np.array([Area[0,0]])
qsw = np.array([Qsw[0,0]])

for i in range(365):
    hIceMean, hSnowMean, Area, TIceSnow, saltflux, EvPrecRun, Qsw, Qnet, seaIceLoad = (
        growth(hIceMean, hIceMeanMask, hSnowMean, Area, salt, TIceSnow, precip, snowPrecip, evap, 
        runoff, wspeed, theta, Qnet, Qsw, SWDown, LWDown, ATemp, aqh)    )

    x = np.append(x, i)
    ice = np.append(ice, hIceMean[0,0])
    snow = np.append(snow, hSnowMean[0,0])
    qnet = np.append(qnet, Qnet[0,0])
    iceTemp = np.append(iceTemp, TIceSnow[0,0,:])
    area_arr = np.append(area_arr, Area[0,0])
    qsw = np.append(qsw, Qsw[0,0])


fig, axs = plt.subplots(3,2, figsize=(10,6))
axs[0,0].plot(x,ice)
axs[0,0].set_ylabel("Ice Thickness")
axs[0,1].plot(x,snow)
axs[0,1].set_ylabel("Snow Thickness")
axs[1,0].plot(x,area_arr)
axs[1,0].set_ylabel("Area")
axs[1,1].plot(x,qnet)
axs[1,1].set_ylabel("Qnet")
axs[2,0].plot(x,iceTemp)
axs[2,0].set_ylabel("Ice Temperature")
axs[2,1].plot(x,qsw)
axs[2,1].plot("Qsw")
fig.tight_layout()
plt.show()
#print(x)
#print(f"ice:{ice}")
#print(f"snow:{snow}")
#print(f"qnet:{qnet}")
#print(f"iceTemp:{iceTemp}")