import numpy as np

from seaice_size import *
from seaice_params import *

from seaice_solve4temp import solve4temp

import matplotlib.pyplot as plt

hIceActual = np.ones((nx,ny))*0.5
hSnowActual = np.ones((nx,ny))*0
TSurf = celsius2K*np.ones((nx,ny))
ug = np.ones((nx,ny))*2
SWDown = np.ones((nx,ny))*0
LWDown = np.ones((nx,ny))*50
ATemp = celsius2K*np.ones((nx,ny))-20
aqh = np.ones((nx,ny))*0.0001
TempFrz = celsius2K*np.ones((nx,ny))-2

T = np.array([TSurf[0,0]])
Fio = np.array([])
Fia = np.array([])

x = [i for i in range(0,365)]
for i in range(1,365):
    TSurf, F_io_net, F_ia_net, F_ia, IcePenetSW, FWsublim = solve4temp(hIceActual, hSnowActual, TSurf, TempFrz, ug, SWDown, LWDown, ATemp, aqh)
    T = np.append(T,TSurf[0,0])
    Fio = np.append(Fio,F_io_net[0,0])
    Fia = np.append(Fia,F_ia_net[0,0])

#print(TSurf, F_io_net, F_ia_net, F_ia, IcePenetSW, FWsublim)
#print(T)
#print(Fio)
#print(Fia)
plt.plot(x,T)

#print(x)
#plt.plot(x,T)
plt.show()