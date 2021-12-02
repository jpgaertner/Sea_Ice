import numpy as np

from seaice_size import *
from seaice_params import *

from seaice_solve4temp import solve4temp

import matplotlib.pyplot as plt

hIceActual = np.ones((sNx,sNy))*0.3
hSnowActual = np.ones((sNx,sNy))*0
TSurf = celsius2K*np.ones((sNx,sNy))-5
ug = np.ones((sNx,sNy))*2
SWDown = np.ones((sNx,sNy))*0
LWDown = np.ones((sNx,sNy))*70
ATemp = celsius2K*np.ones((sNx,sNy))-30
aqh = np.ones((sNx,sNy))*0.002
TempFrz = celsius2K*np.ones((sNx,sNy))

T = np.array([TSurf[0,0]])
Fio = np.array([])
Fia = np.array([])

x = [i for i in range(0,3)]
for i in range(1,10):
    TSurf, F_io_net, F_ia_net, F_ia, IcePenetSW, FWsublim = solve4temp(hIceActual, hSnowActual, TSurf, TempFrz, ug, SWDown, LWDown, ATemp, aqh)
    T = np.append(T,TSurf[0,0])
    Fio = np.append(Fio,F_io_net[0,0])
    Fia = np.append(Fia,F_ia_net[0,0])

#print(TSurf, F_io_net, F_ia_net, F_ia, IcePenetSW, FWsublim)
print(T)
print(Fio)
print(Fia)
#print(x)
#plt.plot(x,T)
#plt.show()