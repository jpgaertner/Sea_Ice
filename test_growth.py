from veros import runtime_settings
runtime_settings.backend = 'jax'

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

timesteps = 30*12

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