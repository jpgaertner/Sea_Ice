from veros import runtime_settings
backend = 'jax' # flag which backend to use (numpy or jax)
runtime_settings.backend = backend


import matplotlib.pyplot as plt

from model import model
from seaice_size import *

from veros.core.operators import numpy as npx

from initialize import state
from time import time
#state = get_state()

#model(state)
start = time()

# 1441 (2days)

for i in range(180):
    print(i)
    model(state)

end = time()

print('runtime =', end - start)



# save model results
ice = state.variables.hIceMean
snow = state.variables.hSnowMean
area = state.variables.Area
uIce = state.variables.uIce
vIce = state.variables.vIce
tauX = state.variables.tauX
tauY = state.variables.tauY

results = [ice, snow, area, uIce, vIce, tauX, tauY]
npx.save('results_8000', results)

# fig, axs = plt.subplots(2,2, figsize=(9, 6.5))
# ax0 = axs[0,0].pcolormesh(state.variables.uWind)
# axs[0,0].set_title('uWind')
# ax1 = axs[1,0].pcolormesh(state.variables.vWind)
# axs[1,0].set_title('vWind')
# ax2 = axs[0,1].pcolormesh(state.variables.uVel)
# axs[0,1].set_title('uVel')
# ax3 = axs[1,1].pcolormesh(state.variables.vVel)
# axs[1,1].set_title('vVel')

# plt.colorbar(ax0, ax=axs[0,0])
# plt.colorbar(ax1, ax=axs[1,0])
# plt.colorbar(ax2, ax=axs[0,1])#
# plt.colorbar(ax3, ax=axs[1,1])

fig, axs = plt.subplots(2,2, figsize=(8,6))
ax0 = axs[0,0].pcolormesh(ice[oly:-oly,olx:-olx])
axs[0,0].set_title('ice thickness')
ax1 = axs[1,0].pcolormesh(area[oly:-oly,olx:-olx])
axs[1,0].set_title('Area')
ax2 = axs[0,1].pcolormesh(uIce[oly:-oly,olx:-olx])
axs[0,1].set_title('uIce')
ax3 = axs[1,1].pcolormesh(vIce[oly:-oly,olx:-olx])
axs[1,1].set_title('vIce')

plt.colorbar(ax0, ax=axs[0,0])
plt.colorbar(ax1, ax=axs[1,0])
plt.colorbar(ax2, ax=axs[0,1])
plt.colorbar(ax3, ax=axs[1,1])

fig.tight_layout()
#plt.show()