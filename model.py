from veros import runtime_settings
from seaice_params import backend
runtime_settings.backend = backend

from veros import veros_routine, veros_kernel, KernelOutput

from seaice_advection import update_Transport, update_Advection
from seaice_dynsolver import update_IceVelocities, update_SeaIceMass, update_SurfaceForcing
from seaice_ocean_stress import update_OceanStress

from seaice_size import *


# get initial state
from initialize import state

@veros_routine
def model(state):

    # calculate sea ice mass centered around c, u, v points
    update_SeaIceMass(state)

    # calculate surface forcing due to wind
    update_SurfaceForcing(state)

    # calculate ice velocities
    update_IceVelocities(state)

    # calculate stresses on ocean surface
    update_OceanStress(state)

    # calculate transport through tracer cells
    update_Transport(state)

    # calculate change in sea ice fields due to advection
    update_Advection(state)



model(state)

import matplotlib.pyplot as plt

plt.pcolormesh(state.variables.hIceMean[olx:-olx,oly:-oly])
plt.colorbar()
plt.show()