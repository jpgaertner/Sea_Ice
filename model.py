from veros import runtime_settings
from seaice_params import backend
runtime_settings.backend = backend

from veros import veros_routine, veros_kernel, KernelOutput

from seaice_reg_ridge import update_clean_up_advection, update_ridging
from dynamics_routines import update_IceStrength
from seaice_advection import update_Transport, update_Advection
from seaice_dynsolver import update_AreaWS, update_IceVelocities, \
            update_SeaIceMass, update_SurfaceForcing
from seaice_ocean_stress import update_OceanStress
from seaice_growth import update_Growth

from seaice_size import *


# get initial state
from initialize import state


@veros_routine
def model(state):

    # calculate sea ice mass centered around c, u, v points
    update_SeaIceMass(state)

    # calculate surface forcing due to wind
    update_SurfaceForcing(state)

    # calculate ice strength
    update_IceStrength(state)

    # calculate sea ice cover fraction centered around u,v points
    update_AreaWS(state)

    # calculate ice velocities
    update_IceVelocities(state)

    # calculate stresses on ocean surface
    update_OceanStress(state)

    # calculate transport through tracer cells
    update_Transport(state)

    import matplotlib.pyplot as plt

    # fig, axs = plt.subplots(1,2, figsize=(14,6))
    # axs[0].pcolormesh(state.variables.uIce[olx:-olx,oly:-oly])
    # axs[0].set_title('uIce')
    # axs[1].pcolormesh(state.variables.vIce[olx:-olx,oly:-oly])
    # axs[1].set_title('vIce')
    # plt.show()

    # calculate change in sea ice fields due to advection
    update_Advection(state)

    # correct overshoots and other pathological cases after advection
    update_clean_up_advection(state)

    # cut off ice cover fraction at 1 after advection
    update_ridging(state)

    # calculate thermodynamic ice growth
    update_Growth(state)


for i in range(5):
    model(state)

import matplotlib.pyplot as plt

plt.pcolormesh(state.variables.hIceMean[olx:-olx,oly:-oly])
plt.colorbar()
plt.title('hIceMean')
plt.show()