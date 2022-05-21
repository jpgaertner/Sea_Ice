from veros import veros_routine

from model import model
from initialize import set_inits, var_meta, sett_meta


@veros_routine
def setup_seaice(state):
    set_inits(state)


@veros_routine
def main_seaice(state):
    model(state)





__VEROS_INTERFACE__ = dict(
    name = 'seaice_plugin',
    setup_entrypoint = setup_seaice,
    run_entrypoint = main_seaice,
    settings = sett_meta,
    variables = var_meta
)

# @veros_routine
# def seaice_main(state):
#     SIstate.variables.uVel  = state.variables.u
#     SIstate.variables.vVel  = state.variables.v
#     SIstate.variables.tauX  = state.variables.surface_taux #TODO: get actual surface stress, i.e. including ice velocity
#     SIstate.variables.tauY  = state.variables.surface_tauy
#     SIstate.variables.ssh  = state.variables.ssh

#     SIstate.settings.dxC    = state.variables.dxt
#     SIstate.settings.dyC    = state.variables.dyt
#     SIstate.settings.dxU    = state.variables.dxu
#     SIstate.settings.dyU    = state.variables.dyu
#     SIstate.settings.rA     = state.variables.area_t
#     SIstate.settings.rAw    = state.variables.area_u
#     SIstate.settings.rAs    = state.variables.area_v

#     # maskW is a vertical mass

#     # the grid has to be rectangular? 
#     # can i initialize a setting dependent on another setting, e.g. recip_deltatTherm?