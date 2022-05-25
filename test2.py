from veros import runtime_settings
backend = 'numpy'
runtime_settings.backend = backend
from veros.core.operators import numpy as npx

from initialize import state

from veros_seaice_plugin import main_seaice

print(npx.mean(state.variables.uIce))
main_seaice(state)
print(npx.mean(state.variables.uIce))
