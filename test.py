from veros import runtime_settings
from seaice_params import backend
runtime_settings.backend = backend

from initialize import state
from seaice_dynsolver import update_IceVelocities
from veros import veros_routine, veros_kernel, KernelOutput
from veros.core.operators import numpy as npx
from veros.core.operators import update, at

@veros_routine
def f(state):
    state.variables.uIce = state.variables.uIce + 1.2

@veros_kernel
def func(state):
    x = state.variables.uIce
    x = x + 20
    uIce = x
    vIce = x
    f(state)
    uIce = uIce + state.variables.uIce
    return uIce, vIce

@veros_routine
def routine(state):
    uIce, vIce = func(state)
    #result = func(state)

    state.variables.uIce = uIce

# print(state.variables.uIce)
# routine(state)
# print(state.variables.uIce)

@veros_kernel
def func():
    x = 0
    while(x<100):
        print(x)
        x += 1


x = npx.arange(10)
a = (x>4)
print(a)
x *= ~a
print(x)


# x = npx.zeros(10)
# print('type 2', type(x[0]))
# print(x[0])

# y = update(x, at[:], x)
# print('type 3', type(y[0]))
# print(y[0])