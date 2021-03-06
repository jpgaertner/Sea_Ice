from veros.core.operators import numpy as npx
from veros.core.operators import update, at

from seaice_fill_overlap import fill_overlap

# # for the 1d test
# nx = 1
# ny = 1

# number of cells of tile along x, y axis
nx = 64+1
ny = nx

# number of cells in tile overlap
olx = 2
oly = 2

# number of ice thickness categories
# instead of assuming just one thickness for a grid cell, a distribution of
# nITD thicknesses (centered around the mean value and containing equal fractions of the area) is assumed.
# Bitz et al. (2001, JGR) suggest a minimum of nITD = 5
nITC = 1
recip_nITC = 1 / nITC

# grid cell width [m]
# gridcellWidth = 512e3/(nx-1) #grid cell width in m
gridcellWidth = 8000 #grid cell width in m
print('gridcellWidth = %e'%gridcellWidth)

# grid descriptor variables
deltaX = npx.ones((ny+2*oly,nx+2*olx))*gridcellWidth
dxC = deltaX    # distance between two adjacent cell centers in x direction across western cell wall [m]
dyC = deltaX    # distance between two adjacent cell centers in y direction across southern cell wall [m]
dxG = deltaX    # distance between cell faces (cell width) in x direction along southern cell wall [m]
dyG = deltaX    # distance between cell faces (cell width) in y direction along western cell wall [m]
dxU = deltaX    # distance between cell faces (cell width) in x direction through cell center [m]
dyV = deltaX    # distance between cell faces (cell width) in y direction through cell center [m]
dxV = deltaX    # distance between two adjacent v points in x direction through south-west corner of the cell [m]
dyU = deltaX    # distance between two adjacent u points in y direction through south-west corner of the cell [m]
# TODO remove here and use state obejct
recip_dxC = 1 / dxC
recip_dyC = 1 / dyC
recip_dxG = 1 / dxG
recip_dyG = 1 / dyG
recip_dxU = 1 / dxU
recip_dyV = 1 / dyV
recip_dxV = 1 / dxV
recip_dyU = 1 / dyU

rA = dxU * dyV      # grid area with c point at center
rAz = dxV * dyU     # grid area with z point at center
rAw = dxC * dyG     # grid area with u point at center
rAs = dxG * dyC     # grid area with v point at center
recip_rA = 1 / rA
recip_rAz = 1 / rAz
recip_rAw = 1 / rAw
recip_rAs = 1 / rAs

# coriolis parameter at grid center point
fCori = npx.ones((ny+2*oly,nx+2*olx)) * 0#1.46e-4

# masks for introducing boundaries
maskInC = npx.ones((ny+2*oly,nx+2*olx))
maskInC = update(maskInC, at[ny+oly-1,:], 0)
maskInC = update(maskInC, at[:,nx+olx-1], 0)
maskInC = fill_overlap(maskInC)

maskInW = maskInC*npx.roll(maskInC,1,axis=1)
maskInW = fill_overlap(maskInW)

maskInS = maskInC*npx.roll(maskInC,1,axis=0)
maskInS = fill_overlap(maskInS)

iceMask = maskInC
SeaIceMaskU = maskInW
SeaIceMaskV = maskInS

k1AtC = npx.zeros((ny+2*oly,nx+2*olx))
k2AtC = npx.zeros((ny+2*oly,nx+2*olx))
k1AtZ = npx.zeros((ny+2*oly,nx+2*olx))
k2AtZ = npx.zeros((ny+2*oly,nx+2*olx))

recip_hIceMean = npx.ones_like(iceMask) #TODO

# # for the 1d test
# maskInC = npx.ones((ny+2*oly,nx+2*olx))
# maskInW = npx.ones((ny+2*oly,nx+2*olx))
# maskInS = npx.ones((ny+2*oly,nx+2*olx))
# iceMask = maskInC
# SeaIceMaskU = maskInW
# SeaIceMaskV = maskInS

globalArea = (maskInC*rA).sum()
if globalArea == 0: print('globalArea = 0, something is wrong')