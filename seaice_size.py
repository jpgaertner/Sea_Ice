from veros.core.operators import numpy as npx
from veros.core.operators import update, at

from seaice_fill_overlap import fill_overlap

# # for the 1d test
# sNx = 1
# sNy = 1
# OLx = 2
# OLy = 2

# number of cells of tile along x, y axis
sNx = 32+1
sNy = sNx

# number of cells in tile overlap
OLx = 2
OLy = 2

# number of ice thickness categories
# instead of assuming just one thickness for a grid cell, a distribution of
# nITD thicknesses (centered around the mean value and containing equal fractions of the area) is assumed.
# Bitz et al. (2001, JGR) suggest a minimum of nITD = 5
nITC = 1
recip_nITC = 1 / nITC

# grid cell width [m]
gridcellWidth = 512e3/(sNx-1) #grid cell width in m
print('gridcellWidth = %e'%gridcellWidth)

# grid descriptor variables
deltaX = npx.ones((sNy+2*OLy,sNx+2*OLx))*gridcellWidth
dxC = deltaX.copy() #distance between two adjacent cell centers in x direction across western cell wall [m]
dyC = deltaX.copy() #distance between two adjacent cell centers in y direction across southern cell wall [m]
dxG = deltaX.copy() #distance between cell faces (cell width) in x direction along southern cell wall [m]
dyG = deltaX.copy() #distance between cell faces (cell width) in y direction along western cell wall [m]
dxF = deltaX.copy() #distance between cell faces (cell width) in x direction through cell center [m]
dyF = deltaX.copy() #distance between cell faces (cell width) in y direction through cell center [m]
dxV = deltaX.copy() #distance between two adjacent v points in x direction across south-west corner of the cell [m]
dyU = deltaX.copy() #distance between two adjacent u points in y direction across south-west corner of the cell [m]
recip_dxC = 1 / dxC
recip_dyC = 1 / dyC
recip_dxG = 1 / dxG
recip_dyG = 1 / dyG
recip_dxF = 1 / dxF
recip_dyF = 1 / dyF
recip_dxV = 1 / dxV
recip_dyU = 1 / dyU
dxN = 1

rA = dxF * dyF #grid width with c point at center
rAz = dxV * dyU #grid width with z point at center
rAw = dxC * dyG #grid width with u point at center
rAs = dxG * dyC #grid width with v point at center
recip_rA = 1 / rA
recip_rAz = 1 / rAz
recip_rAw = 1 / rAw
recip_rAs = 1 / rAs

# coriolis parameter at grid center point
fCori = npx.ones((sNy+2*OLy,sNx+2*OLx)) * 0. #1.e-4

# masks for introducing boundaries
maskInC = npx.ones((sNy+2*OLy,sNx+2*OLx))
maskInC = update(maskInC, at[sNy+OLy-1,:], 0)
maskInC = update(maskInC, at[:,sNx+OLx-1], 0)
maskInC = fill_overlap(maskInC)

maskInW = maskInC*npx.roll(maskInC,1,axis=1)
maskInW = fill_overlap(maskInW)

maskInS = maskInC*npx.roll(maskInC,1,axis=0)
maskInS = fill_overlap(maskInS)

iceMask = maskInC.copy()
SeaIceMaskU = maskInW.copy()
SeaIceMaskV = maskInS.copy()

k1AtC = npx.zeros((sNy+2*OLy,sNx+2*OLx))
k2AtC = npx.zeros((sNy+2*OLy,sNx+2*OLx))
k1AtZ = npx.zeros((sNy+2*OLy,sNx+2*OLx))
k2AtZ = npx.zeros((sNy+2*OLy,sNx+2*OLx))

# # for the 1d test
# maskInC = npx.ones((sNy+2*OLy,sNx+2*OLx))
# maskInW = npx.ones((sNy+2*OLy,sNx+2*OLx))
# maskInS = npx.ones((sNy+2*OLy,sNx+2*OLx))
# iceMask = maskInC.copy()
# SeaIceMaskU = maskInW.copy()
# SeaIceMaskV = maskInS.copy()

globalArea = (maskInC*rA).sum()
if globalArea == 0: print('globalArea = 0, something is wrong')