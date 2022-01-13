import numpy as np


### input:
# A: field of size (sNy+2*OLy, sNx+2*OLx) where [OLy:-OLy,OLx:-OLx] (=[OLy:sNy+OLy,OLx:sNx+OLx]) is the actual cell

### output:
# A with filled overlaps (using the values from the actual cell)
# this requires that the cell size is larger than the overlap (sN > OL)!

sNx = 65 #Number of X points in tile
sNy = 65 #Number of Y points in tile
OLx = 2 #Tile overlap extent in X
OLy = 2 #Tile overlap extent in Y

def fill_overlap(A):
    A[:OLy,:OLx] = A[sNy:-OLy,sNx:-OLx] #fill upper left overlap with lower right cell values
    A[:OLy,-OLx:] = A[sNy:-OLy,OLx:2*OLx] #fill lower left overlap with upper right cell values
    A[-OLy:,:OLx] = A[OLy:2*OLy,sNx:-OLx] #fill upper right overlap with lower left cell values
    A[-OLy:,-OLx:] = A[OLy:2*OLy,OLx:2*OLx] #fill lower right overlap with upper left cell values
    A[:OLy,OLx:-OLx] = A[sNy:-OLy,OLx:-OLx] #fill left overlap with right cell values
    A[-OLy:,OLx:-OLx] = A[OLy:2*OLy,OLx:-OLx] #fill right overlap with left cell values
    A[OLy:-OLy,:OLx] = A[OLy:-OLy,sNx:-OLx] #fill upper overlap with lower cell values
    A[OLy:-OLy,-OLx:] = A[OLy:-OLy,OLx:2*OLx] #fill lower overlap with upper cell values

    return A


def fill_overlap3d(A):
    A[:,:OLy,:OLx] = A[:,sNy:-OLy,sNx:-OLx] #fill upper left overlap with lower right cell values
    A[:,:OLy,-OLx:] = A[:,sNy:-OLy,OLx:2*OLx] #fill lower left overlap with upper right cell values
    A[:,-OLy:,:OLx] = A[:,OLy:2*OLy,sNx:-OLx] #fill upper right overlap with lower left cell values
    A[:,-OLy:,-OLx:] = A[:,OLy:2*OLy,OLx:2*OLx] #fill lower right overlap with upper left cell values
    A[:,:OLy,OLx:-OLx] = A[:,sNy:-OLy,OLx:-OLx] #fill left overlap with right cell values
    A[:,-OLy:,OLx:-OLx] = A[:,OLy:2*OLy,OLx:-OLx] #fill right overlap with left cell values
    A[:,OLy:-OLy,:OLx] = A[:,OLy:-OLy,sNx:-OLx] #fill upper overlap with lower cell values
    A[:,OLy:-OLy,-OLx:] = A[:,OLy:-OLy,OLx:2*OLx] #fill lower overlap with upper cell values

    return A
