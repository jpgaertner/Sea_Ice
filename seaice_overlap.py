import numpy as np

from seaice_params import *
from seaice_size import *

# this function cuts of the overlaps on each side of the cell
# input: field of size (sNx+2*OLx, sNy+2*OLy) where [OLx:-OLx,OLy:-OLy] is the actual cell
# output: field of size (sNx,sNy)
def cut_overlap(A):

    return A[OLx:-OLx,OLy:-OLy]

# this function adds an overlap to the cell with values 1
# input: field of size (sNx,sNy)
# output: field of size (sNx+2*OLx, sNy+2*OLy)
def add_overlap(A):
    B = np.ones((sNx+2*OLx,sNy+2*OLy))
    B[OLx:-OLx,OLy:-OLy] = A

    return B

# this function fills the overlap with the values from the cell
# this requires that the cell size is larger than the overlap!
def fill_overlap(A):
    A[:OLx,:OLy] = A[sNx:sNx+OLx,sNy:sNy+OLy] #fill upper left overlap with lower right cell values
    A[sNx+OLx:sNx+2*OLx,:OLy] = A[OLx:2*OLx,sNy:sNy+OLy] #fill lower left overlap with upper right cell values
    A[:OLx,sNy+OLy:sNy+2*OLy] = A[sNx:sNx+OLx,OLy:2*OLy] #fill upper right overlap with lower left cell values
    A[sNx+OLx:sNx+2*OLx,sNy+OLy:sNy+2*OLy] = A[OLx:2*OLx,OLy:2*OLy] #fill lower right overlap with upper left cell values
    A[OLx:sNx+OLx,:OLy] = A[OLx:sNx+OLx,sNy:sNy+OLy] #fill left overlap with right cell values
    A[OLx:sNx+OLx,sNy+OLy:sNy+2*OLy] = A[OLx:sNx+OLx,OLy:2*OLy] #fill right overlap with left cell values
    A[:OLx,OLy:sNy+OLy] = A[sNx:sNx+OLx,OLy:sNy+OLy] #fill upper overlap with lower cell values
    A[sNx+OLx:sNx+2*OLx,OLy:sNy+OLy] = A[OLx:2*OLx,OLy:sNy+OLy] #fill lower right overlap with upper cell values

    return A

def add_fill_overlap(A):
    A = add_overlap(A)
    A = fill_overlap(A)
    
    return A
