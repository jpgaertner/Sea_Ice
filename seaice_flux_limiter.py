from veros.core.operators import numpy as npx


def limiter(Cr):
    
    #return 0       (upwind)
    #return 1       (Lax-Wendroff)
    #return np.max((0, np.min((1, Cr))))    (Min-Mod)
    return npx.maximum(0, npx.maximum(npx.minimum(1,2*Cr), npx.minimum(2,Cr)))