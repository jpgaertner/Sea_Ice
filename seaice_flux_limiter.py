# statement function to describe flux limiter

import numpy as np

def limiter(Cr):
    
    #return 0       (upwind)
    #return 1       (Lax-Wendroff)
    #return np.max((0, np.min((1, Cr))))    (Min-Mod)
    return np.maximum(0, np.maximum(np.minimum(1,2*Cr), np.minimum(2,Cr)))