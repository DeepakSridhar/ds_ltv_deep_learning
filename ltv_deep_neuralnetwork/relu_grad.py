def relu_gradient(z):
    import numpy as np
    #k=np.nonzero(z)[0]
    g=np.zeros(z.shape)
    g[z>0]=1
    return g