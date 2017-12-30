def softmax(z):
    import numpy as np
    shiftz = z - np.max(z)
    g=np.exp(shiftz)/np.sum(np.exp(shiftz),keepdims=True)
    cache=z
    return g,cache