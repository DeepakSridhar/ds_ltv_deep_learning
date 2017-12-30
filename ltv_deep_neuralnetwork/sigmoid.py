def sigmoid(z):
    import numpy as np
    g=1/(1+np.exp(-z))
    cache=z
    return g,cache