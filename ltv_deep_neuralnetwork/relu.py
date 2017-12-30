def relu(z):
    import numpy as np
    g=np.maximum(0,z)
    assert (g.shape == z.shape)
    cache=z
    return g,cache