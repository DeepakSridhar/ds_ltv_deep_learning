def relu_backward(dA, cache):
    import numpy as np
    from relu_grad import relu_gradient
    z=cache
    dZ=np.multiply(dA,relu_gradient(z))
    # z = cache
    # dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
    #
    # # When z <= 0, you should set dz to 0 as well.
    # dZ[z <= 0] = 0

    assert (dZ.shape == z.shape)
    return  dZ

