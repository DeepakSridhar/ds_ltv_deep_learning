def softmax_backward(dA, cache):
    from softmax_grad import softmax_gradient
    z=cache
    dZ=dA*softmax_gradient(z)
    assert (dZ.shape == z.shape)
    return  dZ

