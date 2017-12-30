def softmax_gradient(z):
    from softmax import softmax
    import numpy as np
    # m=z.shape[0]
    # n=z.shape[1]
    # for i in range(0,m):
        # a,c=softmax(z[i])
    a,c=softmax(z)
    g=np.multiply(a,1-a)
    return g