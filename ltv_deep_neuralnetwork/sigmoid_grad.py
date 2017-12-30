def sigmoid_gradient(z):
    from sigmoid import sigmoid
    a=sigmoid(z)[0]
    g=a*(1-a)
    return g