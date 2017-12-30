def compute_cost(AL, Y, num_labels, parameters, lambd):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    import numpy as np

    m = Y.shape[1]
    cost=0
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    # W3 = parameters["W3"]

    # Compute loss from AL and y.
    # cost = -(1/m)*np.sum(np.multiply(Y,np.log(AL)))
    for k in range(1,num_labels+1):
        yk=(Y==k)
        cost += -(1 / m) * np.sum(np.multiply(np.log(AL), yk))
    L2_regularization_cost = (lambd / (2 * m)) * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    cost=cost+L2_regularization_cost
    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert (cost.shape == ())

    return cost
