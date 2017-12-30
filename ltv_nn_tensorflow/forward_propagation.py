def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    import tensorflow as tf

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    # W3 = parameters['W3']
    # b3 = parameters['b3']
    # W4 = parameters['W4']
    # b4 = parameters['b4']
    # W5 = parameters['W5']
    # b5 = parameters['b5']

    ### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)  # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)  # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)  # Z2 = np.dot(W2, a1) + b2
    # A2 = tf.nn.relu(Z2)  # A2 = relu(Z2)
    # Z3 = tf.add(tf.matmul(W3, A2), b3)  # Z3 = np.dot(W3,a2) + b3
    # A3 = tf.nn.relu(Z3)  # A3 = relu(Z3)
    # Z4 = tf.add(tf.matmul(W4, A3), b4)  # Z4 = np.dot(W4, a3) + b4
    # A4 = tf.nn.relu(Z4)  # A4 = relu(Z4)
    # Z5 = tf.add(tf.matmul(W5, A4), b5)  # Z5 = np.dot(W5,a4) + b5

    return Z2