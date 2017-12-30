def compute_cost(Z3, Y, lambd, parameters):
    """
    Computes the cost

    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    """
    import tensorflow as tf

    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    ### START CODE HERE ### (1 line of code)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    regularizer = tf.nn.l2_loss(parameters["W1"])+tf.nn.l2_loss(parameters["W2"])
                  # +tf.nn.l2_loss(parameters["W3"])\
                  # +tf.nn.l2_loss(parameters["W4"])+tf.nn.l2_loss(parameters["W5"])
    cost=tf.reduce_mean(cost+lambd*regularizer)
    ### END CODE HERE ###

    return cost