def predict(X_test, y_test, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    """

    import  tensorflow as tf
    from forward_propagation import forward_propagation
    from create_placeholder import  create_placeholders

    (n_x, m) = X_test.shape  # (n_x: input size, m : number of examples in the test set)
    n_y = y_test.shape[0]  # n_y : output size

    X, Y = create_placeholders(n_x, n_y)

    Z3 = forward_propagation(X, parameters)

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        # Run the initialization
        sess.run(init)

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Test Accuracy:", accuracy.eval({X: X_test, Y: y_test}))

    return None