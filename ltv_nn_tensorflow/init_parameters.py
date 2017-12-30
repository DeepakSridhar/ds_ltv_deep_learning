def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [25, 12288]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [6, 12]
                        b3 : [6, 1]

    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
    import scipy.io as spio

    import tensorflow as tf

    tf.set_random_seed(1)  # so that your "random" numbers match ours

    init = spio.loadmat('ltv_init_params_trained.mat', squeeze_me=True)
    initial_Theta1=init["initial_Theta1"]
    initial_Theta2 = init["initial_Theta2"]
    # print(type(initial_Theta1))

    W1 = tf.get_variable("W1", initializer=tf.constant(initial_Theta1))
    W2 = tf.get_variable("W2", initializer=tf.constant(initial_Theta2))


    # W1 = tf.get_variable("W1", [500, 1000], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [60, 1], dtype=tf.float64, initializer=tf.zeros_initializer())
    # W2 = tf.get_variable("W2", [250, 500], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [3, 1], dtype=tf.float64,  initializer=tf.zeros_initializer())
    # W3 = tf.get_variable("W3", [125, 250], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    # b3 = tf.get_variable("b3", [125, 1], initializer=tf.zeros_initializer())
    # W4 = tf.get_variable("W4", [25, 125], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    # b4 = tf.get_variable("b4", [25, 1], initializer=tf.zeros_initializer())
    # W5 = tf.get_variable("W5", [3, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    # b5 = tf.get_variable("b5", [3, 1], initializer=tf.zeros_initializer())


    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  # "W3": W3,
                  # "b3": b3,
                  # "W4": W4,
                  # "b4": b4,
                  # "W5": W5,
                  # "b5": b5,
                  }

    return parameters