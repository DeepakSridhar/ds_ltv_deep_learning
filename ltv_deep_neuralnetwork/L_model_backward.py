def L_model_backward_multi(AL, X, Y, caches, parameters, num_labels):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    import  numpy as np
    from lin_act_back import linear_activation_backward
    from L_model_forward import L_model_forward
    from relu_grad import relu_gradient
    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    Delta={}
    delta={}


    # for i in range(1,num_labels):
    #     Yi=(Y==i)
    #     # Initializing the backpropagation
    #     dAL = - (np.divide(Yi, AL) - np.divide(1 - Yi, 1 - AL))
    #
    #     # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    #     current_cache = caches[L - 1]
    #     grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,
    #                                                                                                   current_cache,
    #                                                                                                   activation='sigmoid')
    #
    #     for l in reversed(range(L - 1)):
    #         # lth layer: (RELU -> LINEAR) gradients.
    #         # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
    #         ### START CODE HERE ### (approx. 5 lines)
    #         current_cache = caches[l]
    #         dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache,
    #                                                                     activation='relu')
    #         grads["dA" + str(l + 1)] = dA_prev_temp
    #         grads["dW" + str(l + 1)] = dW_temp
    #         grads["db" + str(l + 1)] = db_temp
    #         ### END CODE HERE ###

    for l in range(1,L+1):
        Delta["Del" + str(l)]=0
    #     delta['del' + str(l)] = np.zeros((AL.shape[0],1))

    delt=np.zeros((num_labels,1))
    for i in range(0,m):
        a1=X[:,i]
        a1=a1.reshape(a1.shape[0],1)
        # print(a1.shape)
        AL, cach, act = L_model_forward(a1, parameters)
        # act.append(a1)
        for k in range(1,num_labels+1):
            Yk=(Y==k)
            a2=Yk[:,i]
            a2 = a2.reshape(a2.shape[0], 3)
            ALk=AL[k-1].reshape(1,1)
            # print(ALk.shape)
            # print(a2.shape)
            delt[k-1]=ALk-a2
            # delt.append(deL)

        delta["del"+str(L)]= delt
        # print(delta['del' + str(L)])
        for l in reversed(range(L - 1)):
            # print(l)
            linear_cache, activation_cache = cach[l]
            z=linear_cache[3]
            dZ = relu_gradient(z)
            Wl=parameters['W'+str(l+2)]
            # print(Wl.shape)
            dl1=delta['del'+str(l+2)]
            DL=np.multiply(np.dot(Wl.T,dl1),dZ)
            DL=DL.reshape(DL.shape[0],1)
            delta["del" + str(l+1)] = DL

        for l in range(1,L+1):
            # print(act[l-1].shape)
            # print(delta["del" + str(l)].shape)
            if l==1:
                Delta["Del" + str(l)] += np.dot(delta["del" + str(l)], a1.T)
                Delta["Del" + str(l)] = Delta['Del' + str(l)] / m

            else:
                Delta["Del" + str(l)] += np.dot(delta["del" + str(l)], act[l-2].T)
                Delta["Del" + str(l)] = Delta['Del' + str(l)] / m

    for l in range(1, L+1):
        # Delta["Del" + str(l)] = Delta["Del" + str(l)].T
        grads["dW" + str(l)]=Delta['Del' + str(l)]
        grads["db" + str(l)]=delta['del' + str(l)]
        # print(grads["dW" + str(l)].shape)
    # print(len(delta))
    # print(len(Delta))
    return grads