import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random as random
error = [990,990,990,990,990,990,990,990,990,990]
error1 = [995,995,995,995,995,995,995,995,995,995]
error0 = [980,980,980,980,980,980,980,980,980,980]
counter = 0
for corruption in error0:
    counter = counter + 1
    precision = tf.float32

    num_train_pts = 1000
    num_test_pts = 5000

    hidden_dim = 300
    depth = 11
    input_dim = 2
    output_dim = 121
    batch_size = 100
    delta = 0.1
    epochs =1000


    x_test = np.load("X_test.npy")
    x_train = np.load("X_train.npy")
    y_train = np.load("Y_train.npy") 
    y_test = np.load("Y_test.npy")
    Gram = np.load("G.npy")

    Sparse =  random.sample(range(1000),corruption)
    for randomsample in Sparse:
        y_train[randomsample] = y_train[randomsample] + np.random.normal(scale = 100, size = 121)


    Gram_train = np.zeros(shape = (1000,1))
    for i in range(y_train.shape[0]):
        Gram_train[i] = np.linalg.norm(np.matmul(Gram,y_train[i].reshape(121,1)))
    Gram_train = np.array(Gram_train).reshape(1,1000)


    Gram_test = np.zeros(shape = (5000,1))
    for i in range(y_test.shape[0]):
        Gram_test[i] = np.linalg.norm(np.matmul(Gram,y_test[i].reshape(121,1)))
    Gram_test = np.array(Gram_test).reshape(5000)

    x_test=np.transpose(x_test).astype(np.float32)
    x_train=np.transpose(x_train).astype(np.float32)
    y_train=np.transpose(y_train).astype(np.float32) 
    y_test=np.transpose(y_test).astype(np.float32)

    zeros = np.zeros(shape = (100))
    zeros2 = np.zeros(shape=(5000))

    rho = tf.nn.leaky_relu


    def default_block(x, layer, dim1, dim2, weight_bias_initializer, rho, precision=tf.float32):
        W = tf.compat.v1.get_variable(name='l' + str(layer) + '_W', shape=[dim1, dim2],
                                      initializer=weight_bias_initializer, dtype=precision)

        b = tf.compat.v1.get_variable(name='l' + str(layer) + '_b', shape=[dim2, 1],
                                      initializer=weight_bias_initializer, dtype=precision)

        return rho(tf.matmul(W, x) + b)


    def funcApprox(x, layers=11, input_dim=2, output_dim=121, hidden_dim=300, precision=tf.float32):
        print('Constructing the tensorflow nn graph')

        weight_bias_initializer = tf.random_normal_initializer(stddev=delta)

        with tf.compat.v1.variable_scope('UniversalApproximator',reuse=tf.compat.v1.AUTO_REUSE):
            # input layer description
            in_W = tf.compat.v1.get_variable(name='in_W', shape=[hidden_dim, input_dim],
                                             initializer=weight_bias_initializer, dtype=precision,)

            in_b = tf.compat.v1.get_variable(name='in_b', shape=[hidden_dim, 1],
                                             initializer=weight_bias_initializer, dtype=precision)

            z = tf.matmul(in_W, x) + in_b

            x = rho(z)



            for i in range(layers):
                choice = 0
                x = default_block(x, i, hidden_dim, hidden_dim, weight_bias_initializer, precision=precision,
                                  rho=rho)
                choice = 1

            out_v = tf.compat.v1.get_variable(name='out_v', shape=[output_dim, hidden_dim],
                                              initializer=weight_bias_initializer, dtype=precision)

            out_b = tf.compat.v1.get_variable(name='out_b', shape=[output_dim, 1],
                                              initializer=weight_bias_initializer, dtype=precision)

            z = tf.math.add(tf.linalg.matmul(out_v, x, name='output_vx'), out_b, name='output')
            return z

    def get_batch(X_in, Y_in, Gram_train, batch_size):
        X_cols = X_in.shape[0]
        Y_cols = Y_in.shape[0]

        for i in range(X_in.shape[1]//batch_size):
            idx = i*batch_size + np.random.randint(0,10,(1))[0]

            yield X_in.take(range(idx,idx+batch_size), axis = 1, mode = 'wrap').reshape(X_cols,batch_size), \
                  Y_in.take(range(idx,idx+batch_size), axis = 1, mode = 'wrap').reshape(Y_cols,batch_size),\
                  Gram_train.take(range(idx,idx+batch_size), axis = 1, mode = 'wrap').reshape(1, batch_size)


    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()


    with tf.compat.v1.variable_scope('Graph',reuse=tf.compat.v1.AUTO_REUSE) as scope:
        # inputs to the NN
        x = tf.compat.v1.placeholder(precision, shape=[2, None], name='input')
        y_true = tf.compat.v1.placeholder(precision, shape=[121, None], name='y_true')
        x_t = tf.compat.v1.placeholder(precision, shape=[2, None], name='x_test')
        y_t = tf.compat.v1.placeholder(precision, shape=[121, None], name='y_test')
        Gram_train_batch = 1



        y = funcApprox(x, layers=11, input_dim=2,output_dim=121, hidden_dim=300,precision=tf.float32)
        z = funcApprox(x_t, layers=11, input_dim=2,output_dim=121, hidden_dim=300,precision=tf.float32)   
        with tf.compat.v1.variable_scope('Loss'):    
            loss = tf.compat.v1.losses.absolute_difference(tf.math.pow(tf.linalg.norm(tf.math.divide(tf.linalg.matmul(Gram,y)-tf.linalg.matmul(Gram,y_true),Gram_train_batch),axis =0),1/64),zeros)
            #loss = tf.compat.v1.losses.absolute_difference(tf.math.pow(tf.linalg.norm(tf.linalg.matmul(Gram,y)-tf.linalg.matmul(Gram,y_true),axis =0),0.5),zeros)
            validationloss = tf.compat.v1.losses.absolute_difference(tf.linalg.norm(tf.math.divide(tf.linalg.matmul(Gram,z)-tf.linalg.matmul(Gram,y_t),Gram_test),axis =0),zeros2)
        init_rate = 0.0002
        lrn_rate = init_rate
        opt = tf.compat.v1.train.AdamOptimizer(learning_rate=lrn_rate)
        train_op = opt.minimize(loss)

        losses = []
        testloss = []

        print(np.shape(x_train))
        with tf.compat.v1.Session() as sess:
            # init variables
            sess.run(tf.compat.v1.global_variables_initializer())
            for i in range(epochs):

                for x_train_batch, y_true_train_batch, Gram_train_batch in get_batch(x_train, y_train, Gram_train,
                                                                                     batch_size):
                    current_loss, current_testloss, _ = sess.run([loss, validationloss, train_op],
                                                          feed_dict={x: x_train_batch, \
                                                                     y_true: y_true_train_batch, \
                                                                     x_t: x_test, \
                                                                     y_t: y_test,})
                    losses.append(current_loss)
                    testloss.append(current_testloss)



    print('done')
    x = range(int(epochs*1000/batch_size))
    plt.figure(counter)
    plt.title(corruption)
    plt.loglog(x,losses)
    plt.loglog(x,testloss)
    print(corruption,testloss[9999])
