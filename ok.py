import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random as random

precision = np.float32
num_train_pts = 1000
num_test_pts = 5000
hidden_dim = 300
depth = 11
input_dim = 2
output_dim = 121
rho = tf.nn.leaky_relu
batch_size = 100
epochs =1000
lrn_rate = 0.0002
delta = 0.1
p = 1
Noise_size = 0
corruption = 0

x_train = np.load("X_train.npy")
y_train = np.load("Y_train.npy") 
x_test = np.load("X_test.npy")
y_test = np.load("Y_test.npy")
Gram = np.load("G.npy")

Sparse =  random.sample(range(num_train_pts),corruption)
for randomsample in Sparse:
    y_train[randomsample] = y_train[randomsample] + np.random.normal(scale = 100, size = 121)
    
Normal = np.zeros(shape = (num_train_pts,output_dim))
for i in range(num_train_pts):
    x = np.random.standard_normal(output_dim)
    x = x/np.linalg.norm(x)
    Normal[i]=x
y_train = y_train + Normal*Noise_size
    
Sobolev_train = np.zeros(shape = (num_train_pts,1))
for i in range(y_train.shape[0]):
    Sobolev_train[i] = np.linalg.norm(np.matmul(Gram,y_train[i].reshape(121,1)))
Sobolev_train = np.array(Sobolev_train).reshape(1,num_train_pts)

Sobolev_test = np.zeros(shape = (num_test_pts,1))
for i in range(y_test.shape[0]):
    Sobolev_test[i] = np.linalg.norm(np.matmul(Gram,y_test[i].reshape(121,1)))
Sobolev_test = np.array(Sobolev_test).reshape(1,num_test_pts)

x_test=np.transpose(x_test).astype(precision)
x_train=np.transpose(x_train).astype(precision)
y_train=np.transpose(y_train).astype(precision) 
y_test=np.transpose(y_test).astype(precision)

zeros = np.zeros(shape = (batch_size))
zeros2 = np.zeros(shape=(num_test_pts))

def default_block(x, layer, dim1, dim2, weight_bias_initializer, rho):
    W = tf.compat.v1.get_variable(name='l' + str(layer) + '_W', shape=[dim1, dim2],
                                    initializer=weight_bias_initializer, dtype=precision)

    b = tf.compat.v1.get_variable(name='l' + str(layer) + '_b', shape=[dim2, 1],
                                    initializer=weight_bias_initializer, dtype=precision)

    return rho(tf.matmul(W, x) + b)


def funcApprox(x, depth, input_dim, output_dim, hidden_dim):
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



        for i in range(depth):
            choice = 0
            x = default_block(x, i, hidden_dim, hidden_dim, weight_bias_initializer, rho)
            choice = 1

        out_v = tf.compat.v1.get_variable(name='out_v', shape=[output_dim, hidden_dim],
                                            initializer=weight_bias_initializer, dtype=precision)

        out_b = tf.compat.v1.get_variable(name='out_b', shape=[output_dim, 1],
                                            initializer=weight_bias_initializer, dtype=precision)

        z = tf.math.add(tf.linalg.matmul(out_v, x, name='output_vx'), out_b, name='output')
        return z

def get_batch(X_in, Y_in,Sobolev_train,batch_size):
    X_cols = X_in.shape[0]
    Y_cols = Y_in.shape[0]

    for i in range(X_in.shape[1]//batch_size):
        idx = i*batch_size + np.random.randint(0,10,(1))[0]

        yield X_in.take(range(idx,idx+batch_size), axis = 1, mode = 'wrap').reshape(X_cols,batch_size), \
              Y_in.take(range(idx,idx+batch_size), axis = 1, mode = 'wrap').reshape(Y_cols,batch_size),\
              Sobolev_train.take(range(idx,idx+batch_size), axis = 1, mode = 'wrap').reshape(1, batch_size)

        

tf.compat.v1.reset_default_graph()
tf.compat.v1.disable_eager_execution()

with tf.compat.v1.variable_scope('Graph',reuse=tf.compat.v1.AUTO_REUSE) as scope:

    x = tf.compat.v1.placeholder(precision, shape=[input_dim, None], name='input')
    y_true = tf.compat.v1.placeholder(precision, shape=[output_dim, None], name='y_true')
    x_t = tf.compat.v1.placeholder(precision, shape=[input_dim, None], name='x_test')
    y_t = tf.compat.v1.placeholder(precision, shape=[output_dim, None], name='y_test')
    Sobolev_train_b = tf.compat.v1.placeholder(precision, shape=[1, None], name='S_train')



    y = funcApprox(x, depth, input_dim,output_dim, hidden_dim)
    z = funcApprox(x_t, depth, input_dim,output_dim, hidden_dim)   
    with tf.compat.v1.variable_scope('Loss'):    
        loss = tf.compat.v1.losses.absolute_difference(tf.math.pow(tf.linalg.norm(tf.math.divide(
               tf.linalg.matmul(Gram,y-y_true),Sobolev_train_b),axis =0),p),zeros)
        test_loss = tf.compat.v1.losses.absolute_difference(tf.linalg.norm(tf.math.divide(
                         tf.linalg.matmul(Gram,z-y_t),Sobolev_test),axis =0),zeros2)

    opt = tf.compat.v1.train.AdamOptimizer(learning_rate=lrn_rate)
    train_op = opt.minimize(loss)

    losses = []
    testloss = []

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for i in range(epochs):

            for x_train_batch, y_train_batch, Sobolev_train_batch in get_batch(x_train, y_train, Sobolev_train,
                                                                                     batch_size):
                current_loss, current_testloss, _ = sess.run([loss, test_loss, train_op],
                                                          feed_dict={x: x_train_batch, \
                                                                     y_true: y_train_batch, \
                                                                     x_t: x_test, \
                                                                     y_t: y_test,
                                                                    Sobolev_train_b : Sobolev_train_batch})
                losses.append(current_loss)
                testloss.append(current_testloss)
        y_res = sess.run([y], feed_dict = {x: x_test})
        y_NN = y_res[0]
print('done')
x = range(len(losses))
plt.title(corruption)
plt.loglog(x,losses)
plt.loglog(x,testloss)
plt.show()
print(testloss[-1])
