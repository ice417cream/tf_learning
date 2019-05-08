import tensorflow as tf
import matplotlib.pyplot as plt

#P62
from numpy.random import RandomState
#构建网络
batch_size = 8
w1 = tf.Variable(tf.random_normal([2,3],stddev = 1, seed = 1))
w2 = tf.Variable(tf.random_normal([3,1],stddev = 1, seed = 1))

x = tf.placeholder(tf.float32,shape=(None,2),name = 'x-input')
y_ = tf.placeholder(tf.float32,shape=(None,1),name = 'y-input')

a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

y = tf.sigmoid(y)

cross_entroy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0))+(1-y_)*tf.log(tf.clip_by_value(1-y,1e-10,1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entroy)

#形成训练数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,2)
Y = [[int(x1+x2<1)] for (x1,x2) in X]

# 形成测试数据集
rdm_test = RandomState(1)
dataset_size_test = 1000
X_test = rdm.rand(dataset_size_test, 2)
Y_test = [[int(x1 + x2 < 1)] for (x1, x2) in X_test]

#训练过程

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print(sess.run(w1))
    print(sess.run(w2))

    STEPS = 10000
    for i in range(STEPS):
        start = (i*batch_size) % dataset_size
        end = min(start+batch_size,dataset_size)

        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})

        if i %1000 == 0:
            total_cross_entropy = sess.run(cross_entroy,feed_dict={x:X,y_:Y})
            print("total_cross_entropy=",total_cross_entropy)

    print(sess.run(w1))
    print(sess.run(w2))

    print(sess.run(y, feed_dict={x: X_test}))



