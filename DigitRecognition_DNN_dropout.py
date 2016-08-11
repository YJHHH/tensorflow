'''
Digit recognition with tensorflow
Using MFCCs for extracting feature 
And using DNN
dropout on.
'''
import tensorflow as tf
import numpy as np
keep_prob=tf.placeholder("float")
#import feat_remove_silence2 as train
ydat=np.loadtxt('text.txt', unpack=True, dtype='float32')
xdat= np.loadtxt('output.txt', unpack=True, dtype='float32')
t_xdat=np.loadtxt('test_feat.txt', unpack=True, dtype='float32')
t_ydat=np.loadtxt('test_answer.txt',unpack=True,dtype='float32')
x_data=np.transpose(xdat[0:390])
y_data=np.transpose(ydat[0:10])
t_x_data=np.transpose(t_xdat[0:390])
t_y_data=np.transpose(t_ydat[0:10])
x= tf.placeholder(tf.float32, [None, 390])
y_=tf.placeholder(tf.float32,[None,10])
#dropout_rate=tf.placeholder(tf.float32)
#keep_prob = tf.placeholder("float")
w1=tf.Variable(tf.random_normal([390,256]))
w2=tf.Variable(tf.random_normal([256,256]))
w3=tf.Variable(tf.random_normal([256,256]))
w4=tf.Variable(tf.random_normal([256,256]))
w5=tf.Variable(tf.random_normal([256,10]))

b1=tf.Variable(tf.random_normal([256]))
b2=tf.Variable(tf.random_normal([256]))
b3=tf.Variable(tf.random_normal([256]))
b4=tf.Variable(tf.random_normal([256]))
b5=tf.Variable(tf.random_normal([10]))

_l1=tf.nn.relu(tf.add(tf.matmul(x,w1),b1))
l1=tf.nn.dropout(_l1,keep_prob)
_l2=tf.nn.relu(tf.add(tf.matmul(l1,w2),b2))
l2=tf.nn.dropout(_l2,keep_prob)
_l3=tf.nn.relu(tf.add(tf.matmul(l2,w3),b3))
l3=tf.nn.dropout(_l3,keep_prob)
_l4=tf.nn.relu(tf.add(tf.matmul(l3,w4),b4))
l4=tf.nn.dropout(_l4,keep_prob)

y=tf.add(tf.matmul(l4,w5),b5)

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y,y_))
#cross_entropy=-tf.reduce_sum(y_*tf.log(y))
train_step=tf.train.AdamOptimizer(0.01).minimize(cost)
init=tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(train_step, feed_dict={x:x_data, y_:y_data,keep_prob:0.7})
    if i%200==0:
          print i, sess.run(cost, feed_dict={x:x_data, y_:y_data,keep_prob:0.7}),sess.run(w1)

correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(acc, feed_dict={x:t_x_data,y_:t_y_data,keep_prob:1}))
