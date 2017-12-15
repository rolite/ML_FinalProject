# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import math

from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


height = 32
width = 32
channels = 3
n_inputs = height * width * channels

conv1_fmaps = 48
conv1_ksize = 3
conv1_stride = 1
conv1_pad = "SAME"

conv2_fmaps = 48
conv2_ksize = 3
conv2_stride = 1
conv2_pad = "SAME"

pool3_fmaps = conv2_fmaps

conv4_fmaps = 96
conv4_ksize = 3
conv4_stride = 2
conv4_pad = "SAME"

pool5_fmaps = conv4_fmaps

n_fc1 = 96
n_outputs = 10

dropout_rate = 0.25
dropout_rate2 = 0.5 
tf.reset_default_graph()


with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
    y = tf.placeholder(tf.int32, shape=[None], name="y")
    y_cls = tf.argmax(y,dimension=1)

with tf.name_scope("cv"):
    conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, kernel_size=conv1_ksize,
                             strides=conv1_stride, padding=conv1_pad,
                              name="conv1")
    cv1_bn = tf.layers.batch_normalization(conv1, momentum=0.9)
    conv1_act = tf.nn.elu(cv1_bn)
    
    conv2 = tf.layers.conv2d(conv1_act, filters=conv2_fmaps, kernel_size=conv2_ksize,
                             strides=conv2_stride, padding=conv2_pad,
                             name="conv2")
    cv2_bn = tf.layers.batch_normalization(conv2, momentum=0.9)
    conv2_act = tf.nn.elu(cv2_bn)
    
    pool3 = tf.nn.max_pool(conv2_act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    
    conv4 = tf.layers.conv2d(pool3, filters=conv4_fmaps, kernel_size=conv4_ksize,
             strides=conv4_stride, padding=conv4_pad,
              name="conv4")
    conv4_drop = tf.layers.dropout(conv4, dropout_rate)
    cv4_bn = tf.layers.batch_normalization(conv4, momentum=0.9)
    conv4_act = tf.nn.elu(cv4_bn)
    
    conv5 = tf.layers.conv2d(conv4_act, filters=conv4_fmaps, kernel_size=conv4_ksize,
             strides=conv4_stride, padding=conv4_pad,
              name="conv5")
    cv5_bn = tf.layers.batch_normalization(conv5, momentum=0.9)
    conv5_act = tf.nn.elu(cv5_bn)
   

with tf.name_scope("pool5"):
    pool5 = tf.nn.max_pool(conv5_act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    pool5_flat = tf.reshape(pool5, shape=[-1, pool5_fmaps * 4])


with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(pool5_flat, n_fc1,  name="fc1")
    fc1_drop = tf.layers.dropout(pool5_flat, dropout_rate2)
    fc1_bn = tf.layers.batch_normalization(fc1, momentum=0.9)
    fc1_act = tf.nn.relu(fc1_bn)
    
    fc2 = tf.layers.dense(fc1_bn, n_fc1, name="fc2")
    fc2_drop = tf.layers.dropout(fc1, dropout_rate2)
    fc2_bn = tf.layers.batch_normalization(fc2, momentum=0.9)
    fc2_act = tf.nn.relu(fc2_bn)
    
with tf.name_scope("output"):
    logits = tf.layers.dense(fc2_bn, n_outputs, name="output")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")

with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
# load data
path = "C:/Users/David/Documents/College2/CST495-Special-DataScience/FinalProject/datasets/cifar-10-python/cifar-10-batches-py/"
train = unpickle(path + "data_batch_1")
train.update(unpickle(path + "data_batch_2"))
train.update(unpickle(path + "data_batch_3"))
train.update(unpickle(path + "data_batch_4"))

test = unpickle(path + "test_batch")
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/")

num_examples = len(train[b'data'])
dat_reshaped, test_reshaped = [],[]
for img in train[b'data']:
    dat_reshaped.append(np.transpose(np.reshape(img,(3, 32,32)), (1,2,0)))
for img in test[b'data']:
    test_reshaped.append(np.transpose(np.reshape(img,(3, 32,32)), (1,2,0)))

n_epochs = 20
batch_size = 500

tr_dat = train[b'data']
tr_labs = train[b'labels']

te_dat = test[b'data']
te_labs = test[b'labels']

ohe = LabelBinarizer()
#tr_labs = ohe.fit_transform(tr_labs)
#te_labs = ohe.fit_transform(te_labs)

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        bestloss = -1
        #rng_state = np.random.get_state()
        #np.random.shuffle(tr_dat)
        #np.random.set_state(rng_state)
        #np.random.shuffle(tr_labs)
        for iteration in range(num_examples // batch_size):
            #X_batch = []
            #y_batch = []
            #for i in range(iteration*batch_size, iteration*(batch_size+1)):
               #X_batch.append(tr_dat[keys[i]])
                #y_batch.append(tr_labs[keys[i]])
            X_batch = tr_dat[iteration*batch_size:iteration*(batch_size+1)]
            y_batch = tr_labs[iteration*batch_size:iteration*(batch_size+1)]
            
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
            loss = tf.reduce_mean(xentropy)
            _, loss_val = sess.run([training_op, loss], feed_dict={X: X_batch, y: y_batch})
            if( not math.isnan(loss_val) and (loss_val < bestloss or bestloss == -1)):
                bestloss = loss_val
        
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test  = accuracy.eval(feed_dict={X: te_dat, y: te_labs})
        print(epoch, "Train accuracy:", acc_train, " - Test accuracy:", acc_test, " - Loss:", bestloss)
        