
# coding: utf-8

# In[ ]:

import numpy as np
import tensorflow as tf
from PIL import Image
import csv
import time
import os
import urllib.request
import zipfile


# In[ ]:

BATCH_SIZE = 64
NUM_EPOCHS = 40
NUM_BATCHES_TRAIN = 450
NUM_BATCHES_TEST = 57


# In[ ]:

train_csv = open('train.csv','a')
train_writer = csv.writer(train_csv,delimiter=',')
test_csv = open('test.csv','a')
test_writer = csv.writer(test_csv,delimiter=',')
with open('fer2013.csv',"rt") as csvfile:
    reader = csv.reader(csvfile,delimiter = ',')
    i = 0
    for row in reader:
        if i == 0:
            i+= 1
            continue
        if row[2] == 'Training':
            train_writer.writerow(row)
        else:
            test_writer.writerow(row)
        i+= 1


# In[ ]:

train_csv.close()
test_csv.close()


# In[ ]:

#Function to generate batches of data
def generate_sample(filename):
    while True:
        with open(filename,'rt') as csvfile:
            reader = csv.reader(csvfile,delimiter = ',')
            for row in reader:
                y = np.zeros(7)
                y[row[0]] = 1
                yield (np.array(row[1].split()).astype(np.float),y)
                       
def get_batch(filename,batch_size = BATCH_SIZE):
    itr = generate_sample(filename)
    while True:
        batch_x = []
        batch_y = []
        for i in range(batch_size):
            a = next(itr)
            batch_x.append(a[0])
            batch_y.append(a[1])
        yield np.array(batch_x),np.array(batch_y).astype(np.float32)


# In[ ]:

#A function in case you want to save images in folders. Not required though for our purpose
def save_images(filename,mode):
    with open(filename,'rt') as csvfile:
        reader = csv.reader(csvfile,delimiter = ',')
        i = 0
        for row in reader:
            img = np.array(row[1].split()).astype(np.float)
            img = img.reshape([48,48,1])
            img = np.concatenate((img,img,img),axis = 2)
            img = img.astype(np.uint8)
            img = Image.fromarray(img)
            img.save('data/'+mode+'/'+str(row[0]) + '/'+str(i)+'.jpg')
            i += 1


# In[ ]:

#save_images('train.csv','train')


# In[ ]:

#save_images('test.csv','test')


# In[ ]:

a = get_batch('train.csv',1)


# In[ ]:

c = next(a)


# In[ ]:

img = c[0][0].reshape(48,48)
print(img.shape)
img = Image.fromarray(img)
img.show()


# In[ ]:

a = np.concatenate((a,a,a),axis = 2)


# In[ ]:

a = a.reshape([48,48,3])
a = a.astype(np.uint8)
img = Image.fromarray(a)
print(img.size)
img.save('foo.jpg')


# In[ ]:

b = Image.open('foo.jpg')


# In[ ]:

b = np.array(b)
print(b.shape)


# # Building TensorFlow model

# In[ ]:

#Placeholders
input_placeholder = tf.placeholder(tf.float32, shape = [BATCH_SIZE,2304])
output_placeholder = tf.placeholder(tf.float32, shape = [BATCH_SIZE, 7])


# In[ ]:

#Building CNN
regularizer = tf.contrib.layers.l2_regularizer(scale=0.25)
a1 = tf.reshape(input_placeholder,[-1,48,48,1])
conv1 = tf.layers.conv2d(
        inputs = a1,
        filters = 64,
        kernel_size = 3,
        padding = 'same',
        kernel_regularizer = regularizer,
        activation = tf.nn.relu)

bn1 = tf.layers.batch_normalization(conv1,axis = 1)
conv2 = tf.layers.conv2d(
        inputs = bn1,
        filters = 64,
        kernel_size = 3,
        padding = 'same',
        kernel_regularizer = regularizer,
        activation = tf.nn.relu)
bn2 = tf.layers.batch_normalization(conv2, axis = 1)

pool1 = tf.layers.max_pooling2d(
        inputs = bn2,
        pool_size = [2,2],
        strides = 2)

conv3 = tf.layers.conv2d(
        inputs = pool1,
        filters = 128,
        kernel_size = 3,
        padding = 'same',
        kernel_regularizer = regularizer,
        activation = tf.nn.relu)

bn3 = tf.layers.batch_normalization(conv3, axis = 1)

conv4 = tf.layers.conv2d(
        inputs = bn3,
        filters = 128,
        kernel_size = 3,
        padding = 'same',
        kernel_regularizer = regularizer,
        activation = tf.nn.relu)

bn4 = tf.layers.batch_normalization(conv4, axis = 1)

pool2 = tf.layers.max_pooling2d(
        inputs = bn4,
        pool_size = [2,2],
        strides = 2)

avg_pool = tf.nn.avg_pool(pool2,[1,12,12,1],strides=[1,1,1,1],padding="VALID")
avg_pool = tf.reshape(avg_pool,[-1,128])
dense = tf.layers.dense(inputs=avg_pool,units=1024,kernel_regularizer = regularizer,activation=tf.nn.relu)
bn5 = tf.layers.batch_normalization(dense,axis=1)
y_out = tf.layers.dense(inputs = bn5, units=7)


# In[ ]:

#Calculating Cross Entropy Loss
loss = tf.nn.softmax_cross_entropy_with_logits(labels = output_placeholder, logits = y_out)
loss = tf.reduce_mean(loss)


# In[ ]:

#Creating Adam Optimizer
global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
optimizer = tf.train.AdamOptimizer()
train_step = optimizer.minimize(loss,global_step)
saver = tf.train.Saver()


# In[ ]:

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    itr = get_batch('train.csv')
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/'))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    initial_step = global_step.eval()
    total_loss = 0
    start_time = time.time()
    for i in range(initial_step,NUM_EPOCHS*NUM_BATCHES_TRAIN):
        X_batch, y_batch = next(itr)
        _,loss_batch = sess.run([train_step,loss],feed_dict={
            input_placeholder : X_batch,
            output_placeholder : y_batch
        })
        total_loss += loss_batch
        if i%200 == 0:
            print('Average loss at step {}: {:5.1f}'.format(i + 1, total_loss / 200))
            saver.save(sess, 'checkpoints/fer_convenet', i)
            total_loss = 0
            itrtest = get_batch('test.csv')
            total_correct_preds = 0
            for i in range(20):
                X_batch, y_batch = next(itrtest)
                pred = sess.run(y_out,feed_dict = {
                    input_placeholder : X_batch,
                    output_placeholder : y_batch
                })
                pred = tf.nn.softmax(pred)
                correct_preds = tf.equal(tf.argmax(pred),tf.argmax(y_batch))
                accuracy = tf.reduce_sum(tf.cast(correct_preds,tf.float32))
                total_correct_preds += sess.run(accuracy)
            print("Accuracy {0}".format(total_correct_preds/(20*BATCH_SIZE)))
            

    print("Optimization Finished!")
    print("Total time: {0} seconds".format(time.time() - start_time))
    
    itr = get_batch('test.csv')
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/'))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    for i in range(NUM_BATCHES_TEST):
        X_batch, y_batch = next(itr)
        pred = sess.run(y_out,feed_dict = {
            input_placeholder : X_batch,
            output_placeholder : y_batch
        })
        pred = tf.nn.softmax(pred)
        correct_preds = tf.equal(tf.argmax(pred,1),tf.argmax(y_batch,1))
        accuracy = tf.reduce_sum(tf.cast(correct_preds,tf.float32))
        total_correct_preds += sess.run(accuracy)
    
    print("Accuracy {0}".format(total_correct_preds/(NUM_BATCHES_TEST*BATCH_SIZE)))


# In[ ]:

tf.get_default_graph().as_graph_def()


# In[ ]:

with tf.Session() as sess:
    itr = get_batch('test.csv')
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/'))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    total_correct_preds = 0
    for i in range(NUM_BATCHES_TEST):
        X_batch, y_batch = next(itr)
        pred = sess.run(y_out,feed_dict = {
            input_placeholder : X_batch,
            output_placeholder : y_batch
        })
        pred = tf.nn.softmax(pred)
        correct_preds = tf.equal(tf.argmax(pred,1),tf.argmax(y_batch,1))
        accuracy = tf.reduce_sum(tf.cast(correct_preds,tf.float32))
        total_correct_preds += sess.run(accuracy)
    
    print("Accuracy {0}".format(total_correct_preds/(NUM_BATCHES_TEST*BATCH_SIZE)))


# In[ ]:



