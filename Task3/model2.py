from __future__ import print_function, division

import pickle
import numpy as np
import tensorflow as tf
import sys
from tensorflow.contrib.rnn import MultiRNNCell, LSTMCell
from tqdm import tqdm

from utils import batch_generator

SEQUENCE_LENGTH = 10
HIDDEN_SIZE = 50
KEEP_PROB = 0.8
BATCH_SIZE = 64
NUM_EPOCHS = 50
MODEL_PATH = './model3'
NUM_INPUTS = 1
NUM_OUTPUTS = 1
N_LAYERS = 3
LEARNING_RATE = 0.001
PATIENCE = 5
TRAINING_SIZE = 50000

#using dst and imf time series to predict next time step dst value
dst_samples = pickle.load(open("dst_inp.pkl", "rb"))
imf_samples = pickle.load(open("imf_inp.pkl", "rb"))
labels = pickle.load(open("dst_label.pkl", "rb")) 

dst_samples = dst_samples[:TRAINING_SIZE]
imf_samples = imf_samples[:TRAINING_SIZE]
labels = labels[:TRAINING_SIZE]

print (len(dst_samples))
print (len(imf_samples))
#splitting data into 80-20
train_index = int(round((0.8*len(dst_samples)),0))
xtrain_dst = dst_samples[:train_index]
xtrain_imf = imf_samples[:train_index]
ytrain = labels[:train_index]

xtest_dst = dst_samples[train_index:]
xtest_imf = imf_samples[train_index:]
ytest = labels[train_index:]

print ("dst info... ")
print(xtrain_dst.shape)
print(xtest_dst.shape)
print(len(xtrain_dst))
print(len(xtest_dst))

print ("imf info... ")
print(xtrain_imf.shape)
print(xtest_imf.shape)
print(len(xtrain_imf))
print(len(xtest_imf))

xtrain = np.array(list(zip(xtrain_dst.tolist(), xtrain_imf.tolist())))
xtest = np.array(list(zip(xtest_dst.tolist(), xtest_imf.tolist())))

#--------------------------------------------------------------------------------------------
X1 = tf.placeholder(tf.float32, shape=(None, SEQUENCE_LENGTH, NUM_INPUTS), name="dst_inputs")
X2 = tf.placeholder(tf.float32, shape=(None, SEQUENCE_LENGTH, NUM_INPUTS), name="imf_inputs")
Y = tf.placeholder(tf.float32, shape=(None, NUM_OUTPUTS), name="labels")


#Defining the stacked LSTM cells for dst inputs
lstm_cells1 = [LSTMCell(HIDDEN_SIZE, forget_bias=1.0, use_peepholes=True, activation=tf.nn.relu) for _ in range(N_LAYERS)]
stacked_lstm1 = MultiRNNCell(lstm_cells1)
#Defining rnn layer to get the annotations for dst data
rnn_outputs1, states1 = tf.nn.dynamic_rnn(stacked_lstm1, inputs=X1, dtype=tf.float32, time_major=False, scope="rnn_dst")

#Defining the stacked LSTM cells for imf inputs
lstm_cells2 = [LSTMCell(HIDDEN_SIZE, forget_bias=1.0, use_peepholes=True, activation=tf.nn.relu) for _ in range(N_LAYERS)]
stacked_lstm2 = MultiRNNCell(lstm_cells2)
#Defining rnn layer to get the annotations for imf data
rnn_outputs2, states2 = tf.nn.dynamic_rnn(stacked_lstm2, inputs=X2, dtype=tf.float32, time_major=False, scope="rnn_imf")


print (rnn_outputs1.shape)
print (rnn_outputs2.shape)

#concatenating the dst and imf annotations to predict the next time step dst value
rnn_outputs = tf.concat([rnn_outputs1, rnn_outputs2], 1, name="concat")
print (rnn_outputs.shape)

# Dropout
drop = tf.nn.dropout(rnn_outputs, KEEP_PROB)

outputs = tf.transpose(drop, [1, 0, 2])
prediction = tf.layers.dense(outputs[-1], NUM_OUTPUTS)

#Defining squared loss
loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(prediction, Y), axis=1))
tf.summary.scalar('loss', loss)

#Defining optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

merged = tf.summary.merge_all()

#function call to batch_generator which returns the data of batch_size in every epoch
train_batch_generator = batch_generator(xtrain, ytrain, BATCH_SIZE)
test_batch_generator = batch_generator(xtest, ytest, BATCH_SIZE)

train_writer = tf.summary.FileWriter('./logdir/train', loss.graph)
test_writer = tf.summary.FileWriter('./logdir/test', loss.graph)

saver = tf.train.Saver()

if __name__ == "__main__":
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loss_test_prev = 100000
        flag_increase = 0
        count_increase = 0
        print("Start learning...")

        #Running the model epoch number of times
        for epoch in range(NUM_EPOCHS):
            loss_train = 0
            loss_test = 0

            print("epoch: {}\t".format(epoch), end="")

            # Training
            num_batches = xtrain.shape[0] // BATCH_SIZE
            for b in tqdm(range(num_batches)):
                batch_x, batch_y = next(train_batch_generator)
                batch_x_dst, batch_x_imf = zip(*batch_x)
                batch_x_dst = np.array(batch_x_dst)
                batch_x_imf = np.array(batch_x_imf)
                x_batch_dst = batch_x_dst.reshape((BATCH_SIZE, SEQUENCE_LENGTH, NUM_INPUTS))
                x_batch_imf = batch_x_imf.reshape((BATCH_SIZE, SEQUENCE_LENGTH, NUM_INPUTS))
                y_batch = batch_y.reshape((BATCH_SIZE, NUM_OUTPUTS))
                
                #running the model on train data
                loss_tr, _, summary = sess.run([loss, optimizer, merged],
                                                    feed_dict={X1: x_batch_dst,
                                                               X2: x_batch_imf, 
                                                               Y: y_batch})
                #accumulating total loss over training period
                loss_train += loss_tr
                train_writer.add_summary(summary, b + num_batches * epoch)            
            loss_train /= num_batches
            
            # Testing
            num_batches = xtest.shape[0] // BATCH_SIZE
            for b in tqdm(range(num_batches)):
                batch_x, batch_y = next(test_batch_generator)
                batch_x_dst, batch_x_imf = zip(*batch_x)
                batch_x_dst = np.array(batch_x_dst)
                batch_x_imf = np.array(batch_x_imf)
                x_batch_dst = batch_x_dst.reshape((BATCH_SIZE, SEQUENCE_LENGTH, NUM_INPUTS))
                x_batch_imf = batch_x_imf.reshape((BATCH_SIZE, SEQUENCE_LENGTH, NUM_INPUTS))
                y_batch = batch_y.reshape((BATCH_SIZE, NUM_OUTPUTS))
                
                #running the model on the test data to get the predictions
                loss_test_batch, summary = sess.run([loss, merged],
                                                         feed_dict={X1: x_batch_dst,
                                                                    X2: x_batch_imf, 
                                                                    Y: y_batch})
                #accumulating total loss over test period
                loss_test += loss_test_batch
                test_writer.add_summary(summary, b + num_batches * epoch)
            loss_test /= num_batches

            print("train_loss: {:.3f}, test_loss: {:.3f}".format(loss_train, loss_test))
            
            #implementing early stopping 
            if(loss_test > loss_test_prev):
                if (flag_increase == 1):                    
                    count_increase += 1 
                if (count_increase > PATIENCE):
                    print ("Early stopping")
                    break               
                flag_increase = 1
            else:
                flag_increase = 0
                count_increase = 0
            loss_test_prev = loss_test            
        train_writer.close()
        test_writer.close()
        saver.save(sess, MODEL_PATH)
        print("Run 'tensorboard --logdir=./logdir' to checkout tensorboard logs.")
        
'''
with shuffle
train_loss: 351.895, test_loss: 336.490

wihtout shuffle
train_loss: 357.747, test_loss: 312.767
'''           
