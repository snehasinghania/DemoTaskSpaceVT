from __future__ import print_function, division

import pickle
import numpy as np
import tensorflow as tf
import sys
from tensorflow.contrib.rnn import MultiRNNCell, LSTMCell
from tqdm import tqdm

from utils import batch_generator

SEQUENCE_LENGTH = 10
HIDDEN_SIZE = 100
KEEP_PROB = 0.8
BATCH_SIZE = 64
NUM_EPOCHS = 50
MODEL_PATH = './model21/model21'
NUM_INPUTS = 1
NUM_OUTPUTS = 1
N_LAYERS = 2
LEARNING_RATE = 0.001
PATIENCE = 5
TRAINING_SIZE = 50000

#using dst time series to predict next dst value
samples = pickle.load(open("dst_inp.pkl", "rb"))
labels = pickle.load(open("dst_label.pkl", "rb")) 
samples = samples[:TRAINING_SIZE]
labels = labels[:TRAINING_SIZE]

print (len(samples))
#splitting data into 80-20
train_index = int(round((0.8*len(samples)),0))
xtrain = samples[:train_index]
ytrain = labels[:train_index]
xtest = samples[train_index:]
ytest = labels[train_index:]

print(xtrain.shape)
print(xtest.shape)
print(len(xtrain))
print(len(xtest))

#--------------------------------------------------------------------------------------------
#defining the input and true labels placeholders
X = tf.placeholder(tf.float32, shape=(None, SEQUENCE_LENGTH, NUM_INPUTS), name="inputs")
Y = tf.placeholder(tf.float32, shape=(None, NUM_OUTPUTS), name="labels")


#Defining the stacked LSTM cells
lstm_cells = [LSTMCell(HIDDEN_SIZE, forget_bias=1.0, use_peepholes=True, activation=tf.nn.relu) for _ in range(N_LAYERS)]
stacked_lstm = MultiRNNCell(lstm_cells)
#Defining rnn layer to get the annotations
rnn_outputs, states = tf.nn.dynamic_rnn(stacked_lstm, inputs=X, dtype=tf.float32, time_major=False)
tf.summary.histogram('rnn_outputs', rnn_outputs)

# Dropout
drop = tf.nn.dropout(rnn_outputs, KEEP_PROB)

#rnn_outputs are outputs in the column format. We need only the final output from RNN layer to get the predicted value. Hence taking a transpose to extract the final output
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
                x_batch = batch_x.reshape((BATCH_SIZE, SEQUENCE_LENGTH, NUM_INPUTS))
                y_batch = batch_y.reshape((BATCH_SIZE, NUM_OUTPUTS))
                
                #running the model on train data
                loss_tr, _, summary = sess.run([loss, optimizer, merged],
                                                    feed_dict={X: x_batch,
                                                               Y: y_batch})
                
                #accumulating total loss over training period
                loss_train += loss_tr
                train_writer.add_summary(summary, b + num_batches * epoch)            
            loss_train /= num_batches
            
            # Testing
            num_batches = xtest.shape[0] // BATCH_SIZE
            for b in tqdm(range(num_batches)):
                batch_x, batch_y = next(test_batch_generator)
                x_batch = batch_x.reshape((BATCH_SIZE, SEQUENCE_LENGTH, NUM_INPUTS))
                y_batch = batch_y.reshape((BATCH_SIZE, NUM_OUTPUTS))
                
                #running the model on the test data to get the predictions
                loss_test_batch, summary, predictions = sess.run([loss, merged, prediction],
                                                         feed_dict={X: x_batch,
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
      
#with shuffle - model11       
#loss: 38.087, val_loss: 50.626 

#without shuffle - model12
#51.063, test_loss: 58.054       
