#!/usr/bin/python
"""
Uses saved model, so it should be executed after model*.py
"""

import matplotlib.pyplot as plt
import pickle

from model2 import *
saver = tf.train.Saver()

#the path for the saved model which was already trained
MODEL_PATH = './model21/model21'

size = 3000

with tf.Session() as sess:
    #loading the saved model
    saver.restore(sess, MODEL_PATH)
    
    #getting the data
    bz_curve = pickle.load(open("imf_data_complete.pkl", "rb"))
    all_samples = pickle.load(open("imf_inp.pkl", "rb"))
    all_labels = pickle.load(open("dst_label.pkl", "rb"))     
    
    com_len = len(all_samples)
    #getting last 3000 data points to test. The last data points are test points because the model was trained on unshuffled data
    #here its -11 because last 11 values are not considered while forming the data points in the sliding window approach
    bz_curve = np.array(bz_curve[com_len-size:-11])
    all_samples = all_samples[com_len-size:]
    all_labels = all_labels[com_len-size:]
    
    samples = all_samples.reshape((size, SEQUENCE_LENGTH, NUM_INPUTS))
    labels = all_labels.reshape((size, NUM_OUTPUTS))
    
    #running the model and getting the prediction
    test_loss, predictions = sess.run([loss, prediction], feed_dict={X:samples, Y:labels})
    
#ploting the values to analyze the predicted values
# Create plots with pre-defined labels.
fig, ax = plt.subplots()
ax.plot(np.arange(0, 3000, 1), predictions, label="Predicted DST Curve")
ax.plot(np.arange(0, 3000, 1), all_labels, label="True DST Curve")
legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')
# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('#00FFCC')
plt.savefig("fig1.png")

# Create plots with pre-defined labels.
fig1, ax1 = plt.subplots()
ax1.plot(np.arange(0, 3000, 1), bz_curve, label="True IMF Curve")
ax1.plot(np.arange(0, 3000, 1), all_labels, label="True DST Curve")
legend1 = ax1.legend(loc='upper right', shadow=True, fontsize='x-large')
# Put a nicer background color on the legend.
legend1.get_frame().set_facecolor('#00FFCC')
plt.savefig("fig2.png")
