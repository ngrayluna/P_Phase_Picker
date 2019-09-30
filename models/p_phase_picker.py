#!/bin/usr/python3
#------------------------------------------------------------------------------
# File Name: p_phase_picker.py
# Author: Noah Grayson Luna <nluna@berkeley.edu>
#
# Simple CNN model using  S-phase picker loosely based on the Generalized Phase Picker from: 
# "Ross, Z. E., Meier, M.-A., Hauksson, E., and T. H. Heaton (2018).  
# Generalized Seismic Phase Detection with Deep Learning, 
# Bull. Seismol. Soc. Am., doi: 10.1785/0120180080 [arXiv:1805.01075]"
#
# See 'model parameters' ofr specifics. For now we are using only 3 1D CNNs, Maxpooling,
# with some batch normalization.
#
#
# Dependencies: 
# -------------
# training_data_wave.npz: 	.npz file containing NumPy, three-component, 
# 							time-series waveforms.
#
# training_data_labels.npz: seismic wave arrival time, scalar value for
# 							corresponding waveform in training data.
#
#
# Outputs:
# --------
# model.json:	 				 contains saved trained model
# model.h5: 	 				 contains the weights
# ml_spicks.npz:				 P-picks made by our trained DNN
# waves4picking_file_name.npz:   Waveforms used for 'test' p-picking 
# ppicks_used.npz:				 p-picks used for 'test' set picking
# 
# images generated (in .png):	loss-val curves
#								 random selected waveforms with their original and
#								 predicted p-picks
#------------------------------------------------------------------------------


import os
import numpy as np
import pandas as pd
from random import randint
from keras.layers import Input, Dense
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout, Activation, Flatten
from keras.models import Model

import matplotlib.pyplot as plt

import mlflow
import mlflow.keras

plt.style.use('ggplot')

def main():

	# Let's read this in and check it is right.
	data_waves = np.load("./REFORM_Tdataset_3comp_clean.npz")
	data_labels = np.load("./dataset_3comp_cleanlabels.npz")   # labels

	# Convert to list so that we can work with it
	tmp_list = []
	for event in data_waves:
	    tmp_list.append(data_waves[event])

	label_list = []
	for label in data_labels:
	    label_list.append(data_labels[label])
	        

	    
	# Convert to Numpy array
	# (# of instances, # of features)
	data_array = np.array(tmp_list)
	labels = np.array(label_list)


	# From (1, 7156, 3, 1800) to (7156, 3, 1800)
	data_array = data_array.reshape(data_array.shape[1], data_array.shape[2], data_array.shape[3])
	labels = labels.reshape(labels.shape[1], labels.shape[2]) # (1, #samples, labels)


	# How it is set up: [trace0, trace1, trace2]
	# Index of Number of traces
	num_traces = data_array.shape[0]   # e.g. (1, 5, 3)

	# Index Number of features (here two waveforms + one s-arrival = 3)
	num_feat   = data_array.shape[2]

	# npts
	npts = data_array.shape[1]

	# Grab P-arrivals
	p_arrivals = labels[:,0] 

	# Split data into training and validation 
	# The percent of data that will be included in the test set (here 20%)
	TRAIN_TEST_SPLIT = 0.2

	# TRAIN/TEST split 
	nr_test_data = int(num_traces * TRAIN_TEST_SPLIT)

	x_train = data_array[nr_test_data:, :] # x composed of two traces
	y_train = p_arrivals[nr_test_data :] # y values has the s-arrival time

	x_test  = data_array[:nr_test_data, :]
	y_test  = p_arrivals[: nr_test_data]

	# Begin to track metrics used.
	with mlflow.start_run():

		# Create a ranadom test number
		test_num = randint(100,999)

		print("This is test file number {}".format(test_num))


		# Create unique directory identifier
		unique_date = '0925'

		# How decoded images do you want?
		events2print = 20

		# model_parameters 
		batch_size = 256
		epochs     = 400
		filters    = [32, 64, 128]
		#units = [8, 32, 64]
		kernel_size = 6
		pool_size = 2
		padding = 'same'
		hidden_act_fun = 'relu'
		final_act_fun  = 'linear'
		optimizer = 'adam'
		loss_type = 'mean_squared_error' # Try Huber loss function later
		name = 's_phase_picker'

		 # this is our input placeholder
		input_trace = Input(shape=(x_train.shape[1], x_train.shape[2]))  # Currently need to change npts (18453)

		x = Conv1D(filters = filters[0], kernel_size = kernel_size, activation=hidden_act_fun, padding = padding)(input_trace)
		x = MaxPooling1D(pool_size = pool_size, padding = padding)(x)
		x = Conv1D(filters = filters[1], kernel_size = kernel_size, activation=hidden_act_fun, padding = padding)(x)		
		x = MaxPooling1D(pool_size = pool_size, padding = padding)(x)
		# x = Conv1D(filters = filters[2], kernel_size = kernel_size, activation=hidden_act_fun, padding = padding)(x)
		# x = MaxPooling1D(pool_size = pool_size, padding = padding)(x)
		cnn_feat = Flatten()(x)


		# x = Dense(units= 64, activation=hidden_act_fun)(cnn_feat)
		# x = BatchNormalization()(x)
		x = Dense(units= 32, activation=hidden_act_fun)(cnn_feat)
		x = BatchNormalization()(x)
		x = Dense(units = 8, activation=hidden_act_fun)(x)
		x = BatchNormalization()(x)
		dense = Dense(units= 1, activation=final_act_fun)(x)

		# Compile S-Phase Picker Nidek
		s_phase_picker = Model(input_trace, dense, name = name)
		s_phase_picker.compile(optimizer = optimizer, loss = loss_type)
		s_phase_picker.summary()

		history = s_phase_picker.fit(x = x_train,
									y = y_train,
									epochs= epochs,
									batch_size=batch_size,
									validation_data=(x_test, y_test))


		# Keep track of the last loss values (for easy comparison later)
		train_loss = history.history['loss']
		last_train_loss_value = train_loss[len(train_loss)-1]
		val_loss = history.history['val_loss']
		last_val_loss_value = val_loss[len(val_loss) - 1]

		# Directory where to store images
		PROJECT_ROOT_DIR = './images'
		FOLDER_NAME = 'training_model' + '_' + unique_date + '.' + str(test_num)
		IMAGEDIR = os.path.join(PROJECT_ROOT_DIR, FOLDER_NAME)

		# Check if image folder exists, if yes make a another digit name:
		if os.path.isdir(IMAGEDIR):
			test_num = randint(100,999)
			FOLDER_NAME = 'training_model' + '_' + unique_date + '.' + str(test_num)
			IMAGEDIR = os.path.join(PROJECT_ROOT_DIR, FOLDER_NAME)
			os.makedirs(IMAGEDIR)
		else:
			os.makedirs(IMAGEDIR)

		# Validation loss curves #
		fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (10,6))
		axes.plot(history.history['loss'])
		axes.plot(history.history['val_loss'])
		axes.set_title('Model Loss')
		axes.set_ylabel('loss')
		axes.set_xlabel('epoch')
		axes.legend(['Train', 'Validation'], loc='upper left')

		# name of loss val file    
		model_png_name = 'model_loss' +  '_' + final_act_fun + '_' + loss_type + '.png'
		IMAGES_PATH = os.path.join(IMAGEDIR, model_png_name)
		plt.savefig(IMAGES_PATH, bbox_inches = 'tight')

		print("saving loss vs. epoch figure...")

		# fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (10,6))
		# axes.plot(history.history['acc'])
		# axes.plot(history.history['val_acc'])
		# plt.title('Model accuracy')
		# axes.set_xlabel('epoch')
		# axes.set_ylabel('accuracy')
		# axes.legend(['Training', 'Validation'], loc='lower right')

		# # name of loss val file    
		# model_png_name2 = 'accuracy' +  '_' + final_act_fun + '_' + loss_type + '.png'
		# IMAGES_PATH = os.path.join(IMAGEDIR, model_png_name2)
		# plt.savefig(IMAGES_PATH, bbox_inches = 'tight')

		## PREDICTION 'TEST' on test set ##
		# Let's do a test run and make a prediction
		ML_SPICKS = s_phase_picker.predict(x_test)


		# Save model and traces
		OUTPUT_ROOT_DIR = './data_outputs'
		FOLDER_NAME = 'training_model' + '_' + unique_date + '.' + str(test_num)
		DATADIR = os.path.join(OUTPUT_ROOT_DIR, FOLDER_NAME)


		# Check if folder exists, if yes make an another digit name:
		if os.path.isdir(DATADIR):
			test_num = randint(100,999)
			FOLDER_NAME = 'training_model' + '_' + unique_date + '.' + str(test_num)
			DATADIR = os.path.join(OUTPUT_ROOT_DIR, FOLDER_NAME)
			os.makedirs(DATADIR)
		else:
			os.makedirs(DATADIR)


		# picks made from DNN (output from training)
		spicked_file_name = 'ml_spicks'+ '_'+ unique_date + '.' + str(test_num) +'.npz'
		SPICKED_PATH_NAME = os.path.join(DATADIR, spicked_file_name)
		np.savez_compressed(SPICKED_PATH_NAME, ML_SPICKS)

		# Waveforms used for test s-picking  (This is only temporary)
		waves4picking_file_name = 'waves4picking_file_name'+ '_'+ unique_date + '.' + str(test_num) +'.npz'
		W4PICKING_PATH_NAME = os.path.join(DATADIR, waves4picking_file_name)
		np.savez_compressed(W4PICKING_PATH_NAME, x_test)

		# s-picks used for test set picking
		spicks_used4test = 'ppicks_used'+ '_'+ unique_date + '.' + str(test_num) +'.npz'
		originalSpicks = os.path.join(DATADIR, spicks_used4test)
		np.savez_compressed(originalSpicks, y_test)

		## MlFlow ##
		# log parameters
		mlflow.log_param('training_size', x_train.shape[0])
		mlflow.log_param('validation_size', x_test.shape[0])
		mlflow.log_param('kernel_size', kernel_size)
		mlflow.log_param('hidden_layers', hidden_act_fun)
		mlflow.log_param('output_layer', final_act_fun)
		mlflow.log_param('epochs', epochs)
		mlflow.log_param('optimizer', optimizer)
		mlflow.log_param('loss_function', loss_type)

		# log metrics
		mlflow.log_metric("train_loss", last_train_loss_value)
		mlflow.log_metric("validation_loss", last_val_loss_value)

		# log artifacts
		mlflow.log_artifacts(IMAGEDIR, "images")
		mlflow.log_artifacts(DATADIR, "data_outputs")

		# log model
		mlflow.keras.log_model(s_phase_picker, "models")

		print('Saving model and model weights...')

		# Serialize model to JSON
		model_file_name = "model.json"
		MODEL_PATH_NAME = os.path.join(DATADIR, model_file_name)
		
		# DOUBLE CHECK THIS IS WRITTING TO WRITE FOLDER
		model_json = s_phase_picker.to_json()
		with open(MODEL_PATH_NAME, "w") as json_file:
			json_file.write(model_json)

		# Serialize weights to HDF5
		weights_file_name = "model.h5"
		WEIGHT_PATH_NAME = os.path.join(DATADIR, weights_file_name)
		s_phase_picker.save_weights(WEIGHT_PATH_NAME)
		print("Saved model to disk")



main()