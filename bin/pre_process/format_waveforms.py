#!/bin/usr/python3
#------------------------------------------------------------------------------
# File Name: format_waveforms.py
# Author: Noah Grayson Luna <nluna@berkeley.edu>
#
#
# Read in .npz and format them to have array shape:
# 		(# instances, sample points, channels)
# Note: append this to 'make_data_set.py' in future.
#
#
# Dependencies:
# [1] mseed files of clean traces in mseed file format
# [2] mseed files of noisy traces in mseed file format

#------------------------------------------------------------------------------
import os
import numpy as np



def reformat_data(data_array):
	"""
	Formate array with shape: (# instances, channels, # of sample points) to 
	(# instances, # of sample points, channels)

	:param 		data_aray: NumPy array of shape (# instances, channels, # of sample points)

	:return		new_shape: Numpy array of shape (# instances, # of sample points, channels)
	"""

	# Do the first one manually
	tr0 = data_array[0, 0, :]; tr1 = data_array[0, 1, :]; tr2 = data_array[0, 2, :]
	new_formL = [tr0, tr1, tr2]
	new_formA = np.array(new_formL)
	new_formT = new_formA.T
	new_shape = np.reshape(new_formT, newshape=(1, new_formT.shape[0], new_formT.shape[1]))

	for inst in range(1, data_array.shape[0]):
		tr0 = data_array[inst, 0, :]; tr1 = data_array[inst, 1, :]; tr2 = data_array[inst, 2, :]

		new_formL = [tr0, tr1, tr2]
		new_formA = np.array(new_formL)
		new_formT = new_formA.T
		new_shape2 = np.reshape(new_formT, newshape=(1, new_formT.shape[0], new_formT.shape[1]))

		new_shape = np.concatenate((new_shape, new_shape2))

	return new_shape



def main():
	# Read in waveforms
	data_waves = np.load("./dataset_3comp_clean.npz")

	# Convert to list so that we can work with it
	wave_list = []
	for event in data_waves:
	    wave_list.append(data_waves[event])
	    
	# Convert to Numpy array
	# (# of instances, # of features)
	data_array = np.array(wave_list)

	# From (1, 7156, 3, 1800) to (7156, 3, 1800)
	data_array = data_array.reshape(data_array.shape[1], data_array.shape[2], data_array.shape[3])

	# How it is set up: [trace0, trace1, trace2]
	# Index of Number of traces
	num_traces = data_array.shape[0]   # e.g. (1, 5, 3)

	# Index Number of features (here two waveforms + one s-arrival = 3)
	num_feat   = data_array.shape[1]

	# npts
	npts = data_array.shape[2]

	reformatted = reformat_data(data_array)

	print("Saving reformatted .npz file...")
	np.savez_compressed('./REFORM_Tdataset_3comp_clean.npz', reformatted)
main()