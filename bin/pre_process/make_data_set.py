#!/bin/usr/python3
#------------------------------------------------------------------------------
# File Name: make_data_sets.py
# Author: Noah Grayson Luna <nluna@berkeley.edu>
#
# Dependencies:
# [1] mseed files of clean traces in mseed file format
# [2] mseed files of noisy traces in mseed file format
# [3] mseed files of test set in mseed file format
#
# Reads in .mseed data from specified directory and does some pre-processing:
# 0) detrend data
# 1) high pass filter is applied with max frequency of 1/2
# 2) traces are normalized
# 3) traces are trimed 30 seconds before the p-arrival and 60 seconds after
# Traces are then saved as a .npz file
#
# Notes:
# [1] Currently only reading in 2014 - 2018
# [2] Takes a while to make the training (clean and the noisy variation) and the
# 	  test set. Consider separating the two.
# [3] Manually specifiy the sampling rate, delta, and number of samples (npts)
#     in lines 178 - 182		
#------------------------------------------------------------------------------


import os
from obspy.clients.fdsn import Client
from obspy import read
from obspy import UTCDateTime
from obspy import Stream
from obspy.core.event import read_events

from tools import apply_highpass
import numpy as np
import pandas as pd

import glob


def normalize_and_store(stream):
	"""
	Normalize traces and keeps record of maximum amplitudes for future reference.

	: param		stream: Obspy Stream Object

	: return	list_trace: list containing Obspy Traces in Stream Object
	: return 	max_amplitudes:  list of max amplitude for each Trace in Stream 
				Object.
	"""
	
	list_traces = []
	max_amplitudes = []

	for i in range(len(stream)):
		max_amp = stream[i].max()
		normalized  = stream[i].data / abs(max_amp)

		# Max amplitudes
		max_amplitudes.append(max_amp)

		# Stream
		list_traces.append(normalized)

	return list_traces, max_amplitudes


def proc_wave_4train(training_file, channel):
	"""
	Read pathname and use ObsPy's read fn. to read in stream. Stream
	is then filtered by specified channel, high-passed filtered, 
	decimated, and trimmed s.t. all events start 30 seconds before
	the event time and 1 minute after.

	:param		training_file: mseed .str path
	:param 		channel: str specifying channel, check .mseed for
						 appropriate options.


	:return		Stream ObsPy object
	"""
	
	# Read traces in
	st = read(training_file)

	# Select BH* Channels
	st_sel = st.select( channel = channel)

	# Detrend and apply high pass to clean waveforms
	max_freq = (1/2)
	st_selhp = Stream(apply_highpass(stream = st_sel, freq = max_freq ))

	# Decimate
	st_selhp.decimate(factor = 2, strict_length=False)

	# Cut both traces 30 seconds before event time and 1 minute after.
	st_selhp.trim(starttime = st_selhp[0].stats.starttime + 30, endtime = st_selhp[0].stats.endtime - (3 * 60))

	return st_selhp


def check_npts(stram_ob):
	"""
	Check each Trace in ObsPy Stream that they have the same number of
	samples (npts).


	:param		stream_ob: ObsPy Stream object

	:return		if each trace has the correct number of npts, return
				Stream. Else, return False (0).
	"""

	index = 0
	indices = []

	for trace in stram_ob:
		if trace.stats.npts != 1800:
			print("Index: {} has trace with {} npts".format(index, trace.stats.npts))
			indices.append(index)
			index += 1
		else:
			index += 1
			continue

	# Delete those where npts != 1800
	for i in sorted(indices, reverse=True): 
		del stram_ob[i]

	if len(stram_ob) > 0:
		return stram_ob
	else:
		return 0


def remove_empty_df(years):
	"""
	Check Pandas DataFrame is not empty. If yes, remove from list of 
	DataFrames to index through.

	:param 		years: str specifying dataframe by  year

	:return 	years: return list of years which are not empty
	"""

	years2remove = []
	for year in years:
		df = pd.read_pickle(open("df" + str(year) +"inf_all.pkl", "rb"))

		if len(df) == 0:
			years2remove.append(year)

	for remove in years2remove:
		years.remove(remove)

	return years





def main():
	# Absolute path of where .mseed files are
	mseed_dir = '/Users/Luna/Documents/Master_Thesis/denoise_files/DATA_SETS/ncedc_all/mseed'
	noise_dir = '/Users/Luna/Documents/Master_Thesis/denoise_files/DATA_SETS/ncedc_all/combined_mseed_test'

	# Empty lists need in for loop
	tmp_rows  = []
	tmp_noise = []
	df_rows = []
	tmp_Clab = []
	tmp_Nlab = []

	eventnum = 0

	# Specify the sampling rate, npts, and delta of Traces.
	sampling_rate = 20.0
	npts = 1800 
	delta = 0.05
	time = np.arange( 0, npts / sampling_rate, delta)

	# Column names for DataFrame
	columns = ['network', 'stnm', 'channel', 'dist','phase','phase_time','onset',\
	'polarity','evaluation_mode','evaluation_status','evtime','evla',\
	'evlo','mag','mag_type','eventId','orig_fname','noise_fname','comb_fnam']

	# Let's try this for all the years we have
	years = [2014 + i for i in range(5)]

	# Check the DataFrames are not empty
	years = remove_empty_df(years)

	for year in years:
		df = pd.read_pickle(open("df" + str(year) +"inf_all.pkl", "rb"))

		# Remove events which don't have a file associated with them.
		df = df[df["orig_fname"] != "fname"]

		try:
			# Let's first shuffle the data frame.
			df = df.sample(frac = 1)
		except Exception as e:
			continue

		for ix, row in df.iterrows():

			orig_fname = row.orig_fname
			comb_fname = row.comb_fname
			phase_time = row.phase_time.timestamp

			clean_fname = os.path.join(mseed_dir, orig_fname)
			noise_fname = os.path.join(noise_dir, comb_fname)
	       

			try:
				stream_clean = proc_wave_4train(clean_fname, channel = 'BH*')
				stream_noise = proc_wave_4train(noise_fname, channel = 'BH*')

				stream_clean = check_npts(stream_clean)
				stream_noise = check_npts(stream_noise)

				# Calculate phase arrival (in seconds)
				pphase_arrival = abs(stream_clean[0].stats.starttime.timestamp - phase_time)

				# Convert from seconds to npts
				p_phase_conver = pphase_arrival * sampling_rate

				if (len(stream_clean) == len(stream_noise)) and (len(stream_clean) != 0):
					norm_an_procCL, max_ampsC = normalize_and_store(stream_clean)
					norm_an_procNO, max_ampsN = normalize_and_store(stream_noise)

					trace0, trace1, trace2 = norm_an_procCL[0], norm_an_procCL[1], norm_an_procCL[2] # clean traces
					tr0_MC, tr1_MC, tr2_MC = max_ampsC[0]     , max_ampsC[1]     , max_ampsC[2]      # amps for norm.

					tracn0, tracn1, tracn2 = norm_an_procNO[0], norm_an_procNO[1], norm_an_procNO[2] # noisy traces
					tr0_MN, tr1_MN, tr2_MN = max_ampsN[0]     , max_ampsN[1]     , max_ampsN[2]      # amps for norm.

					# P-phase arrival is in npts
					tmp_rowclean = [trace0, trace1, trace2]
					tmp_labclean = [p_phase_conver, tr0_MC, tr1_MC, tr2_MC]
					tmp_rows.append(tmp_rowclean)
					tmp_Clab.append(tmp_labclean)

					tmp_rownoise = [tracn0, tracn1, tracn2]
					tmp_labnoise = [p_phase_conver, tr0_MN, tr1_MN, tr2_MN]
					tmp_noise.append(tmp_rownoise)
					tmp_Nlab.append(tmp_labnoise)

					# DataFramce information
					df_list = row.values
					df_rows.append(df_list)

					eventnum += 1

			except Exception as e:
				print(e)

	print("We found {} events.".format(eventnum))
	# Convert data to numpy array
	clean_data_set = np.array(tmp_rows)  # clean data (no noise added)
	clean_labels   = np.array(tmp_Clab)  # noisy data
	
	noise_data_set = np.array(tmp_noise) # p, norm. values
	noise_labels   = np.array(tmp_Nlab)  # p, norm. values

	# Convert information to DataFrame
	pd.DataFrame(df_rows, columns=columns)

	print("Saving waveforms and labels...")

	# Save into a .npz for training later
	np.savez_compressed('./dataset_3comp_clean.npz', clean_data_set)
	np.savez_compressed('./dataset_3comp_cleanlabels.npz', clean_labels)
	np.savez_compressed('./dataset_3comp_noisy.npz', noise_data_set)
	np.savez_compressed('./dataset_3comp_noisylabels.npz', noise_labels)

main()