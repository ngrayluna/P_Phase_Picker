#!/bin/usr/python3
#------------------------------------------------------------------------------
# File Name: get_mass_picks.py
# Author: Noah Grayson Luna <nluna@berkeley.edu>
# 
# 
# Goes through .quakeml files and returns a readable .pkl file with the
# network, station, channel, distance, phase, phase_time, onset, polarity,
# evaluation_mode, evaluation_status, event time, event latitude, event longitude
# magnitude, magnitude type and event ID.
# Pickle files are saved on a yearly basis.
#
#
# Dependencies:
# [1] stations.pkl:     .pkl file containing stations of interest for download.
# [2] *.quakeml :       quakeml file containing catalogue search results for
#                       specified years and stations.
#------------------------------------------------------------------------------

import os
from obspy import read
from obspy import UTCDateTime
from obspy.core.event import read_events
from joblib import Parallel, delayed

from download_tools import get_station_pick

import pandas as pd
from datetime import datetime
import pickle

import glob

def main():

    # Read station information
    stations = pickle.load(open('./stations.pkl', 'rb'))

    # Define years to grab pick data from. We'll start with 5 here.
    years = ['2014', '2015', '2016', '2017', '2018']

    for year in years:
        print('Grabbing picks for year:', year)

        file = './quakeml/' + year + '-*'

        results = []
        results = Parallel(n_jobs= -1, backend="threading")\
            (delayed(get_station_pick)(item, stations) for item in glob.glob(file))

        # Store in pandas dataframe
        temp = []
        for item in results:
            temp += item

        df = pd.DataFrame(temp, columns=['network', 'stnm', 'channel',\
                                         'dist', 'phase', 'phase_time',\
                                         'onset', 'polarity', 'evaluation_mode',\
                                         'evaluation_status', 'evtime', 'evla',\
                                         'evlo', 'mag', 'mag_type', 'eventId'])

        print('Saving results to .pkl')
        # save into pkl
        df.to_pickle('df' + year + '.pkl')

main()



