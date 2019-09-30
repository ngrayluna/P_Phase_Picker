#!/bin/usr/python
#------------------------------------------------------------------------------
# File Name: download_tools.py
# Author 1: Qingkai Kong
# Author 2: Noah G. Luna
#
# Functions used for various stages of mass file query, downloading and format.
# Do not modify without checking dependies of files within main directory!
#------------------------------------------------------------------------------


import os
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from obspy.core.event import read_events
from joblib import Parallel, delayed

import pandas as pd
from datetime import datetime
import pickle

from tools import find_regional_station_distance

import glob
import errno

def get_station_pick(filename, stations):
    station_picks = []
    print(filename)
    cat = read_events(filename)
    # loop through the picks and
    for n, event in enumerate(cat):

        if len(event.origins) == 0:
            continue
        elif len(event.origins) > 1:
            print('More origins present %d'%(len(event.origins)))
            origin_info = []

            for ooo in event.origins:
                origin_info += ooo.arrivals
        else:
            origin_info = event.origins[0].arrivals

        for i, pick in enumerate(event.picks):
            waveform_id = pick['waveform_id']
            stnm = waveform_id['station_code']
            channel = waveform_id['channel_code']
            network = waveform_id['network_code']
            onset = pick.onset
            polarity = pick.polarity
            evaluation_mode = pick.evaluation_mode
            evaluation_status = pick.evaluation_status
            phase_time = pick.time
            origin = event.origins[0]
            evtime = origin.time
            evla = origin.latitude
            evlo = origin.longitude
            try:
                stla, stlo, stel = stations[network + '_' + stnm]
            except:
                print('Can not find %s, %s'%(network, stnm))
                continue

            dist = find_regional_station_distance(evla, evlo, stla, stlo)
#            if dist > 30:
#                continue

            mags = event.magnitudes[0]
            mag = mags.mag
            mag_type = mags.magnitude_type
            eventId = event.resource_id.id.split('/')[-1]

            arrival = origin_info[i]
            pick_id = pick.resource_id
            pid = arrival['pick_id']

            if pid != pick_id:

                new_arrival = \
                    [arrival for arrival in origin_info if arrival['pick_id'] == pick_id]
                if len(new_arrival) < 1:
                    continue
                phase = new_arrival[0]['phase']
            else:
                phase = arrival['phase']

            station_picks.append([network, stnm, channel,dist, phase, phase_time, onset, polarity, evaluation_mode, evaluation_status, \
                                    evtime, evla, evlo, mag, mag_type, eventId])

    return station_picks

def get_station_data(row, stations, client):
    t = row.phase_time
    # Original time window for clean & noisy traces
    # These were 5 minute records
    t0 = t - 60
    t1 = t + 240

    net = row.network
    stnm = row.stnm
    comp = row.channel
    onset = row.onset
    status = row.evaluation_status
    mode = row.evaluation_mode
    timestamp = str(int(t.timestamp*1000))
    eventId = row.eventId

    stla, stlo, stel = stations[net + '_' + stnm]
    if onset is None:
        return

    if (comp.lower()[-1] != 'z') & (comp[-1] != '3'):
        #print(comp)
        return

    # Output file directory
    output_file_dir = './mseed'

    # Check if folder exists:
    if not os.path.isdir(output_file_dir):
        os.makedirs(output_file_dir)


    output_name =  net + '_' + stnm + '_' \
          + '_' + onset + '_' + status + '_' + eventId

    output = os.path.join(output_file_dir, output_name)

    # skip for duplicates
    if os.path.exists(output ):
        print("Skiping " + output, comp)
        return

    #if row.onset != 'impulsive':
    #    continue
    try:
        st = client.get_waveforms(net, stnm, '*', channel='BH?,HH?', starttime=t0, endtime=t1)
    except:
        print('No data for %s, %s, %s, %s, %s'%(net, stnm, comp, t0, t1))
        return

    # put the data into the
    for tr in st:
        tr.stats.sac = {}
        tr.stats.sac.evla = row.evla
        tr.stats.sac.evlo = row.evlo
        tr.stats.sac.mag = row.mag
        tr.stats.sac.stla = stla
        tr.stats.sac.stlo = stlo
        comp = tr.stats.channel

        #tr.write(output + '_' + comp + '.SAC', format = 'SAC')
    st.write(output + '.mseed', format = 'MSEED')
    print('writing ' + output)
    dist = row.dist



def get_noise(row, stations, client):
    t = row.phase_time
    # Here we get data 20 minutes and 10 seconds before event
    t0 = row.noise_startt
    # The record will be 8 minutes (480 seconds) long
    t1 = row.noise_endt

    net = row.network
    stnm = row.stnm
    comp = row.channel
    timestamp = str(int(t.timestamp*1000))
    eventId = row.eventId

    stla, stlo, stel = stations[net + '_' + stnm]

    if (comp.lower()[-1] != 'z') & (comp[-1] != '3'):
        #print(comp)
        return

    # Output file directory
    output_file_dir = './mseed_noise'

    # Check if folder exists:
    if not os.path.isdir(output_file_dir):
        try:
            os.makedirs(output_file_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


    # File name
    output_name = './mseed_noise/' + 'noise' + '_' + net + '_' + stnm + '_' + eventId
    os.path.join(output_file_dir, output_name)

    # skip for duplicates
    if os.path.exists(output_name):
        print("Skiping " + output_name, comp)
        return

    try:
        st = client.get_waveforms(net, stnm, '*', channel='BH?,HH?', starttime=t0, endtime=t1)
    except:
        print('No data for %s, %s, %s, %s, %s'%(net, stnm, comp, t0, t1))
        return

    # put the data into the
    for tr in st:
        tr.stats.sac = {}
        tr.stats.sac.evla = row.evla
        tr.stats.sac.evlo = row.evlo
        tr.stats.sac.mag = row.mag
        tr.stats.sac.stla = stla
        tr.stats.sac.stlo = stlo
        comp = tr.stats.channel

        #tr.write(output + '_' + comp + '.SAC', format = 'SAC')
    st.write(output_name + '.mseed', format = 'MSEED')
    print('writing ' + output_name)
    dist = row.dist



# Download catalog function
def download_catalog(i, date_range, client, stations, minmag, maxrad):

    t0 = UTCDateTime(date_range[i])
    t1 = UTCDateTime(date_range[i+1])

    for coord in stations:
        station = coord
        stla, stlo, stel = stations[coord]

        output = t0.strftime('%Y-%m-%d') + '_' + t1.strftime('%Y-%m-%d') + '_' + station + '.quakeml'

        # if file exist, we just skip it
        if os.path.exists(output):
            return

        try:
            cat = client.get_events(starttime=t0, endtime=t1, latitude=stla, longitude=stlo,\
                maxradius=maxrad, minmagnitude=minmag,\
                includearrivals = True, filename = output)

        except Exception as e:
            print('There was an error in obtaining data for station', station.upper())
            print('Time stamp: ', t0.strftime('%Y-%m-%d') + '_' + t1.strftime('%Y-%m-%d') +' will be skipped\n')
