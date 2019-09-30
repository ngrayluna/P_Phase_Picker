#!/bin/usr/python3
#------------------------------------------------------------------------------
# File Name: tools.py
# Author: Noah Grayson Luna <nluna@berkeley.edu>
#------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import obspy
from obspy import UTCDateTime, Stream, Trace
from obspy.clients.fdsn import Client
from obspy.core.event import read_events
from obspy.clients.fdsn.mass_downloader.domain import CircularDomain
#from obspy.clients.fdsn.mass_downloader import Restrictions

from tools import *
from math import sin, cos, sqrt, atan2, radians

plt.style.use('ggplot')



def save_fig(EventID, save_output_fname, tight_layout=True, fig_extension="png", resolution=300):
    """
    Saves plots in folders specified by the fig_id

    Parameters
    ----------
    fig_id: str
        Identifer of event. This is the name of the directory where plot will be saved.
    save_output_fname: str
        Name of image.
    fig_extenstion: str, optional
        Default image format is .png  . Could save as .pdf, see matplotlib for more information.
    Returns
    -------
    Saves images.
    """
    # Where to save figures
    PROJECT_ROOT_DIR = "."
    ID = EventID
    IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, ID)

    # Check if folder exists:
    if not os.path.isdir(IMAGES_PATH):
        os.makedirs(IMAGES_PATH)

    path = os.path.join(IMAGES_PATH, save_output_fname + "." + fig_extension)

    print("Saving figure", EventID)

    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, dpi=resolution, bbox_inches='tight')




def plot_3comp(st, EventID, print_file, fig_extension="png", resolution=300):

    """
    Plots translational data with option to save images.

    Parameters
    ----------
    st: Obspy.Stream object
        Stream containting three component translational data

    EventID: str
        Event identifier

    print_file: Boolean
        Whether or not to save image.

    fig_extenstion: str, optional
        Default image format is .png  . Could save as .pdf, see matplotlib for
        more information.

    Returns
    -------
    """

    delta = st[0].stats.delta
    x = np.arange(0, st[0].stats['endtime'] - st[0].stats['starttime']\
                  + delta,delta)

    rows = len(st)
    fig, axes = plt.subplots(ncols=1, nrows=rows,  figsize=(12,12))
    for i in range(len(st)):
        axes[i].plot(np.arange(0, st[i].stats['endtime'] - \
                               st[i].stats['starttime'] + \
                               delta, delta),st[i], 'black', linewidth = 1.0)
        axes[i].set_title(EventID+\
                          ' '+st[i].stats.network+'.'+st[i].stats.station+\
                          '.'+st[i].stats.channel+'.'+st[i].stats.location)
        fig.tight_layout()


    if print_file == True: 
    
        # Where to save figures
        PROJECT_ROOT_DIR = "./"
        FOLDER_NAME = 'images'
        IMAGEDIR = os.path.join(PROJECT_ROOT_DIR, FOLDER_NAME)
        
        # File name
        save_output_fname = EventID +'.'+st[0].stats.network+'.'+ st[0].stats.station + '.'
        IMAGES_PATH = os.path.join(IMAGEDIR, save_output_fname)
        
        # Check if folder exists:
        if not os.path.isdir(IMAGEDIR):
            os.makedirs(IMAGEDIR)
        
        # Check if image exists:
        if os.path.exists(IMAGES_PATH):
            print('Skipping ' + save_output_fname)
            return
                    
        print("Saving figure", EventID)
            
        plt.tight_layout()
        plt.savefig(IMAGES_PATH, dpi=resolution, bbox_inches='tight')




def find_max(gen_stream, stream_type):
    """
    Find maximum amplitude of event using Obspy.trace.max()
    We assume three component translational recordings
    We assume either one or three components recordings

    Parameters
    ----------
    gen_stream: Obspy.Trace object
        Three component translational Obspy.Stream object or
        either a single or three component rotational
        Obspy.Stream object

    stream_type: str
        Specify if it is translational data or rotational
        stream.


    Returns
    -------
    max_amp0, max_amp1,max_amp2: float
        Max amplitude value for each component

    """
    # Translational
    if stream_type[0].lower() == 't':
        channels = [gen_stream[0].stats.channel, gen_stream[1].stats.channel,\
                    gen_stream[2].stats.channel]

        for ttrace in gen_stream:
            if ttrace.stats.channel == channels[0]:
                tmax_amp0 = ttrace.max()
            elif ttrace.stats.channel == channels[1]:
                tmax_amp1 = ttrace.max()
            else:
                tmax_amp2 = ttrace.max()

        return tmax_amp0, tmax_amp1, tmax_amp2

    # Rotational
    if stream_type[0].lower() == 'r':
        for rtrace in gen_stream:
            if len(gen_stream) == 1:
                rmax_amp0 = rtrace.max()
                return rmax_amp0
            else:
                rmax_amp0 = rtrace.max()
                rmax_amp1 = rtrace.max()
                rmax_amp2 = rtrace.max()
            return rmax_amp0, rmax_amp1,rmax_amp2


def check_if_data_exists(client, events_found,network,station,loc,channels,event_duration):
    """
    Check Data Center if the data can be retrieved for given event specified by its
    UTC Time. If it can not find data and/or encounters a problem, it will toss out
    event from event list (given by events_found).

    Parameters
    ----------
    events_found:
    network: str
        Network of seismic station.
    station: str
        Name of seismic station.
    loc: str
        Station location code.
    channels: list, str
        Station's channels

    event_duration: int
        Minutes after origin time.

    Returns
    -------
    resource_ids:
    UTC_UTCTimes: list, UTC-based datetime object.
        UTC Times for events found on the network server.

    floats_magnitudes: list, float
        Magnitude for events found on the network server.

    floats_latitudes : list, float
        Latitudes for events found on the network server.

    floats_longitudes: list, float
        Longitudes for events found on the network server.
    """
    UTCTimes_tmp  = []; resource_ids   = [];
    latitudes_tmp = []; longitudes_tmp = [];
    magnitudes_tmp = [];

    for event in events_found:
        max_str_event   = str(event)
        starttime_check = UTCDateTime(max_str_event[7:34])
        endtime_check   = UTCDateTime(starttime_check) + (60 * event_duration)

        try:
            stream = client.get_waveforms(network=network,station=station,\
                                          location=loc,channel=channels,\
                                          starttime =starttime_check, endtime=endtime_check)

            resource_ids.append(str(event.resource_id)[-14:])
            UTCTimes_tmp.append(max_str_event[7:34])
            try:
                magnitudes_tmp.append(float(max_str_event[57:65]))
            except ValueError:
                magnitudes_tmp.append(float(max_str_event[57:61]))
            latitudes_tmp.append(max_str_event[37:44])
            longitudes_tmp.append(max_str_event[47:54])

        except Exception as e:
            print("There was an error retrieving data for event: " + max_str_event[7:34] )
            print()

    #Convert lists of strings to floats
    UTC_UTCTimes = [UTCDateTime(n) for n in UTCTimes_tmp]
    floats_magnitudes = [float(j) for j in magnitudes_tmp]
    floats_latitudes  = [float(l) for l in latitudes_tmp ]
    floats_longitudes = [float(m) for m in longitudes_tmp]

    return resource_ids, UTC_UTCTimes, floats_magnitudes, floats_latitudes, floats_longitudes


def find_regional_station_distance(lat1,long1,lat2,long2):
    """
    Distance based on the Haversine formula, which assumes the earth is a sphere.

    Parameters
    ----------
    lat1,long1: float
        Either station or event coordinates.
    lat2,long2: float
        Either station or event coordiantes.

    Returns
    -------
    distance: float
        Distance in km rounded to the third decimal place
    """

    # approximate radius of earth in km
    R = 6373.0

    lat1_rad = radians(lat1)
    lon1_rad = radians(long1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(long2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c

    return round(distance, 3)


def make_amp_mag_plots(stream, comp0_values, comp1_values, comp2_values, floats_magnitudes, criteria, mtype, print_file):
    """
    Plots amplitudes as a function of their magnitude for largest
    earthquakes found in catalogue criteria.

    Parameters
    ----------
    stream: Obspy.Stream object
        n-component

    comp_values: list, float

    floats_magnitudes: list, float

    criteria: str

    mtype: str

    print_file: Boolean
        Save plots in current directory.

    Returns
    -------
    .png file. If print_file = True, .png is saved in local
    working directory.

    """

    comp_lists_mag = [comp0_values, comp1_values, comp1_values]
    numrows = len(comp_lists_mag)

    #Size of figures
    size = 8

    #Check if there is data for more than component
    if len(stream) == 1:
        numrows = 1
        fig, ax = plt.subplots(nrows=numrows, ncols=1, figsize=((size,size)))
        ax.scatter(floats_magnitudes, comp0_values, color='black')
        ax.set_title('{}{}{}'.format(stream[0].stats.network+'.'+stream[0].stats.station+\
                                     '.'+stream[0].stats.channel+\
                                     ' ',len(floats_magnitudes), ' '+ criteria.capitalize()\
                                     +' EQs in Catalogue Search Critera'))

        ax.set_ylabel('Amplitude')
        ax.set_xlabel('Magnitude [Ml or ?]')
        fig.tight_layout()
    else:
        numrows = len(comp_lists_mag)
        fig, axes = plt.subplots(nrows=numrows, ncols=1, figsize=((size,size)))
        row = 0
        for magnitude_val in comp_lists_mag:
            axes[row].scatter(floats_magnitudes, magnitude_val, color='black')

            axes[row].set_title('{}{}{}'.format(stream[row].stats.network+'.'+\
                                                stream[row].stats.station+'.'+\
                                                stream[row].stats.channel+\
                                                ' ', len(floats_magnitudes),' '\
                                                +criteria.capitalize() + ' EQs in Catalogue Search Critera'))
            axes[row].set_ylabel('Amplitude')
            row += 1
            axes[numrows - 1].set_xlabel('Magnitude [Ml or ?]')
        fig.tight_layout()

    #Save plots
    if print_file == True:
        print("Saving amplitude plots in current directory.")

        #Current directory
        PROJECT_ROOT_DIR = '.'

        #Save as .png
        fig_extension ='png'

        #Output file name
        save_output_fname = mtype + 'catfound_' + criteria + 'eq'

        #Path
        path = os.path.join(PROJECT_ROOT_DIR, save_output_fname + "." + fig_extension)
        plt.savefig(path, bbox_inches='tight')


def save_cat_search_results(UTCTime ,magnitude, latitude , longitude):
    """
    Writes UTC_Time, latitude, longitude, and magnitude of each event
    into a .txt file. File is saved in folder labeled 'cat_results'
    in current working directory.

    Parameters
    ----------
    UTCTime: UTC-based datetime object.

    magnitude: list, float
        List of magnitudes found from catalogue search.

    latitude: list, float
        List of latitudes for each given event.

    longitude: list, float
        List of longitudes for each given event.

    Returns
    -------
    .txt files with event information.

    """

    PROJECT_ROOT_DIR = "."
    CAT_PATH = 'cat_results'
    save_path = os.path.join(PROJECT_ROOT_DIR, CAT_PATH)

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    cat_path = os.path.join(save_path, str(UTCDateTime(UTCTime))[:11] +'_info.txt')

    # Create file
    cat_info = open(cat_path,'w')

    cat_info.write('-'*40)
    cat_info.write(("\n"))
    cat_info.write("Origin Time: " + str(UTCDateTime(UTCTime)))
    cat_info.write(("\n"))
    cat_info.write("Magnitude [Ml or ?]: " +  str(magnitude))
    cat_info.write(("\n"))
    cat_info.write("Latitude: " + str(latitude) +\
                   '|'+" Longitude: "+ str(longitude) +'\n')
    cat_info.write(("\n"))

    cat_info.close()

    print('-'*40)
    print("Origin Time: " + str(UTCDateTime(UTCTime)) )
    print("Magnitude [Ml or ?]: " +  str(magnitude))
    print("Latitude: " + str(latitude) + '|'+" Longitude: "+ str(longitude) +'\n')


def compute_event_dist(fltlat,fltlong,stnlat,stnlong):
    """
    Determines the distance between the station and earthquakes source given
    a list of event latitude and longitudes and the station latitude and
    longitude.

    Parameters
    ----------
    fltlat: list, floats
        list of event latitudes.

    fltlong: list, floats
        list of event longitudes

    stnlat: float
        station latitude

    stnlong: float
        station longitude

    Output
    ------
    list of distances from station to event. Distances found based on the
    Haversine formula. See find_regional_station_distance function for more
    information.
    """
    # Compute distances
    event_to_station_distance = []
    for lat, lon in zip(fltlat, fltlong):
        distance = find_regional_station_distance(stnlat,stnlong,lat,lon)
        event_to_station_distance.append(distance)
    return event_to_station_distance


def make_mag_dist_plt(evntdis,fltmag,UTCTimes,crit,saveplt=False):
    """
    Plots magnitudes of events as a fuction of distance from the station
    coordinates.

    Parameters
    ----------
    evntdis: float
        list of distances bewtween the station and earthquake events.

    fltmag: list, float
        list containing magnitudes of events

    UTCTimes: list of UTC_Time Objects
        list containing the UTC event times

    crit: string
        String specifying criteria used to search through catalogue. Current
        options are 'smallest' and 'largest'.

    saveplt: boolean
        If true, saves plot in working directory.

    Returns
    -------
    .png of event magnitude as a function of their distance. If saveplt is
    set to true, plot will be saved in current working directory.

    """


    # Plot magnitude and distance
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))

    for i in range(0,len(fltmag)):
        ax.annotate(str(UTCTimes[i])[:10] ,xy=(evntdis[i],fltmag[i]),\
                    textcoords='offset points',xytext = (0,0), fontsize = 10)

    ax.scatter(evntdis, fltmag)
    ax.set_ylabel('Magnitude [Ml or ?]')
    ax.set_xlabel('Distance from station [km]')
    ax.set_title('Magnitude & Distances of {} Regional Events of EQ Cat'.format(crit.title()))



    if saveplt == True:
        #Current directory
        PROJECT_ROOT_DIR = '.'

        #Save as .png
        fig_extension ='png'

        #Output file name
        save_output_fname = 'magnitude_distance'

        #Path
        path = os.path.join(PROJECT_ROOT_DIR, save_output_fname + "." + fig_extension)
        plt.savefig(path, bbox_inches='tight')

        print("Saving {}.png".format(save_output_fname))


def find_noise_times(df):
    """
    Calculates and stores start and end time of noise to later add to stations

    Input is a pandas.Dataframe with columns 'phase_time'
    """


    #Make a copy to not mess with original
    dfcopy = df.copy()

    # Sort by phase time
    dfcopy.sort_values(by=['evtime'], inplace = True)

    # Sort by station
    dfcopy.sort_values(by=['stnm'], inplace = True)

    # Calculate the differences between phase times
    timesdif = dfcopy['phase_time'].diff()

    # If zero (i.e same station), replace with previos nonzero term
    timesdif = timesdif.replace(to_replace=0, method='ffill')

    # Fill NaN with zeros
    timesnonan = timesdif.fillna(0)

    # Make all values positive.
    timesdif = timesdif.abs()

    # Store phase times as Series, a bit easier to work with
    phase_times = dfcopy['phase_time']


    # Time to find noise based on start and end times of events
    noisestatime = []
    noiseendtime = []

    for time,dif in zip(phase_times, timesnonan):
        # Start time of previous event
        t0 = time - (dif)
    
        # Check if record is less than an hour long
        if dif < 3600 and dif != 0.0:
            # start 5 minutes after event time
            noisttart = t0 + 300
            noisestatime.append(noisttart)
        
            # get a 5 minute record
            noiseend = noisttart + 300
            noiseendtime.append(noiseend)

        else:        
            # Calculate middle of time window between events to grab data from
            # Start 10 minutes before this calculated time (i.e. 600 seconds)
            noisttart = (t0 + (dif/2)) - (600/2)
            noisestatime.append(noisttart)
        
            # Get a 10 minute record
            noiseend  =  noisttart + (600)
            noiseendtime.append(noiseend)
    
    
    dfcopy['noise_startt'] = pd.Series(noisestatime, index = dfcopy.index)
    dfcopy['noise_endt']   = pd.Series(noiseendtime, index = dfcopy.index)

    return dfcopy


def apply_highpass(stream, freq):
    """
    Parameters
    ----------
    stream:

    freq: 

    Returns
    -------
    Function returns original Stream object but with traces detrended, 
    and high passed filtered.
    """
    
    stream_copy = stream.copy()
    
    traces_processed = []
    
    for tr in stream_copy:      
        # Detrend
        tr.detrend()
        
        # Apply filter
        tr.filter('highpass', freq = freq)
        
        traces_processed.append(tr)
    
    return traces_processed




def modify_noise_traces(stream, noise_stream):
    """
    Parameters
    ----------
    stream: Obspy stream object

    noise_stream: Osbspy stream object

    Returns
    -------

    """
    
    modified_traces = []
    
    for i in range(len(stream)):
        
        # Get npts from true trace
        npts = stream[i].stats.npts
    
        # Calculate durtion for the clean traces, use for trimmin noise traces
        duration = stream[i].stats.endtime.timestamp - stream[i].stats.starttime.timestamp

        if len(stream) != len(noise_stream):
            return Exception
        
        try:
            # Start time of the noise trace, needed for cutting
            starttime = stream[i].stats.starttime

            # Change starttime
            noise_stream[i].stats.starttime = starttime
    
            # Cut the trace so it is as long as the clean trace
            noise_stream[i].trim(starttime = starttime, endtime = starttime + duration)
        
            #Normalize the traces as well
            #noise_stream[i].normalize()
        
            # Change npts for the noise trace
            noise_stream[i].stats.npts = npts
        
            # Detrend the noise, too?
            noise_stream[i].detrend()

    
            modified_traces.append(noise_stream[i])
        
        except Exception as e:
            print('Could not modify file: ', noise_stream)
    
    return Stream(modified_traces)


def add_traces(clean_stream, noise_stream, wtype, eventID):
    """
    Input is two stream objects that you want to combine

    Parameters:
    -----------
    clean_stream: Obspy Stream object

    noise_stream: Obspy Stream object

    eventID: string

    Output: 
    -------
    Saves combined trace as an mseed
    """
    
    combined_trace_list = []
    
    for i in range(len(clean_stream)):
        # Add numpy arrays then convert to Trace
        combined_trace = Trace(clean_stream[i].data + noise_stream[i].data)

        # Normalize trace
        #combined_trace.normalize()
        
        combined_trace.stats.network  = clean_stream[i].stats.network
        combined_trace.stats.station  = clean_stream[i].stats.station
        combined_trace.stats.location = clean_stream[i].stats.location
        combined_trace.stats.channel  = clean_stream[i].stats.channel
        combined_trace.stats.starttime = clean_stream[i].stats.starttime
        combined_trace.stats.sampling_rate = clean_stream[i].stats.sampling_rate
        combined_trace.stats.delta = clean_stream[i].stats.delta
        
        # Put into list to convert into Stream
        combined_trace_list.append(combined_trace)
    
    #Make into stream
    final_stream = Stream(combined_trace_list)

    # Output file directory
    output_file_dir = './combined_mseed_test' 

    # Check if folder exists:
    if not os.path.isdir(output_file_dir):
        os.makedirs(output_file_dir)
    
    #Name of output file
    output_name = 'combined' + '_' + final_stream[0].stats.network + '_' + final_stream[0].stats.station + '_' + wtype + '_'+ eventID

    output = os.path.join(output_file_dir, output_name)

    # Skip duplicates
    if os.path.exists(output):
        print("Skiping " + output)
        return
    
    final_stream.write(output + '.mseed', format = 'MSEED')
    print('writing ' + output)

    return output_name
