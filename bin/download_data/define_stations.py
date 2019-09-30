#!/bin/usr/python3
#------------------------------------------------------------------------------
# File Name: define_stations.py
# Author: Noah Grayson Luna <nluna@berkeley.edu>
#
# Creates .pkl file with information about stations specified in
# client.get_stations().
#
# Note:
# [1] Obspy.Client.get_stations() keyword "station" requires station names
# to be capitalized. 
#------------------------------------------------------------------------------

import os
from obspy.clients.fdsn import Client

import pandas as pd
import pickle

def main():

    # Define Northern California EQ Center
    client = Client("NCEDC")

    inv = client.get_stations(network ='*', station = 'CMB,HUMO,MNRC,PKD',\
                              channel = '*')

    stations = {}
    for net in inv:
        network = net.code
        for station in net:
            stla = station.latitude
            stlo = station.longitude
            stel = station.elevation
            stnm = station.code
            net_stnm = network + '_' + stnm
            stations[net_stnm] = (stla, stlo, stel)

    stations_pd = pd.DataFrame(stations)
    stations_pd.to_pickle('./stations.pkl')

main()


