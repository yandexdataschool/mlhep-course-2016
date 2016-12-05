__author__ = 'mikhail91'

import numpy

def get_sample_weight(event):

    stat_views = event.StatNb.values * 100 + event.ViewNb.values * 10 + event.PlaneNb.values
    unique, counts = numpy.unique(stat_views, return_counts=True)

    sample_weight = numpy.zeros(len(event))

    for val, count in zip(unique, counts):

        sample_weight += (stat_views == val) * 1. / count

    return sample_weight


import matplotlib.pyplot as plt

def plot_event(event_id, data, tracks):

    event = data[data.EventID == event_id]
    track = tracks[event_id]

    event12 = event[(event.StatNb == 1) + (event.StatNb == 2)]
    event34 = event[(event.StatNb == 3) + (event.StatNb == 4)]

    track12 = track['params12']
    track34 = track['params34']

    plt.figure(figsize=(14, 10))

    plt.subplot(2,2,1)
    plt.scatter(event12.Z.values, event12.Y.values)

    for track_id in range(len(track12)):

        plt.plot(event12.Z.values, event12.Z.values * track12[track_id][0][0] + track12[track_id][0][1])

    plt.xlabel('Z')
    plt.ylabel('Y')
    plt.title('Stations 1&2')

    plt.subplot(2,2,2)
    plt.scatter(event12.Z.values, event12.X.values)

    for track_id in range(len(track12)):

        if len(track12[track_id][1]) == 0:
            continue

        plt.plot(event12.Z.values, event12.Z.values * track12[track_id][1][0] + track12[track_id][1][1])

    plt.xlabel('Z')
    plt.ylabel('X')
    plt.title('Stations 1&2')

    plt.subplot(2,2,3)
    plt.scatter(event34.Z.values, event34.Y.values)

    for track_id in range(len(track34)):

        plt.plot(event34.Z.values, event34.Z.values * track34[track_id][0][0] + track34[track_id][0][1])

    plt.xlabel('Z')
    plt.ylabel('Y')
    plt.title('Stations 3&4')

    plt.subplot(2,2,4)
    plt.scatter(event34.Z.values, event34.X.values)

    for track_id in range(len(track34)):

        if len(track34[track_id][1]) == 0:
            continue

        plt.plot(event34.Z.values, event34.Z.values * track34[track_id][1][0] + track34[track_id][1][1])

    plt.xlabel('Z')
    plt.ylabel('X')
    plt.title('Stations 3&4')
    plt.show()
