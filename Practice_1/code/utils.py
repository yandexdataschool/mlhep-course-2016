import numpy
import pandas
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib.pyplot import cm


__author__ = 'mikhail91'


def straight_tracks_generator(n_events, n_tracks, n_noise, sigma, intersection=True, x_range=(0, 10, 1),
                              y_range=(-30, 30, 1), k_range=(-2, 2, 0.1), b_range=(-10, 10, 0.1),
                              random_state=None):
    """
    This function generates events with straight tracks and noise.

    :param n_events: int, number of generated events.
    :param n_tracks: int, number of generated tracks in each event. Tracks will have TrackIDs in range [0, inf).
    :param n_noise: int, number of generated random noise hits. Noise hits will have TrackID -1.
    :param sigma: float, track's hit generated with error which has normal distribution.
                  Sigma is parameter of the distribution.
    :param intersection: booleen, if False the tracks will not intersect.
    :param x_range: tuple (min, max, step), range of x values of the hits.
    :param y_range: tuple (min, max, step), range of y values of the hits. Only for intersection=False.
    :param k_range: tuple (min, max, step), range of k values of the track. y = b + k * x.
    :param b_range: tuple (min, max, step), range of b values of the track. y = b + k * x.
    :param int random_state: random state
    :return: pandas.DataFrame
    """

    list_of_events = []

    numpy.random.seed(random_state)

    for event_id in range(n_events):

        event_tracks = []

        # Add track hits
        if intersection:

            for track_id in range(n_tracks):
                X = numpy.arange(*x_range).reshape((-1, 1))
                k = numpy.random.choice(numpy.arange(*k_range), 1)[0]
                b = numpy.random.choice(numpy.arange(*b_range), 1)[0]
                e = numpy.random.normal(scale=sigma, size=len(X)).reshape((-1, 1))
                # y = b + k * x + e
                y = b + k * X + e

                track = numpy.concatenate(([[event_id]] * len(X),
                                           [[track_id]] * len(X),
                                           X, y), axis=1)
                event_tracks.append(track)

        else:

            y = numpy.arange(*y_range)
            y_start = numpy.random.choice(y, n_tracks, replace=False)
            y_start = numpy.sort(y_start)
            y_end = numpy.random.choice(y, n_tracks, replace=False)
            y_end = numpy.sort(y_end)
            X = numpy.arange(*x_range).reshape((-1, 1))
            delta_x = X.max() - X.min()

            for track_id in range(n_tracks):
                X = numpy.arange(*x_range).reshape((-1, 1))
                k = 1. * (y_end[track_id] - y_start[track_id]) / delta_x
                b = y_start[track_id] - k * X.min()
                e = numpy.random.normal(scale=sigma, size=len(X)).reshape((-1, 1))
                # y = b + k * x + e
                y = b + k * X + e

                track = numpy.concatenate(([[event_id]] * len(X),
                                           [[track_id]] * len(X),
                                           X, y), axis=1)
                event_tracks.append(track)

        # Add noise hits
        if n_noise > 0:
            X = numpy.random.choice(numpy.arange(*x_range), n_noise).reshape((-1, 1))
            k = numpy.random.choice(numpy.arange(*k_range), n_noise).reshape(-1, 1)
            b = numpy.random.choice(numpy.arange(*b_range), n_noise).reshape(-1, 1)
            y = b + k * X
            noise = numpy.concatenate(([[event_id]] * len(X),
                                       [[-1]] * len(X),
                                       X, y), axis=1)
            event_tracks.append(noise)

        event = numpy.concatenate(tuple(event_tracks), axis=0)
        list_of_events.append(event)

    all_events = numpy.concatenate(tuple(list_of_events), axis=0)
    data = pandas.DataFrame(columns=['EventID', 'TrackID', 'X', 'y'], data=all_events)

    return data


def plot_straight_tracks(event, labels=None):
    """
    Generate plot of the event with its tracks and noise hits.

    :param event: pandas.DataFrame with one event with expected columns "TrackID", "X", "y"
    :param labels: numpy.array shape=[n_hits], labels of recognized tracks.
    :return: matplotlib.pyplot object.
    """

    plt.figure(figsize=(10, 7))

    tracks_id = numpy.unique(event.TrackID.values)
    event_id = event.EventID.values[0]

    color = cm.rainbow(numpy.linspace(0, 1, len(tracks_id)))

    # Plot hits
    for num, track in enumerate(tracks_id):

        X = event[event.TrackID == track].X.values.reshape((-1, 1))
        y = event[event.TrackID == track].y.values

        plt.scatter(X, y, color=color[num])

        # Plot tracks
        if track != -1:
            lr = LinearRegression()
            lr.fit(X, y)

            plt.plot(X, lr.predict(X), label=str(track), color=color[num])

    if labels is not None:

        unique_labels = numpy.unique(labels)

        for lab in unique_labels:

            if lab != -1:
                X = event[labels == lab].X.values.reshape((-1, 1))
                y = event[labels == lab].y.values

                lr = LinearRegression()
                lr.fit(X, y)

                X = event.X.values.reshape((-1, 1))
                plt.plot(X, lr.predict(X), color='0', alpha=0.5)

    plt.legend(loc='best')
    plt.title('EventID is ' + str(event_id))
    plt.ylabel('Y', size=15)
    plt.xlabel('X', size=15)
    plt.xticks(size=15)
    plt.yticks(size=15)