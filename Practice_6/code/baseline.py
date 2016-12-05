__author__ = 'mikhail91'

import numpy
import pandas
from sklearn.linear_model import LinearRegression


__author__ = 'mikhail91'

import numpy
import pandas
from sklearn.linear_model import LinearRegression

class Baseline(object):

    def __init__(self, n_max_hits, n_min_hits, window_width):
        """
        This class is simple realization of a Template Matching paradigm for straight tracks in 2D.
        :param n_max_hits: int, max min number of hits to consider the track recognized.
        :param n_min_hits: int, min min number of hits to consider the track recognized.
        :param window_width: float, width of a searching window for searching hits for a track.
        :return:
        """

        self.window_width = window_width
        self.n_max_hits = n_max_hits
        self.n_min_hits = n_min_hits

    def fit(self, x, y, sample_weight=None):
        """
        Fit the method.
        :param x: numpy.ndarray shape=[n_hits, n_features], X of hits.
        :param y: numpy.array shape=[n_hits], y of hits.
        :param sample_weight: numpy.array shape=[n_hits], weight of each hits.
        :return:
        """

        used = numpy.zeros(len(x))
        labels = -1. * numpy.ones(len(x))
        track_id = 0
        tracks_params = []



        # build dict of hits
        index = numpy.arange(len(x))
        unique_x = numpy.unique(x)

        hits = {}
        for x_layer in unique_x:

            hits[x_layer] = index[x == x_layer][numpy.argsort(x[x == x_layer])]

        for n_hits in range(self.n_min_hits, self.n_max_hits+1)[::-1]:

            for n in range(len(unique_x)):


                for first in range(n):

                    last = len(unique_x) - n + first

                    if last <= first:
                        continue

                    if n_hits > last - first + 1:
                        continue


                    for first_ind in hits[unique_x[first]]:

                        for last_ind in hits[unique_x[last]]:

                            if used[first_ind] == 1 or used[last_ind] == 1:
                                continue

                            track = []

                            k = 1. * (y[last_ind] - y[first_ind]) / (x[last_ind] - x[first_ind])
                            b = y[first_ind] - k * x[first_ind]

                            track += [first_ind, last_ind]


                            for inter in range(first+1, last):

                                for inter_ind in hits[unique_x[inter]]:

                                    if used[inter_ind] == 1:
                                        continue

                                    y_pred = k * x[inter_ind] + b

                                    if y_pred + self.window_width >= y[inter_ind]:

                                        if y_pred - self.window_width <= y[inter_ind]:

                                            track += [inter_ind]
                                            break


                            if len(track) >= n_hits:

                                used[track] = 1
                                labels[track] = track_id
                                track_id += 1

                                x_track = x[track]
                                y_track = y[track]

                                if sample_weight != None:
                                    sample_weight_track = sample_weight[track]
                                else:
                                    sample_weight_track = None

                                lr = LinearRegression()
                                lr.fit(x_track.reshape(-1,1), y_track, sample_weight_track)

                                params = list(lr.coef_) + [lr.intercept_]
                                tracks_params.append(params)



        self.labels_ = labels
        self.tracks_params_ = numpy.array(tracks_params)


class SimpleTemplateMatching(object):

    def __init__(self, n_max_hits, n_min_hits, window_width):
        """
        This class is simple realization of a Template Matching paradigm for straight tracks in 2D.
        :param n_max_hits: int, max min number of hits to consider the track recognized.
        :param n_min_hits: int, min min number of hits to consider the track recognized.
        :param window_width: float, width of a searching window for searching hits for a track.
        :return:
        """

        self.window_width = window_width
        self.n_max_hits = n_max_hits
        self.n_min_hits = n_min_hits

    def fit(self, x, y, sample_weight=None):
        """
        Fit the method.
        :param x: numpy.ndarray shape=[n_hits, n_features], X of hits.
        :param y: numpy.array shape=[n_hits], y of hits.
        :param sample_weight: numpy.array shape=[n_hits], weight of each hits.
        :return:
        """

        used = numpy.zeros(len(x))
        labels = -1. * numpy.ones(len(x))
        track_id = 0
        tracks_params = []

        for n_hits in range(self.n_min_hits, self.n_max_hits+1)[::-1]:

            for first_ind in range(len(x)):

                for second_ind in range(len(x)):

                    x1 = x[first_ind]
                    y1 = y[first_ind]

                    x2 = x[second_ind]
                    y2 = y[second_ind]

                    if (x1 >= x2) or (used[first_ind] == 1) or (used[second_ind] == 1):
                        continue

                    k = 1. * (y2 - y1) / (x2 - x1)
                    b = y1 - k * x1

                    y_upper = b + k * x.reshape(-1) + self.window_width
                    y_lower = b + k * x.reshape(-1) - self.window_width

                    track = (y <= y_upper) * (y >= y_lower) * (used == 0)

                    if track.sum() >= n_hits:

                        used[track] = 1
                        labels[track] = track_id
                        track_id += 1

                        x_track = x[track]
                        y_track = y[track]

                        if sample_weight != None:
                            sample_weight_track = sample_weight[track]
                        else:
                            sample_weight_track = None

                        lr = LinearRegression()
                        lr.fit(x_track.reshape(-1,1), y_track, sample_weight_track)

                        params = list(lr.coef_) + [lr.intercept_]
                        tracks_params.append(params)


        self.labels_ = labels
        self.tracks_params_ = numpy.array(tracks_params)
