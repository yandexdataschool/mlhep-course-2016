import numpy
from sklearn import clone


__author__ = 'mikhail91'


class RANSACTracker(object):
    def __init__(self, n_tracks, min_hits, regressor):
        """
        This class uses RANSAC linear regression for the tracks reconstruction.

        :param n_tracks: int, number of searched tracks.
        :param min_hits: int, minimum number of hits in a track.
        :param regressor: scikit-learn RANSACRegressor class object.
        :return:
        """

        self.n_tracks = n_tracks
        self.min_hits = min_hits
        self.regressor = regressor

        self.labels_ = None
        self.tracks_params_ = None

    def fit(self, x, y, sample_weight=None):
        """
        Search for the tracks.

        :param x: list of floats, x-coordinates of the hits.
        :param y: ist of floats, y-coordinates of the hits.
        :param sample_weight: sample_weight: list of floats, weights of the hits.
        """

        labels = -1 * numpy.ones(len(x))
        tracks_params = []
        indices = numpy.arange(len(x))

        for track_id in range(self.n_tracks):
            x_track = x[labels == -1]
            y_track = y[labels == -1]
            indices_track = indices[labels == -1]

            if len(x_track) < self.min_hits or len(x_track) <= 0:
                break

            flag = 0
            # sklearn model sometimes can find zero inliers and exception occurs
            while flag != 1 and flag > -100:
                try:
                    regressor = clone(self.regressor)
                    regressor.fit(x_track.reshape(-1, 1), y_track)
                    flag = 1
                except:
                    flag += -1
            assert flag == 1, "RANSAC is failed due to it is not fitted"
            inlier_mask = regressor.inlier_mask_
            estimator = regressor.estimator_

            if numpy.sum(inlier_mask) >= self.min_hits:
                labels[indices_track[inlier_mask]] = track_id
                tracks_params.append([estimator.coef_[0], estimator.intercept_])

        self.tracks_params_ = numpy.array(tracks_params)
        self.labels_ = labels
