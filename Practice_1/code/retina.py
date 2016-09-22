import numpy
from scipy.optimize import minimize

__author__ = 'mikhail91'


class Retina2DTracker(object):
    def __init__(self, n_tracks, residuals_threshold, min_hits):
        """
        The base class, retina idea for the tracks reconstruction.

        :param n_tracks: int, number of searched tracks.
        :param residuals_threshold: float, residual threshold value for a hit to be considered as a track's hit.
        :param min_hits: int, min number of hits in a track.
        """
        self.n_tracks = n_tracks
        self.min_hits = min_hits
        self.residuals_threshold = residuals_threshold

        self.labels_ = None
        self.tracks_params_ = None

    def retina_func(self, track_prams, x, y, sigma, sample_weight=None):
        """
        Compute retina response.

        :param track_prams: list of floats [k, b], track parameters for the line: y = kx + b.
        :param x: list of floats, x-coordinates of the hits.
        :param y: list of floats, y-coordinates of the hits.
        :param sample_weight: list of floats, weights for each hit.
        :param sigma: float, sigma values.

        :return: -R
        """
        residual = track_prams[0] * x + track_prams[1] - y
        if sample_weight is None:
            retina = numpy.exp(- (residual / sigma) ** 2)
        else:
            retina = numpy.exp(- (residual / sigma) ** 2) * sample_weight

        return -retina.sum()

    def retina_grad(self, track_prams, x, y, sigma, sample_weight=None):
        """
        Retina grad.
        :param track_prams: list of floats [k, b], track parameters for the line: y = kx + b.
        :param x: list of floats, x-coordinates of the hits.
        :param y: list of floats, y-coordinates of the hits.
        :param sigma: float, sigma values.

        :return: -R_grad
        """
        residual = track_prams[0] * x + track_prams[1] - y
        if sample_weight is None:
            retina = numpy.exp(- (residual / sigma) ** 2)
        else:
            retina = numpy.exp(- (residual / sigma) ** 2) * sample_weight
        grad_k = - 2. * residual / sigma ** 2 * retina * x
        grad_b = - 2. * residual / sigma ** 2 * retina

        return -numpy.array([grad_k.sum(), grad_b.sum()])

    def fit_one_track(self, x, y, sample_weight=None):
        pass

    def fit(self, x, y, sample_weight=None):
        """
        Search for all tracks.

        :param x: list of floats, x-coordinates of the hits.
        :param y: list of floats, y-coordinates of the hits.
        :param sample_weight: list of floats, weights of the hits.
        """

        labels = -1 * numpy.ones(len(x))
        tracks_params = []
        used = numpy.zeros(len(x), dtype=bool)

        for track_id in range(self.n_tracks):
            x_track = x[labels == -1]
            y_track = y[labels == -1]

            if sample_weight is None:
                sample_weight_track = None
            else:
                sample_weight_track = sample_weight[labels == -1]

            if len(numpy.unique(x_track)) < self.min_hits or len(x_track) <= 0:
                break

            one_track_params = self.fit_one_track(x_track, y_track, sample_weight_track)
            tracks_params.append(one_track_params)

            distance = numpy.abs(one_track_params[0] * x + one_track_params[1] - y)

            if numpy.sum((distance <= self.residuals_threshold) & (used == False)) < self.min_hits:
                used[distance <= self.residuals_threshold] = True
                continue

            labels[(distance <= self.residuals_threshold) & (used == False)] = track_id
            used[distance <= self.residuals_threshold] = True

        self.labels_ = labels
        self.tracks_params_ = numpy.array(tracks_params)


class Retina2DTrackerOne(Retina2DTracker):
    def __init__(self, n_tracks, residuals_threshold, sigma_range, sigma_decay_rate, min_hits):
        """
        This class is realization of the retina idea for the tracks reconstruction.

        :param n_tracks: int, number of searched tracks.
        :param residuals_threshold: float, residual threshold value for a hit to be considered as a track's hit.
        :param sigma_range: list of floats [min, max], min and max sigma values.
        :param sigma_decay_rate: float, sigma value will be multiplied by this value
        on each iteration of the optimization.
        :param min_hits: int, min number of hits in a track.
        """
        Retina2DTracker.__init__(self,
                                 n_tracks=n_tracks,
                                 residuals_threshold=residuals_threshold,
                                 min_hits=min_hits)

        self.sigma_range = sigma_range
        self.sigma_decay_rate = sigma_decay_rate

    def fit_one_track(self, x, y, sample_weight=None):
        """
        Search for one track.

        :param x: list of floats, x-coordinates of the hits.
        :param y: list of floats, y-coordinates of the hits.
        :param sample_weight: list of floats, weights for each hit.

        :return: list of track params
        """
        sigma_min = self.sigma_range[0]
        sigma_max = self.sigma_range[1]

        sigma = sigma_max
        params = numpy.array([0, y[-1]])

        while sigma >= sigma_min:
            result = minimize(self.retina_func, params, args=(x, y, sigma, sample_weight),
                              method='BFGS', jac=self.retina_grad, options={'gtol': 1e-6, 'disp': False})
            sigma *= self.sigma_decay_rate
            params = result.x

        return result.x


class Retina2DTrackerTwo(Retina2DTracker):
    def __init__(self, n_tracks, residuals_threshold, sigma, min_hits):
        """
        This class is realization of the retina idea for the tracks reconstruction.

        :param n_tracks: int, number of tracks searching for.
        :param residuals_threshold: float, residual threshold value for a hit to be considered as a track's hit.
        :param sigma: float, sigma value for the retina function on each iteration of the optimization.
        :param min_hits: int, min number of hits in a track.
        """
        Retina2DTracker.__init__(self,
                                 n_tracks=n_tracks,
                                 residuals_threshold=residuals_threshold,
                                 min_hits=min_hits)

        self.sigma = sigma

    def fit_one_track(self, x, y, sample_weight=None):
        """
        Search for one track.

        :param x: list of floats, x-coordinates of the hits.
        :param y: list of floats, y-coordinates of the hits.
        :param sample_weight: list of floats, weights for each hit.

        :return: list of track params
        """

        sigma = self.sigma

        rs = []
        ks = []
        bs = []

        for i in range(len(x)):
            for j in range(len(x)):

                if x[j] <= x[i]:
                    continue

                x1 = x[i]
                x2 = x[j]
                y1 = y[i]
                y2 = y[j]

                k0 = (y2 - y1) / (x2 - x1)
                b0 = y1 - k0 * x1

                r = -self.retina_func([k0, b0], x, y, sigma, sample_weight)

                rs.append(r)
                ks.append(k0)
                bs.append(b0)

        rs = numpy.array(rs)
        ks = numpy.array(ks)
        bs = numpy.array(bs)

        index = numpy.argmax(rs)
        params = [ks[index], bs[index]]

        result = minimize(self.retina_func, params, args=(x, y, sigma, sample_weight), method='BFGS',
                          jac=self.retina_grad,
                          options={'gtol': 1e-6, 'disp': False})
        return result.x
