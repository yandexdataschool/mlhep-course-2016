__author__ = 'mikhail91'

import numpy


class HitsMatchingEfficiency(object):
    def __init__(self, eff_threshold=0.5, n_tracks=None):
        """
        This class calculates tracks efficiencies, reconstruction efficiency, ghost rate and clone rate for one event using hits matching.
        :param eff_threshold: float, threshold value of a track efficiency to consider a track reconstructed.
        :return:
        """

        self.eff_threshold = eff_threshold
        self.n_tracks = n_tracks

    def fit(self, true_labels, labels):
        """
        The method calculates all metrics.
        :param true_labels: numpy.array, true labels of the hits.
        :param labels: numpy.array, recognized labels of the hits.
        :return:
        """

        unique_labels = numpy.unique(labels)

        # Calculate efficiencies
        efficiencies = []
        tracks_id = []

        for lab in unique_labels:

            if lab != -1:
                track = true_labels[labels == lab]
                # if len(track[track != -1]) == 0:
                #    continue
                unique, counts = numpy.unique(track, return_counts=True)

                eff = 1. * counts.max() / len(track)
                efficiencies.append(eff)

                tracks_id.append(unique[counts == counts.max()][0])

        tracks_id = numpy.array(tracks_id)
        efficiencies = numpy.array(efficiencies)
        self.efficiencies_ = efficiencies

        # Calculate avg. efficiency
        avg_efficiency = efficiencies.mean()
        self.avg_efficiency_ = avg_efficiency

        # Calculate reconstruction efficiency
        true_tracks_id = numpy.unique(true_labels)

        if self.n_tracks == None:
            n_tracks = (true_tracks_id != -1).sum()
        else:
            n_tracks = self.n_tracks

        reco_tracks_id = tracks_id[efficiencies >= self.eff_threshold]
        unique, counts = numpy.unique(reco_tracks_id[reco_tracks_id != -1], return_counts=True)

        if n_tracks != 0:
            recognition_efficiency = 1. * len(unique) / (n_tracks)
        else:
            recognition_efficiency = 0
        self.recognition_efficiency_ = recognition_efficiency

        # Calculate ghost rate
        if n_tracks != 0:
            ghost_rate = 1. * (len(tracks_id) - len(reco_tracks_id[reco_tracks_id != -1])) / (n_tracks)
        else:
            ghost_rate = 0
        self.ghost_rate_ = ghost_rate

        # Calculate clone rate
        reco_tracks_id = tracks_id[efficiencies >= self.eff_threshold]
        unique, counts = numpy.unique(reco_tracks_id[reco_tracks_id != -1], return_counts=True)

        if n_tracks != 0:
            clone_rate = (counts - numpy.ones(len(counts))).sum() / (n_tracks)
        else:
            clone_rate = 0
        self.clone_rate_ = clone_rate


class TracksReconstractionMetrics(object):
    def __init__(self, eff_threshold, n_tracks=None):
        """
        This class calculates tracks efficiencies, reconstruction efficiency, ghost rate and clone rate for one event using hits matching.
        :param eff_threshold: float, threshold value of a track efficiency to consider a track reconstructed.
        :return:
        """

        self.eff_threshold = eff_threshold
        self.n_tracks = n_tracks

    def fit(self, labels, event):
        """
        The method calculates all metrics.
        :param labels: numpy.array, recognized labels of the hits.
        :param event: pandas.DataFrame, active straw tubes with Label and IsStereo columns.
        :return:
        """

        true_labels = event.Label.values
        is_stereo = event.IsStereo.values

        # Y-views
        labels_y = labels[is_stereo == 0]
        true_labels_y = true_labels[is_stereo == 0]

        hme = HitsMatchingEfficiency(self.eff_threshold, self.n_tracks)
        hme.fit(true_labels_y, labels_y)

        self.efficiencies_y_ = hme.efficiencies_
        self.avg_efficiency_y_ = hme.avg_efficiency_
        self.recognition_efficiency_y_ = hme.recognition_efficiency_
        self.ghost_rate_y_ = hme.ghost_rate_
        self.clone_rate_y_ = hme.clone_rate_



        # Stereo-views
        labels_stereo = labels[is_stereo == 1]
        true_labels_stereo = true_labels[is_stereo == 1]

        hme = HitsMatchingEfficiency(self.eff_threshold, self.n_tracks)
        hme.fit(true_labels_stereo, labels_stereo)

        self.efficiencies_stereo_ = hme.efficiencies_
        self.avg_efficiency_stereo_ = hme.avg_efficiency_
        self.recognition_efficiency_stereo_ = hme.recognition_efficiency_
        self.ghost_rate_stereo_ = hme.ghost_rate_
        self.clone_rate_stereo_ = hme.clone_rate_



        # All-views
        hme = HitsMatchingEfficiency(self.eff_threshold, self.n_tracks)
        hme.fit(true_labels, labels)

        self.efficiencies_ = hme.efficiencies_
        self.avg_efficiency_ = hme.avg_efficiency_
        self.recognition_efficiency_ = hme.recognition_efficiency_
        self.ghost_rate_ = hme.ghost_rate_
        self.clone_rate_ = hme.clone_rate_


class CombinatorQuality(object):
    def __init__(self):

        pass

    def fit(self, labels_before, labels_after, tracks_combinations, charges, inv_momentums, event_before, event_after):

        true_labels_before = event_before.Label.values
        true_labels_after = event_after.Label.values

        check_tracks_combinations = []
        labels = []
        true_pdg_codes = []
        true_charges = []
        check_charges = []
        true_inv_momentums = []
        momentums_err = []

        for track_id, one_tracks_combination in enumerate(tracks_combinations):

            track_id_before = one_tracks_combination[0]
            track_id_after = one_tracks_combination[1]

            unique_before, counts_before = numpy.unique(true_labels_before[labels_before == track_id_before],
                                                        return_counts=True)
            max_fraction_true_label_before = unique_before[counts_before == counts_before.max()][0]

            unique_after, counts_after = numpy.unique(true_labels_after[labels_after == track_id_after],
                                                      return_counts=True)
            max_fraction_true_label_after = unique_after[counts_after == counts_after.max()][0]

            if max_fraction_true_label_before == max_fraction_true_label_after:

                check_tracks_combinations.append(1)
                labels.append(max_fraction_true_label_before)

                pdg_code = event_before[event_before.Label == max_fraction_true_label_before].PdgCode.values[0]
                true_pdg_codes.append(pdg_code)

                if pdg_code == 13 or pdg_code == -211:
                    true_charge = -1.
                elif pdg_code == -13 or pdg_code == 211:
                    true_charge = 1.
                else:
                    true_charge = -999.

                true_charges.append(true_charge)

                if true_charge == charges[track_id]:
                    check_charges.append(1)
                else:
                    check_charges.append(0)

                track_before = event_before[event_before.Label == max_fraction_true_label_before]
                true_inv_momentum_before = (1. / numpy.sqrt(track_before.Px.values ** 2 +
                                                            track_before.Py.values ** 2 +
                                                            track_before.Pz.values ** 2)).mean()

                true_inv_momentum_before = true_inv_momentum_before * true_charge
                true_inv_momentums.append(true_inv_momentum_before)
                momentums_err.append(true_inv_momentum_before / inv_momentums[track_id] - 1)

            else:

                check_tracks_combinations.append(0)
                labels.append(-1)
                true_pdg_codes.append(numpy.nan)
                true_charges.append(numpy.nan)
                check_charges.append(numpy.nan)
                true_inv_momentums.append(numpy.nan)
                momentums_err.append(numpy.nan)


        unique, count = numpy.unique(labels, return_counts=True)
        unique_true, count_true = numpy.unique(event_before.Label.values, return_counts=True)

        self.reco_eff_ = 1. * len(unique[unique != -1]) / len(unique_true[unique_true != -1])

        if len(count[unique == -1]) != 0:
            self.ghost_rate_ = 1. * count[unique == -1][0] / len(unique_true[unique_true != -1])
        else:
            self.ghost_rate_ = 0.

        if len(count[unique != -1]) != 0:
            self.clone_rate_ = 1. * (count[unique != -1].sum() - len(count[unique != -1])) / len(unique_true[unique_true != -1])
        else:
            self.clone_rate_ = 0.

        labels = numpy.array(labels)
        self.n_combined_ = len(labels[labels != -1])

        self.check_tracks_combinations_ = numpy.array(check_tracks_combinations)
        self.labels_ = numpy.array(labels)
        self.true_pdg_codes_ = numpy.array(true_pdg_codes)
        self.true_charges_ = true_charges
        self.check_charges_ = check_charges
        self.true_inv_momentums_ = true_inv_momentums
        self.momentums_err_ = momentums_err