import numpy as np
import matplotlib.pyplot as plt
import os, glob
from scipy.stats import ks_2samp

# Sklearn
from sklearn.cross_validation import cross_val_score, StratifiedShuffleSplit, StratifiedKFold
from fittingFun import VonMises
from HelperFun import *


class NeuronArray:
    def __init__(self, data, condition, n_locations=1, location=0):
        self.condition = condition

        # Time axis
        self.edges = np.ravel(data['edges'])
        self.n_time = self.edges.shape[0]

        # Orientations
        self.angles = np.ravel(data['makeStim']['ort'][0, 0])
        self.n_ort = self.angles.shape[0]

        # Probe Location
        assert n_locations == (data[condition].shape[1] - 1) / self.n_ort
        self.n_loc = n_locations

        self.n_cell = 96
        self.pref_loc = np.full((self.n_cell,), 0, 'int16')


        # Firing Rate: X[n_trial, n_cell, n_times
        # Probe Orientation: Y[n_trial]
        # Probe latency: probe_lat[n_trial]
        if n_locations > 1:
            self.X = []
            self.Y = []
            for i in range(n_locations):
                x, y, _ = build_static(data, condition, np.arange(self.n_time), location=i, n_locations=n_locations)
                self.X.append(x)
                self.Y.append(y)

            self.set_pref_loc()

        else:
            self.X, self.Y, self.probe_lat = build_static(data, condition, np.arange(self.n_time), location=0,
                                          n_locations=1)

        if self.condition is 'postsac_change':
            self.flip_ort()

        # No probe Firing rate Data
        self.X_no, _, _ = build_static(data, condition, np.arange(self.n_time), noProbe=True)

        if self.n_loc == 1:
            self.n_trial, self.n_cell, _ = self.X.shape
        else:
            self.n_trial, self.n_cell, _ = self.X[0].shape

        self.n_booth_trial = self.n_trial

        # self.trial_mask = np.ones(self.n_trial, 'bool')
        self.booth_mask = np.ones(self.n_trial, 'bool')

        self.p_val = np.ones((self.n_cell, self.n_time))
        self.cell_mask = np.ones(self.n_cell, 'bool')
        self.visual_latency = np.full((self.n_cell,), 0.125)

        self.remap_cells_mask = np.zeros(self.n_cell)
        self.remap_latency = np.zeros(self.n_cell)

        self.baseline = np.zeros((self.n_cell,))

        self.pref_ort = np.zeros((self.n_cell,), 'int16')
        self.null_ort = np.zeros((self.n_cell,), 'int16')

        # Smoothing Param
        self.tau = 0

        # Normalization Method
        self.method = 'raw'

        # Analysed data
        self.data_dict = {}

    def ks_test(self):
        for t in range(self.n_time):
            for c in range(self.n_cell):
                fr = self.get_fr(cell=c, times=t)
                ort = self.get_ort(cell=c)
                # print(type(fr), ort.shape)
                pref_or = find_peak(fr, ort)
                D, pval = ks_2samp(self.X_no[:, c, t], fr[ort == pref_or])
                self.p_val[c, t] = pval

    def trial_selection(self, bounds):
        mini, maxi = bounds

        trial_mask = ((self.probe_lat > mini) & (self.probe_lat < maxi))
        self.X, self.Y, self.probe_lat = self.X[trial_mask, ...], self.Y[trial_mask, ], self.probe_lat[trial_mask]
        self.n_trial = trial_mask.sum()
        self.n_booth_trial = self.n_trial
        self.booth_mask = np.ones(self.n_trial, 'bool')

    def cell_selection(self, alpha):
        self.cell_mask[:] = False
        for t in np.where(self.edges > 0)[0]:
            for i in range(self.n_cell):
                if not self.cell_mask[i,]:
                    if self.p_val[i, t] < alpha:
                        self.cell_mask[i,] = True
                        self.visual_latency[i,] = self.edges[t]
        self.n_cell = self.cell_mask.sum()

    # def select_trials(self, n_trial):
    #     trials = np.nonzero(self.trial_mask)[0]
    #     trials = np.random.choice(trials, size=n_trial)
    #     self.trial_mask[:] = False
    #     self.trial_mask[trials] = True
    #     self.n_trial = self.trial_mask.sum()

    # def cell_selection_kosher(self, alpha):
    #
    #     good_cells = np.zeros((self.n_cell,))
    #     visual_lat = - 0.1
    #     idx = np.argmin(np.abs(self.edges - visual_lat))
    #     idx = [idx-1, idx, idx +1]
    #     for t in idx:
    #          for i in range(self.n_cell):
    #              if not good_cells[i,]:
    #                  if self.p_val_kosher[i, t] < alpha:
    #                      good_cells[i,] = 1
    #                      # self.visual_latency[i,] = self.edges[t]
    #
    #     self.good_cells = np.nonzero(good_cells)[0]
    #     self.visual_latency = self.visual_latency[np.nonzero(good_cells)[0]]
    #     self.X = self.X[:, self.good_cells, :]
    #     self.n_trial, self.n_cell, _ = self.X.shape
    #
    # def remap_cell_selection(self, alpha):
    #     self.remap_cell = np.zeros(self.good_cells.shape, dtype=bool)
    #     self.remap_latency = np.zeros(self.good_cells.shape)
    #     idx = np.where(self.edges >0)[0]
    #     for t in idx:
    #         for i in range(self.n_cell):
    #             if not self.remap_cell[i,]:
    #                 if self.remap_pval[i, t] < alpha:
    #                     self.remap_cell[i,] = 1
    #                     self.remap_latency[i,] = self.edges[t]

    #def optimize_tau(self, learner):

    def jumble(self):
        """ mix the trials to remove the correlation structure between cells
        """
        for unique in np.unique(self.Y):
            for cell in range(96):
                self.X[self.Y == unique, cell, :] = np.random.permutation(self.X[self.Y == unique, cell, :])

    def smooth(self, Fr):
        """
        Return smoothed firing rate
        Exponential causal filter, for 'retino' condition the firing rate is not smoothed across the saccade
        :param tau: time constant, scalar
        :return: filtered X, where each entry is a weighted sum of all the previous entries, with weights exp(-(dt/tau))
        """
        R = np.zeros(self.X.shape)
        if 'retino' in self.condition:
            zero_idx = np.argmin(np.abs(self.edges))

            for t in range(zero_idx):
                R[:, :, t] = np.sum(Fr[..., :t + 1] * np.exp(-(self.edges[t] - self.edges[:t + 1]) / self.tau), axis=2)

            for t in range(zero_idx, self.n_time):
                R[:, :, t] = np.sum(Fr[..., zero_idx:t + 1] * np.exp(-(self.edges[t] - self.edges[zero_idx:t + 1]) / self.tau), axis=2)

        else:
            for t in range(self.n_time):
                R[:, :, t] = np.sum(Fr[..., :t + 1] * np.exp(-(self.edges[t] - self.edges[:t + 1]) / self.tau), axis=2)

        return R

    def normalize(self, Fr):
        # Return Normalize Firing rate
        # method:   'pink'  - Percentage Increase from baseline
        #           'sub'   - Baseline Substracted

        if self.method == 'pink':
            fr = (Fr - self.baseline[np.newaxis, :, np.newaxis]) / self.baseline[np.newaxis, :, np.newaxis]

        elif self.method == 'sub':
            fr = (Fr - self.baseline[np.newaxis, :, np.newaxis])

        else: fr = self.X

        return fr

    def flip_ort(self):
        self.Y = (self.Y + self.n_ort/2) % self.n_ort

    def set_data_dict(self, dict):
        self.data_dict = dict

    def exist_data(self, task):
        if task not in self.data_dict.keys():
            # print(type(self.data_dict))
            return False
        else:
            return True

    def set_cell(self, cells):
        self.cell_mask[:] = False
        self.cell_mask[cells] = True
        self.n_cell = self.cell_mask.sum()

    def set_tau(self, tau):
        self.tau = tau

    def set_method(self, method):
        self.method = method

    def set_baseline(self, baseline_time=-0.150):
        baseline_mask = (self.edges < baseline_time)
        # print(self.X[self.trial_mask, ...][..., baseline_mask].shape)
        self.baseline = self.X[..., baseline_mask].mean(axis=(0, 2))

    def set_pref_ort(self, t=None):
        # find the prefered orientation for each cell at visual latency
        for cell in np.nonzero(self.cell_mask)[0]:
            if t:
                vis_lat_idx = int(np.argmin(np.abs(self.edges - t)))
            else:
                # vis_lat_idx = int(np.argmin(np.abs(self.edges - self.visual_latency[cell])))
                vis_lat_idx = int(np.argmin(np.abs(self.edges - 0.125)))

            r = self.get_fr(cell=cell, times=vis_lat_idx)
            theta = self.get_theta(cell=cell)
            params = VonMises.fit(r, theta)
            self.pref_ort[cell] = find_closest(np.unique(theta), params[0])
            self.null_ort[cell] = find_null(np.unique(theta), params[0])

    def set_pref_loc(self):
        vis_lat = 0.125
        vis_lat_idx = int(np.argmin(np.abs(self.edges-vis_lat)))

        for cell in range(self.n_cell):
            loc = -1
            maxav = -np.inf
            for l in range(self.n_loc):
                av = np.mean(self.X[l][:, cell, vis_lat_idx])
                if av > maxav:
                    loc = l
                    maxav = av
            self.pref_loc[cell] = loc

        # y = np.zeros((1,))
        # max_trial_ort = []
        # for i in range(self.n_ort):
        #     max_trial = -np.inf
        #     for j in range(self.n_loc):
        #         temp_trial = (self.Y[j] == i).sum()
        #         if temp_trial > max_trial:
        #             max_trial = temp_trial
        #     y = np.hstack((y, (np.full((max_trial,), i, dtype='int16'))))
        #     max_trial_ort.append(max_trial)
        #
        # y = y[1:]
        # x = np.full((y.shape[0], self.n_cell, self.n_time), np.nan)
        #
        # for k in range(self.n_cell):
        #     pref_loc = -1
        #     avg_fr = -np.inf
        #     for j in range(self.n_loc):
        #         temp_avg_fr = np.mean(self.X[j][:,k,vis_lat_idx])
        #         if temp_avg_fr > avg_fr:
        #             avg_fr = temp_avg_fr
        #             pref_loc = j
        #
        #     tr = 0
        #     for i in range(self.n_ort):
        #         temp_trial = self.X[pref_loc][self.Y[pref_loc]==i, k, :].shape[0]
        #         x[tr:tr+max_trial_ort[i], k, :] = self.X[pref_loc][self.Y[pref_loc]==i, k, :][np.random.choice(temp_trial, max_trial_ort[i], replace=True), ...]
        #         tr += max_trial_ort[i]
        #
        # self.X, self.Y = x, y

        # print(self.X[:, 2, vis_lat_idx])

    def set_booth_trial(self):
        trials = np.nonzero(self.trial_mask)[0]
        trials = np.random.choice(trials, size=self.n_booth_trial, replace=False)
        self.booth_mask[:] = False
        self.booth_mask[trials] = True

    def get_fr(self, booth=False, smooth=False, normal=False, times=None, cell=None):
        if self.n_loc == 1:
            fr = self.X
            if cell is None:
                cell = self.cell_mask
        else:
            if cell is None:
                cell = self.cell_mask
                fr = self.X[self.pref_loc[0]]
            else:
                fr = self.X[self.pref_loc[cell]]

        if times is None:
            times = np.arange(self.n_time)

        if smooth:
            fr = self.smooth(fr)

        if normal:
            fr = self.normalize(fr)

        if booth:
            fr = fr[self.booth_mask, ...]

        # print(cell, times)
        fr = fr[:, cell, times]

        return fr

    def get_pref_fr(self):
        fr = self.get_fr(normal=True)
        ort = self.get_ort()

        pref_fr = np.zeros((self.n_time, self.n_cell))
        null_fr = np.zeros((self.n_time, self.n_cell))
        for i in range(self.n_cell):
            pref_fr[:, i] = np.nanmean(fr[np.where(ort == self.pref_ort[i])[0], i, :], axis=0)
            null_fr[:, i] = np.nanmean(fr[np.where(ort == self.null_ort[i])[0], i, :], axis=0)

        return pref_fr, null_fr

    def get_ort(self, booth=False, cell=None):
        if self.n_loc == 1:
            y = self.Y
        else:
            y = self.Y[self.pref_loc[cell]]

        if booth:
            ort = y[self.booth_mask]
        else:
            ort = y

        return ort

    def get_pref_ort(self, t):
        self.set_pref_ort(t)
        return self.pref_ort

    def get_good_cells(self):
        return np.nonzero(self.cell_mask)

    def firing_rate_data(self, normal):
        self.set_baseline()

        self.set_pref_ort()

        self.set_method(normal)

        pref_fr, null_fr = self.get_pref_fr()

        self.data_dict['fr%s' % normal] = {'pref_fr': pref_fr, 'null_fr': null_fr}

    def decoding(self, learner, scorer, smooth, name, n_folds=5):
        """
        plots the time point by time point decoding accuracy time course for every condition in conditions
        :param good_cells: list of index corresponding to good cells.
        :param learner: scikit learn classifier.
        :param name: string, identifier to appears in the title and filename.
        :param smooth: string, smoothing algorithm, must be 'ES' for exponential smoothing (recursive) or 'causal' for a
                        simple exponential filter. Leave None for no smoothing.
        :param tau: Scalar, float, smoothing time constant,
        :param n_folds: Scalar, int, number of cross validation folds or 'max' for automatic setting
        :param location: Scalar, int, must be in range of n_locations
        :param jumbled: Boolean, if true, removes the correlation structure
        :param equal_trials: Boolean, if true, equalize the number of trials between conditions
        :return: decoding time course, numpy array (n_conditions, n_times), decoding accuracy for each condition and time point
                decoding time course error, numpy array (n_conditions, n_times)
        """

        fr = self.get_fr(smooth=smooth, booth=True)

        ort = self.get_ort()

        print('n_trials', fr.shape[0])

        if n_folds == 'max':
            n_folds = max_folds(fr, ort)

        print('n_folds = ', n_folds)

        # initialize cross validation iterator
        k_folds = StratifiedKFold(ort, n_folds, shuffle=True)

        # find the time point decoding accuracy

        decoding_tc = np.zeros((self.n_time,))
        decoding_tc_err = np.zeros((self.n_time,))

        for t in range(self.n_time):
            cv_accuracy = cross_val_score(learner, fr[:, :, t], ort, scoring=scorer, cv=k_folds, n_jobs=-1)
            decoding_tc[t] = cv_accuracy.mean()
            decoding_tc_err[t] = cv_accuracy.std(ddof=1) / np.sqrt(n_folds)
            print('on my way, time point %i of %i' % (t + 1, self.n_time))

        self.data_dict['decode'] = {'decoding_tc':decoding_tc, 'decoding_tc_err':decoding_tc_err, 'info': name}

    def decoding_booth_estimate(self, learner, scorer, smooth, name, n_folds=5, n_est=5):
        """
        plots the time point by time point decoding accuracy time course,
        :param good_cells: list of index corresponding to good cells.
        :param learner: scikit learn classifier.
        :param name: string, identifier to appears in the title and filename.
        :param smooth: string, smoothing algorithm, must be 'ES' for exponential smoothing (recursive) or 'causal' for a
                        simple exponential filter. Leave None for no smoothing.
        :param tau: Scalar, float, smoothing time constant,
        :param n_folds: Scalar, int, number of cross validation folds or 'max' for automatic setting
        :param location: Scalar, int, must be in range of n_locations
        :param jumbled: Boolean, if true, removes the correlation structure
        :param equal_trials: Boolean, if true, equalize the number of trials between conditions
        :return: decoding time course, numpy array (n_conditions, n_times), decoding accuracy for each condition and time point
                decoding time course error, numpy array (n_conditions, n_times)
        """
        decoding_tc_est = np.zeros((self.n_time, n_est))
        decoding_err_est = np.zeros((self.n_time, n_est))

        for i in range(n_est):
            self.set_booth_trial()
            fr = self.get_fr(smooth=smooth, booth=True)
            ort = self.get_ort(booth=True)

            print('n_trials', fr.shape[0])

            if n_folds == 'max':
                n_folds = max_folds(fr, ort)

            print('n_folds = ', n_folds)

            # initialize cross validation iterator
            k_folds = StratifiedKFold(ort, n_folds, shuffle=True)

            # find the time point decoding accuracy
            for t in range(self.n_time):
                cv_accuracy = cross_val_score(learner, fr[:, :, t], ort, scoring=scorer, cv=k_folds, n_jobs=-1)
                decoding_tc_est[t, i] = cv_accuracy.mean()
                decoding_err_est[t, i] = cv_accuracy.std(ddof=1) / np.sqrt(n_folds)

            print('on my way, time point %i of %i' % (i+ 1, n_est))

        self.data_dict['decode'] = {'decoding_tc': decoding_tc_est.mean(axis=1), 'decoding_tc_err': decoding_err_est.mean(axis=1),
                                                    'info': '%s_booth%i' % (name, n_est)}

    def get_theta(self, offset=None, cell=0):
        """ Transform orientation [0, 4] to rad angles
        :param y: orientation vector
        :return: radians angles vector
        """
        if self.n_loc == 1:
            y = self.Y

        else:
            y = self.Y[self.pref_loc[cell]]

        theta = np.zeros(y.shape)

        if offset:
            ort = (y + offset) % self.n_ort
        else:
            ort = y

        for i, a in enumerate(ort):
            theta[i, ] = np.radians(self.angles[int(a)]) % np.pi
        return theta

    def get_orientation_bias(self):

        i = np.complex(0, 1)

        ob = np.sum(self.X * np.exp(i * 2 * self.get_theta()[:, np.newaxis, np.newaxis]), axis=0) / np.sum(self.X, axis=0)

        return np.abs(ob)

    def tuning(self, time):
        """
        Find tuning curve for all the good cells with a von mises fit
        """
        t = np.argmin(abs(self.edges - time))

        vonmises_params = np.zeros((4, self.n_cell))
        good_cells = np.nonzero(self.cell_mask)[0]

        for j, cell in enumerate(good_cells):
            # for k in range(n_conditions):
            # find best fit
            theta = self.get_theta(cell=cell)
            r = self.get_fr(times=t, cell=cell)
            vonmises_params[:, j] = VonMises.fit(r, theta)

        self.data_dict['tuning'] = {'vonmises_param': vonmises_params, 'fr': self.X, 'ort': self.Y, 'cells': good_cells, 'pref_loc': self.pref_loc}

