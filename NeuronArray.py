import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import ks_2samp

# Sklearn
from sklearn.cross_validation import cross_val_score, StratifiedShuffleSplit, StratifiedKFold
from fittingFun import VonMises
from HelperFun import *


class NeuronArray:
    def __init__(self, data, condition, color, n_locations=1, location=0):
        self.condition = condition
        self.col = color


        # Time axis
        self.edges = np.ravel(data['edges'])
        self.n_time = self.edges.shape[0]

        # Orientations
        self.angles = np.ravel(data['makeStim']['ort'][0, 0])
        self.n_ort = self.angles.shape[0]

        # Probe Location
        assert location in range(n_locations)
        assert n_locations == (data[condition].shape[1] - 1) / self.n_ort
        self.n_loc = n_locations
        self.loc = location

        # data
        self.X, self.Y = build_static(data, condition, np.arange(self.n_time), location=location, n_locations=n_locations)
        self.X_no, _ = build_static(data, condition, np.arange(self.n_time), location=location, n_locations=n_locations, noProbe=True)

        self.n_trial, self.n_cell, _ = self.X.shape
        self.p_val = self.ks_test()
        self.good_cells = np.arange(self.n_cell)
        self.visual_latency = np.empty((self.n_cell,))

        self.decoding_tc = np.zeros((self.n_time,))
        self.decoding_tc_err = np.zeros((self.n_time,))

        self.baseline = np.zeros((self.n_cell,))

        self.pref_ort = np.zeros((self.n_cell), 'int16')
        self.null_ort = np.zeros((self.n_cell), 'int16')

    def ks_test(self):
        P_val = np.ones((self.n_cell, self.n_time))

        for t in range(self.n_time):
            for i in range(self.n_cell):
                pref_or = find_peak(self.X[:, i, t], self.Y)
                D, pval = ks_2samp(self.X_no[:, i, t], self.X[self.Y == pref_or, i, t])
                P_val[i, t] = pval

        return P_val

    def cell_selection(self, alpha):
        good_cells = np.zeros((self.n_cell,))
        for t in np.where(self.edges > 0)[0]:
            for i in range(self.n_cell):
                if not good_cells[i,]:
                    if self.p_val[i, t] < alpha:
                        good_cells[i,] = 1
                        self.visual_latency[i,] = self.edges[t]

        self.good_cells = np.nonzero(good_cells)[0]
        self.visual_latency = self.visual_latency[np.nonzero(good_cells)[0]]
        self.X = self.X[:, self.good_cells, :]
        self.n_trial, self.n_cell, _ = self.X.shape

    @staticmethod
    def equalize_trials(neuron_array_list):
        # find the minimum number of trials
        min_trials = np.inf
        for na in neuron_array_list:
            if na.n_trial < min_trials:
                min_trials = na.n_trial

        # equalize the trials for each condition
        for na in neuron_array_list:
            if na.n_trial > min_trials+4:
                # StratifiedShuffleSplit preserve the percentage of sample from each class (orientation)
                sss = StratifiedShuffleSplit(na.Y, n_iter=1, train_size=min_trials, test_size=None)
                for train_idx, test_idx in sss:
                    na.X = na.X[train_idx, ...]
                    na.Y = na.Y[train_idx]
                na.n_trial = min_trials

    # def optimize_tau(self, learner):

    def smooth(self, tau=0.1):
        """
        Exponential causal filter, for 'retino' condition the firing rate is not smooted across the saccade
        :param tau: time constant, scalar
        :return: filtered X, where each entry is a weighted sum of all the previous entries, with weights exp(-(dt/tau))
        """
        R = np.zeros(self.X.shape)
        if 'retino' in self.condition:
            zero_idx = np.argmin(np.abs(self.edges))

            for t in range(zero_idx):
                R[:, :, t] = np.sum(self.X[:, :, :t + 1] * np.exp(-(self.edges[t] - self.edges[:t + 1]) / tau), axis=2)

            for t in range(zero_idx, self.n_time):
                R[:, :, t] = np.sum(self.X[:, :, zero_idx:t + 1] * np.exp(-(self.edges[t] - self.edges[zero_idx:t + 1]) / tau), axis=2)

        else:
            for t in range(self.n_time):
                R[:, :, t] = np.sum(self.X[:, :, :t + 1] * np.exp(-(self.edges[t] - self.edges[:t + 1]) / tau), axis=2)

        self.X = R

    def jumble(self):
        """ mix the trials to remove the correlation structure between cells
        """
        for unique in np.unique(self.Y):
            for cell in range(self.n_cell):
                self.X[self.Y == unique, cell, :] = np.random.permutation(self.X[self.Y == unique, cell, :])

    def decoding(self, learner, scorer, n_folds=5):
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

        if n_folds == 'max':
            n_folds = max_folds(self.X, self.Y)

        print('n_folds = ', n_folds)

        # initialize cross validation iterator
        k_folds = StratifiedKFold(self.Y, n_folds, shuffle=True)

        # find the time point decoding accuracy
        for t in range(self.n_time):
            cv_accuracy = cross_val_score(learner, self.X[:, :, t], self.Y, scoring=scorer, cv=k_folds, n_jobs=-1)
            self.decoding_tc[t] = cv_accuracy.mean()
            self.decoding_tc_err[t] = cv_accuracy.std(ddof=1) / np.sqrt(n_folds)
            print('on my way, time point %i of %i' % (t + 1, self.n_time))

    def normalize(self, method='pink'):
        assert method in ['pink', 'sub']

        if method == 'pink':
            self.X = (self.X - self.baseline[np.newaxis, :, np.newaxis]) / self.baseline[np.newaxis, :, np.newaxis]

        elif method == 'sub':
            self.X = (self.X - self.baseline[np.newaxis, :, np.newaxis])

    def get_theta(self):
        """ Transform orientation [0, 4] to rad angles
        :param y: orientation vector
        :return: radians angles vector
        """
        theta = np.zeros(self.Y.shape)
        for i, a in enumerate(self.Y):
            theta[i,] = np.radians(self.angles[int(a),]) % np.pi
        return theta

    def set_baseline(self, baseline_time=-0.150):
        baseline_mask = (self.edges < baseline_time)
        self.baseline = self.X[:, :, baseline_mask].mean(axis=(0, 2))

    def set_pref_ort(self):
        self.pref_ort = np.zeros((self.n_cell), 'int16')
        self.null_ort = np.zeros((self.n_cell), 'int16')
        # find the prefered orientation for each cell at visual latency
        theta = self.get_theta()
        for i in range(self.n_cell):
            vis_lat_idx = np.argmin(np.abs(self.edges - self.visual_latency[i]))
            r = self.X[:, i, vis_lat_idx]
            params = VonMises.fit(r, theta)
            self.pref_ort[i] = find_closest(np.unique(theta), params[0])
            self.null_ort[i] = find_null(np.unique(theta), params[0])

    def get_pref_fr(self):
        pref_fr = np.zeros((self.n_time, self.n_cell))
        null_fr = np.zeros((self.n_time, self.n_cell))
        for i in range(self.n_cell):

            pref_fr[:, i] = np.nanmean(self.X[np.where(self.Y == self.pref_ort[i])[0], i, :], axis=0)
            null_fr[:, i] = np.nanmean(self.X[np.where(self.Y == self.null_ort[i])[0], i, :], axis=0)

        return pref_fr, null_fr

    ### Plotting Fun ###
    @staticmethod
    def plot_decoding_time_course(na_list, figpath, file, name):
        # plot the result
        fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)

        for na in na_list:
            l1, = axs.plot(na.edges, na.decoding_tc, label=na.condition, c=na.col, linewidth=.5)
            axs.fill_between(na.edges, na.decoding_tc + na.decoding_tc_err,
                             na.decoding_tc - na.decoding_tc_err, facecolor=na.col,
                             alpha=0.25)

        axs.axhline(y=1. / na.n_ort)
        axs.set_ylabel('Accuracy')
        axs.set_xlabel('Time')
        axs.set_xticks(na.edges[np.arange(na.n_time, step=4)])
        axs.legend(loc='lower left')
        fig.suptitle('%s Decoding time course %s' % (file, name))
        plt.grid(True)

        if not os.path.exists('%sdecoding/' % figpath):
            os.makedirs('%sdecoding/' % figpath)
        print('%sdecoding/%s_%s' % (figpath, file, name))
        plt.savefig('%sdecoding/%s_%s' % (figpath, file, name))
        plt.close(fig)

    @staticmethod
    def plot_firing_rate(na_list, figpath, file, normal, null_orientation=True, savemat=False):
        """# find the average and std err firing rate for prefered and null orientation for all times"""

        if null_orientation: n_plot = 2
        else: n_plot = 1

        fig, axs = plt.subplots(n_plot, 1, sharex=True)

        if savemat:
            dict = {}

        for k, na in enumerate(na_list):
            na.set_baseline()

            na.set_pref_ort()

            if normal in ['pink', 'sub']:
                na.normalize(normal)

            pref_fr, null_fr = na.get_pref_fr()

            mean_pref_fr = pref_fr.mean(axis=1)
            std_pref_fr = pref_fr.std(axis=1, ddof=1) / np.sqrt(na.n_cell)
            mean_null_fr = null_fr.mean(axis=1)
            std_null_fr = null_fr.std(axis=1, ddof=1) / np.sqrt(na.n_cell)

            if savemat:
                dict[na.condition] = {'Prefered_Orientation': pref_fr, 'Null_Orientation': null_fr}

            # plot the results
            y_max = np.max(mean_pref_fr + std_pref_fr)
            y_min = np.min(mean_pref_fr - std_pref_fr)

            axs[0].plot(na.edges, mean_pref_fr, label=na.condition, c=na.col)
            axs[0].fill_between(na.edges, mean_pref_fr - std_pref_fr, mean_pref_fr + std_pref_fr, alpha=0.25, facecolor=na.col)
            axs[0].set_ylim((y_min, y_max))
            axs[0].set_title('Preferred Orientation')
            axs[0].grid(True)
            axs[0].set_xlabel('Time')
            axs[0].set_xticks(na.edges[np.arange(na.n_time, step=4)])
            axs[0].legend(loc='upper left')

            if normal == 'pink':
                axs[0].set_ylabel('Firing Rate % Increase r. Base')
                axs[0].axhline(y=0)
            elif normal == 'sub':
                axs[0].set_ylabel('Firing Rate Spike Increase r. Base')
                axs[0].axhline(y=0)
            else:
                axs[0].set_ylabel('Firing Rate')

            if null_orientation:
                axs[1].plot(na.edges, mean_null_fr, label=na.condition, c=na.col)
                axs[1].fill_between(na.edges, mean_null_fr - std_null_fr, mean_null_fr + std_null_fr, alpha=0.25, facecolor=na.col)
                axs[1].set_ylim((y_min, y_max))
                axs[1].set_title('Null Orientation')
                axs[1].grid(True)
                axs[1].set_xticks(na.edges[np.arange(na.n_time, step=4)])

                if normal == 'pink':
                    # axs[1].set_ylabel('Firing Rate \% Increase r. Base')
                    axs[1].axhline(y=0)
                elif normal == 'sub':
                    # axs[1].set_ylabel('Firing Rate Spike Increase r. Base')
                    axs[1].axhline(y=0)
                # else:
                    # axs[1].set_ylabel('Firing Rate')

        if not os.path.exists('%sfiring_rate/' % figpath):
            os.makedirs('%sfiring_rate/' % figpath)

        plt.savefig('%sfiring_rate/%s%s_pop_FR' % (figpath, file, normal))
        plt.close(fig)

        if savemat:
            return dict

    def plot_tuning_curves(self, figpath, file):
        """
        Plot the tuning curve for all the good cells, for all conditions (each cell is a separate subplot, with every
        conditions). Plot comprise of the average firing rate for each orientation (with std error whiskers) with a von
        mises fit overlay

        """

        vonmises_params = np.zeros((4, self.n_cell))

        # initialize the figure
        size = min_square(self.n_cell)
        fig, axs = plt.subplots(size, size, sharex=True, sharey=False)

        m = 0
        n = 0

        for j, cell in enumerate(self.good_cells):
            # for k in range(n_conditions):
                # find best fit
                r = self.X[:, j, self.visual_latency[j]]
                theta = self.get_theta()
                vonmises_params[:, j] = VonMises.fit(r, theta)

                # plot best fit + data
                axs[m, n].plot(theta, VonMises.vonmises(theta, vonmises_params[:, j]), label='Von Mise Fit', c=self.col)
                plot_wiskers(theta, r, axs[m, n], label='Empirical', color=self.col)

                axs[m, n].set_title('Cell #%i, t=%f.2' % (cell, self.edges[self.visual_latency[j]]))

                m += 1
                if m % size == 0:
                    n += 1
                    m = 0
                if n % size == 0:
                    n = 0

        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower right')
        if not os.path.exists('%stuning_curve/' % figpath):
            os.makedirs('%stuning_curve/' % figpath)
        fig.savefig('%stuning_curve/%s_%s_tuningCurve' % (figpath, file, self.condition),
                    dpi=300)
        plt.close(fig)

        return vonmises_params


