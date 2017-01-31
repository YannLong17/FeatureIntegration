import numpy as np
import glob
import matplotlib.pyplot as plt
import os, glob

from NeuronArray import NeuronArray as NA
from HelperFun import *
from fittingFun import VonMises
from sklearn.cross_validation import cross_val_score, StratifiedShuffleSplit, StratifiedKFold


color_list = ['blue', 'black', 'red', 'green']


class RecordingDay:
    def __init__(self, day, conditions):
        path = glob.glob('data/%s*' % day)
        print(path[0])
        path = path[0]
        data = load(path)

        self.day = day
        self.figpath = 'results/Decoding Analysis/%s/' % self.day

        # Time axis
        self.edges = np.ravel(data['edges'])
        self.n_time = self.edges.shape[0]

        # Orientations
        self.angles = np.ravel(data['makeStim']['ort'][0, 0])
        self.n_ort = self.angles.shape[0]

        # Initialize
        n_condition = len(conditions)
        self.NA_list = []
        for i in range(n_condition):
            assert conditions[i] in data.keys()
            self.NA_list.append(NA(data, conditions[i]))

        if 'presac_retino_only' in data.keys():
            self.NA_fix = NA(data, 'presac_retino_only')

        if 'presac_only' in data.keys():
            self.NA_remap = NA(data, 'presac_only')

    def equalize_trials(self):
        # find the minimum number of trials
        min_trials = np.inf
        for na in self.NA_list:
            if na.n_trial < min_trials:
                min_trials = na.n_trial

        # equalize the trials for each condition
        for na in self.NA_list:
            if na.n_trial > min_trials+4:
                # StratifiedShuffleSplit preserve the percentage of sample from each class (orientation)
                sss = StratifiedShuffleSplit(na.Y, n_iter=1, train_size=min_trials, test_size=None)
                for train_idx, test_idx in sss:
                    na.X = na.X[train_idx, ...]
                    na.Y = na.Y[train_idx]
                na.n_trial = min_trials

    def cell_select(self, alpha):
        for na in self.NA_list:
            na.ks_test()
            na.cell_selection(alpha)

    ### Plotting Fun ###
    def plot_decoding_time_course(self, name):
        # plot the result
        fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)

        for k, na in enumerate(self.NA_list):
            l1, = axs.plot(na.edges, na.decoding_tc, label=na.condition, c=color_list[k], linewidth=.5)
            axs.fill_between(na.edges, na.decoding_tc + na.decoding_tc_err,
                                 na.decoding_tc - na.decoding_tc_err, facecolor=color_list[k],
                                 alpha=0.25)

        axs.axhline(y=1. / self.n_ort)
        axs.set_ylabel('Accuracy')
        axs.set_xlabel('Time')
        axs.set_xticks(self.edges[np.arange(na.n_time, step=4)])
        axs.legend(loc='lower left')
        fig.suptitle('%s Decoding time course %s' % (self.day, name))
        plt.grid(True)

        if not os.path.exists('%sdecoding/' % self.figpath):
            os.makedirs('%sdecoding/' % self.figpath)
        filepath = '%sdecoding/%s_%s' % (self.figpath, self.day, name)
        i = 0
        while glob.glob('%s%i.*' % (filepath, i)):
            i += 1

        filepath = '%s%i' % (filepath, i)
        plt.savefig(filepath)
        plt.close(fig)

    def plot_firing_rate(self, normal, name, null_orientation=True, savemat=False):
        """ find the average and std err firing rate for prefered and null orientation for all times"""

        if null_orientation:
            n_plot = 2
        else:
            n_plot = 1

        fig, axs = plt.subplots(n_plot, 1, sharex=True)

        if savemat:
            dict = {}

        for k, na in enumerate(self.NA_list):
            na.set_baseline()

            na.set_pref_ort()

            pref_fr, null_fr = na.get_pref_fr(normal)

            mean_pref_fr = pref_fr.mean(axis=1)
            std_pref_fr = pref_fr.std(axis=1, ddof=1) / np.sqrt(na.n_cell)
            mean_null_fr = null_fr.mean(axis=1)
            std_null_fr = null_fr.std(axis=1, ddof=1) / np.sqrt(na.n_cell)

            if savemat:
                dict[na.condition] = {'Prefered_Orientation': pref_fr, 'Null_Orientation': null_fr}

            # plot the results
            y_max = np.max(mean_pref_fr + std_pref_fr)
            y_min = np.min(mean_pref_fr - std_pref_fr)

            axs[0].plot(na.edges, mean_pref_fr, label=na.condition, c=color_list[k])
            axs[0].fill_between(na.edges, mean_pref_fr - std_pref_fr, mean_pref_fr + std_pref_fr, alpha=0.25,
                                    facecolor=color_list[k])
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
                axs[1].plot(na.edges, mean_null_fr, label=na.condition, c=color_list[k])
                axs[1].fill_between(na.edges, mean_null_fr - std_null_fr, mean_null_fr + std_null_fr, alpha=0.25,
                                        facecolor=color_list[k])
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

        if not os.path.exists('%sfiring_rate/' % self.figpath):
            os.makedirs('%sfiring_rate/' % self.figpath)

        filepath = '%sfiring_rate/%s%s%s_pop_FR' % (self.figpath, self.day, normal, name)
        i = 0
        while glob.glob('%s%i.*' % (filepath, i)):
            i += 1

        filepath = '%s%i' % (filepath, i)
        plt.savefig(filepath)
        plt.close(fig)

        if savemat:
            return dict

    def plot_orientation_bias(self):

        fig, axs = plt.subplots(1, 1, sharey=True)

        for k, na in enumerate(self.NA_list):
            ob = na.get_orientation_bias()
            print(ob.shape)

            axs.plot(self.edges, ob.mean(axis=0), label=na.condition, c=color_list[k])
            axs.fill_between(na.edges, ob.mean(axis=0) - ob.std(axis=0), ob.mean(axis=0) + ob.std(axis=0),
                                alpha=0.25, facecolor=color_list[k])

        axs.set_ylim((0, 1))
        axs.set_title('Orientation Bias')
        axs.grid(True)
        axs.set_xlabel('Time')
        axs.set_xticks(na.edges[np.arange(na.n_time, step=4)])
        axs.legend(loc='upper left')

        plt.show()
        plt.close(fig)




