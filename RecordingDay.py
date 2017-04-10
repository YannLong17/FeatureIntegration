import numpy as np
import glob
import matplotlib.pyplot as plt
import os, glob

from NeuronArray import NeuronArray as NA
from HelperFun import *
from fittingFun import VonMises
from sklearn.cross_validation import cross_val_score, StratifiedShuffleSplit, StratifiedKFold


color_list = ['blue', 'red', 'black', 'green', 'orange', 'magenta']
good_cell_dic = {
                 'p128': [15, 16, 30, 32, 34, 35, 40, 66, 80, 82, 83, 86, 87, 88, 89],
                 'p131': [34, 35, 40, 52, 67, 69, 71, 84, 87, 88, 89],
                 'p132': [15, 34, 40, 88, 89],
                 'p135': [40, 41, 63, 72, 73, 82, 83, 84, 85, 86, 87, 88, 89, 90],
                 'p136': [34, 35, 39, 40, 41, 42, 49, 51, 63, 72, 73, 74, 81, 84, 85, 88, 90],
                 'p137': [38, 40, 41, 83, 82, 88, 87]
                 }

# 'p121': [34, 41, 80]

# n_location_list = {'p091': 2,
#                    'p134': 2,
#                    'p135': 2,
#                    'p136': 2,
#                    'p137': 2,
#                   }

class RecordingDay:
    def __init__(self, day, conditions, alpha, location):
        path = glob.glob('data/%s*' % day)
        print(path[0])
        path = path[0]
        data = load(path)

        self.day = day
        self.figpath = 'results/Decoding Analysis/%s/' % self.day

        # Time axis
        self.edges = np.ravel(data['edges'])
        self.n_time = self.edges.shape[0]

        # Visual Latency
        self.vis_lat = 0.125
        self.vis_lat_idx = int(np.argmin(abs(self.edges - self.vis_lat)))

        # Orientations
        self.angles = np.ravel(data['makeStim']['ort'][0, 0])
        self.n_ort = self.angles.shape[0]

        # Good_cells
        self.good_cells = np.arange(96)
        if day in good_cell_dic.keys():
            self.good_cells = good_cell_dic[day]

        # Locations
        self.n_loc = (data[conditions[0]].shape[1] - 1) // self.n_ort

        # Initialize
        self.conditions = conditions
        n_condition = len(conditions)
        self.NA_list = []

        for i in range(n_condition):
            assert conditions[i] in data.keys()
            self.NA_list.append(NA(data, conditions[i], location))

        bounds = [-0.075, -0.025]
        if 'openloop' in path:
            self.trial_select(bounds)

        self.cell_select(alpha)

        for na in self.NA_list:
            na.set_baseline()

        # if 'presac_retino_only' in data.keys():
        #     self.NA_fix = NA(data, 'presac_retino_only')
        #
        # if 'presac_only' in data.keys():
        #     self.NA_remap = NA(data, 'presac_only')

        # Analysed data File
        self.save = False
        self.unused_data = None

    def open_mat(self):
        if os.path.isfile('%s%s_data.mat' % (self.figpath, self.day)):
            print('old exists ')
            dict = loadmat('%s%s_data.mat' % (self.figpath, self.day))
        else:
            dict = {}

        return dict

    # def check_condition(self, condition):
    #     if condition not in self.data_dict.keys():
    #         return False
    #     else:
    #         return True

    def set_save(self, arg=None):
        if arg in ['load', 'over']:
            self.save = arg
        else:
            self.save = True

        dic = self.open_mat()

        for na in self.NA_list:
            na.set_data_dict(dic.pop(na.condition, {}))

        self.unused_data = dic

    def save_mat(self):
        mydict = {}

        for na in self.NA_list:
            mydict[na.condition] = na.data_dict

        mydict = {**self.unused_data, **mydict}

        sio.savemat('%s%s_data.mat' % (self.figpath, self.day), mdict=mydict)

    def cell_select(self, alpha, intersect=True):
        # find good Cells for each conditions
        # Keep only cells that are common to all condition (intersect)
        for na in self.NA_list:
            na.ks_test()
            na.cell_selection(alpha)
            self.good_cells = np.intersect1d(self.good_cells, np.nonzero(na.cell_mask))

        if intersect:
            for na in self.NA_list:
                na.set_cell(self.good_cells)

    def set_min_trials(self):
        # find the minimum number of trials
        min_trials = np.inf
        for na in self.NA_list:
            if na.n_trial < min_trials:
                min_trials = na.n_trial

        for na in self.NA_list:
            na.n_booth_trial = min_trials

    def set_vis_lat(self, vis_lat):
        self.vis_lat = vis_lat
        self.vis_lat_idx = int(np.argmin(abs(self.edges - self.vis_lat)))

    def trial_select(self, bounds):
        for na in self.NA_list:
            if na.condition is not 'postsac':
                na.trial_selection(bounds)

    def equalize_trials(self):
        # find the minimum number of trials
        # min_trials = np.inf
        # for na in self.NA_list:
        #     if na.n_trial < min_trials:
        #         min_trials = na.n_trial

        # equalize the trials for each condition
        for na in self.NA_list:
            na.set_booth_trial()

        # # StratifiedShuffleSplit preserve the percentage of sample from each class (orientation)
        # sss = StratifiedShuffleSplit(na.Y, n_iter=1, train_size=min_trials, test_size=None)
        # for train_idx, test_idx in sss:
        #     na.X = na.X[train_idx, ...]
        #     na.Y = na.Y[train_idx]
        # na.n_trial = min_trials

    def aligned_ort(self, center, time, arg='indep'):
        """
        For all conditions, align all cell to the the same prefered orientation, according to the first condition prefered orientation.
        :return:
        """
        pref_ort = self.NA_list[0].get_pref_ort(time)
        if arg is 'first':
            pref_ort = self.NA_list[0].get_pref_ort(time)

        elif arg is 'all':
            for na in self.NA_list[1:]:
                na.set_pref_ort(time)
            pref_ort = self.NA_list[0].get_pref_ort(time)
            # print(pref_ort)
            mask = np.ones(len(self.good_cells), dtype=bool)
            for j, cell in enumerate(self.good_cells):
                for na in self.NA_list[1:]:
                    if na.pref_ort[cell] != pref_ort[cell]:
                        mask[j] = 0

            self.good_cells = self.good_cells[mask]

            for na in self.NA_list:
                na.set_cell(self.good_cells)

        # for na in self.NA_list:
        #     if arg is 'indep':
        #         pref_ort = na.get_pref_ort(time)
        #     na.set_cell(self.good_cells)
        #     r = na.get_fr(times=np.argmin(np.abs(self.edges - time)))
        #     theta = np.zeros((na.n_trial, na.n_cell))
        #     for k, cell in enumerate(self.good_cells):
        #         offset = center - pref_ort[cell]
        #         theta[:, k] = na.get_theta(offset)
        #     fr.append(r)
        #     ort.append(theta)

        # offset = center - pref_ort
        #
        # #     r = na.get_fr(times=np.argmin(np.abs(self.edges - time)))
        # #     theta = np.zeros((na.n_trial, na.n_cell))
        # #     for k, cell in enumerate(self.good_cells):
        # #         offset = center - pref_ort[cell]
        # #         theta[:, k] = na.get_theta(offset)
        # #     fr.append(r)
        # #     ort.append(theta)
        # #
        # # return fr, ort
        #
        # return offset

    def set_tau(self, tau):
        for na in self.NA_list:
            na.set_tau(tau)

    def decode(self, learner, scorer, smooth, name, n_folds=5, booth=False):
        if booth:
            for na in self.NA_list:
                na.n_booth_trial = booth
                if na.condition is 'postsac':
                    if na.exist_data('decode%s' % name):
                        if self.save is 'over':
                            na.decoding_booth_estimate(learner, scorer, smooth, name, n_folds=n_folds)
                    else:
                        na.decoding_booth_estimate(learner, scorer, smooth, name, n_folds=n_folds)
                else:
                    if na.exist_data('decode%s' % name):
                        if self.save is 'over':
                            na.decoding(learner, scorer, smooth, name, n_folds=n_folds)
                    else:
                        na.decoding(learner, scorer, smooth, name, n_folds=n_folds)

        else:
            for na in self.NA_list:
                if na.exist_data('decode%s' % name):
                    if self.save is 'over':
                        na.decoding(learner, scorer, smooth, name, n_folds)
                else:
                    na.decoding(learner, scorer, smooth, name, n_folds)

    def neutral_decode(self, learner, scorer, smooth, name, normal):
        # Train Learner on first condition at time
        na = self.NA_list[0]

        ort = na.get_ort()
        fr = na.get_fr(times=self.vis_lat_idx, smooth=smooth, normal=normal)

        print(ort.shape, fr.shape)
        learner.fit(fr, ort)
        print('trained')

        for na in self.NA_list:
            na.decoding(learner, scorer, smooth, name, normal, train=False)


    def firing_rate(self, normal):
        for na in self.NA_list:
            if na.exist_data('fr%s' % normal):
                if self.save is 'over':
                    na.firing_rate_data(normal)
            else:
                na.firing_rate_data(normal)

    def tuning(self):
        time = self.vis_lat
        for na in self.NA_list:
            if na.exist_data('tuning%i' % time):
                if self.save is 'over':
                    na.tuning(time)
            else:
                na.tuning(time)

    ### Plotting Fun ###
    def plot_decoding_time_course(self, name, scorer_name):
        # plot the result
        fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)

        for k, na in enumerate(self.NA_list):
            decoding_tc = na.data_dict['decode']['decoding_tc']
            decoding_tc_err = na.data_dict['decode']['decoding_tc_err']

            l1, = axs.plot(na.edges, decoding_tc, label=na.condition, c=color_list[k], linewidth=2)
            axs.fill_between(na.edges, decoding_tc + decoding_tc_err, decoding_tc - decoding_tc_err,
                             facecolor=color_list[k], alpha=0.25)

        if scorer_name is 'Accuracy':
            axs.axhline(y=1. / self.n_ort)

        axs.set_ylabel(scorer_name)

        axs.set_xlabel('Time')
        axs.set_xticks(self.edges[np.arange(self.n_time, step=4)])
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

    def plot_firing_rate(self, normal, name, null_orientation=True, arg=None, savemat=False):
        """ find the average and std err firing rate for prefered and null orientation for all times"""

        if null_orientation:
            n_plot = 2
        else:
            n_plot = 1

        fig, axs = plt.subplots(n_plot, 1, sharex=True)

        y_min = np.inf
        y_max = -np.inf

        for k, na in enumerate(self.NA_list):
            pref_fr = []
            null_fr = []

            na.set_pref_ort(self.vis_lat)
            pref_ort = na.get_pref_ort()
            null_ort = na.get_null_ort()

            # if normal:
            #     na.set_baseline()

            for j, cell in enumerate(self.good_cells):
                r = na.get_fr(cell=cell, normal=normal)
                ort = na.get_ort(cell=cell)
                pref_fr.append(r[ort == pref_ort[cell]])
                null_fr.append(r[ort == null_ort[cell]])

            if arg is 'ovr':
                pref_r = np.zeros((1, self.n_time))
                null_r = np.zeros((1, self.n_time))

                for j in range(len(pref_fr)):
                    pref_r = np.vstack((pref_r, pref_fr[j]))
                    null_r = np.vstack((null_r, null_fr[j]))

                mean_pref_fr = pref_r.mean(axis=0)
                std_pref_fr = pref_r.std(axis=0, ddof=1) / np.sqrt(na.n_cell)
                mean_null_fr = null_r.mean(axis=0)
                std_null_fr = null_r.std(axis=0, ddof=1) / np.sqrt(na.n_cell)

                # plot the results
                y_max = max(np.max(mean_pref_fr + std_pref_fr), y_max)
                y_min = min(np.min(mean_pref_fr - std_pref_fr), y_min)

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
                    axs[1].fill_between(na.edges, mean_null_fr - std_null_fr, mean_null_fr + std_null_fr,
                                            alpha=0.25,
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

            elif arg is 'all':
                # cmap = plt.get_cmap('jet')
                # colors = cmap(np.linspace(0, 1.0, len(self.good_cells)))
                for j, cell in enumerate(self.good_cells):
                    mean_pref_fr = pref_fr[j].mean(axis=0)
                    std_pref_fr = pref_fr[j].std(axis=0, ddof=1) / np.sqrt(pref_fr[j].shape[0])
                    mean_null_fr = null_fr[j].mean(axis=0)
                    std_null_fr = null_fr[j].std(axis=0, ddof=1) / np.sqrt(null_fr[j].shape[0])

                    # plot the results
                    y_max = max(np.max(mean_pref_fr + std_pref_fr), y_max)
                    y_min = min(np.min(mean_pref_fr - std_pref_fr), y_min)

                    axs[0].plot(na.edges, mean_pref_fr, label=na.condition, c=color_list[k])
                    # Error bar
                    # axs[0].fill_between(na.edges, mean_pref_fr - std_pref_fr, mean_pref_fr + std_pref_fr, alpha=0.25,
                    #                      facecolor=color_list[k])
                    axs[0].set_ylim((y_min, y_max))
                    axs[0].set_title('Preferred Orientation')
                    axs[0].grid(True)
                    axs[0].set_xlabel('Time')
                    axs[0].set_xticks(na.edges[np.arange(na.n_time, step=4)])

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
                        ## error bar
                        # axs[1].fill_between(na.edges, mean_null_fr - std_null_fr, mean_null_fr + std_null_fr,
                        #                     alpha=0.25,
                        #                     facecolor=color_list[k])
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

        handles, label = axs[0].get_legend_handles_labels()
        axs[0].legend((handles[0], handles[-1]), (label[0], label[-1]), loc='upper left')

        if not os.path.exists('%sfiring_rate/' % self.figpath):
            os.makedirs('%sfiring_rate/' % self.figpath)

        filepath = '%sfiring_rate/%s%s%s_pop_FR' % (self.figpath, self.day, normal, name)
        i = 0
        while glob.glob('%s%i.*' % (filepath, i)):
            i += 1

        filepath = '%s%i' % (filepath, i)
        plt.savefig(filepath)
        plt.close(fig)

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
        axs.set_xticks(self.edges[np.arange(self.n_time, step=4)])
        axs.legend(loc='upper left')

        plt.show()
        plt.close(fig)

    def plot_tuning_curves(self, aligned=False, size=None):
        """
        Plot the tuning curve for all the good cells, for all conditions (each cell is a separate subplot, with every
        conditions). Plot comprise of the average firing rate for each orientation (with std error whiskers) with a von
        mises fit overlay
        """
        # if aligned:
        #     center = self.n_ort // 2
        #     self.aligned_ort(center, vis_lat_idx)
        #     print('aligned cells', self.good_cells)

        pref_ort = []
        for na in self.NA_list:
            pref_ort.append(na.get_pref_ort())

        print(pref_ort[0][self.good_cells])

        # initialize the figure
        if not size:
            size = min_square(self.good_cells.shape[0])
        fig, axs = plt.subplots(size, size, sharex=True, sharey=False)

        m = 0
        n = 0

        print(self.good_cells)
        for j, cell in enumerate(self.good_cells):
            for k, na in enumerate(self.NA_list):
                if aligned:
                    center = self.n_ort // 2
                    offset = center - pref_ort[k]
                    r = na.get_fr(cell=cell, times=self.vis_lat_idx)
                    theta = na.get_theta(offset=offset[cell], cell=cell)
                    # print(cell, pref_ort[k][cell], offset[cell])
                    vonmises_params = VonMises.fit(r, theta)
                    # vonmises_params[3] = 0
                    # vonmises_params[0] = 1
                    axs[m, n].axvline(x=np.radians(self.angles[center]), c=color_list[k])

                else:
                    r = na.get_fr(cell=cell, times=self.vis_lat_idx)
                    theta = na.get_theta(offset=0, cell=cell)
                    vonmises_params = VonMises.fit(r, theta)
                    axs[m, n].axvline(x=np.radians(self.angles[pref_ort[k][cell]]), c=color_list[k])

                # plot best fit + data
                thet = np.linspace(0, np.pi, 100)
                axs[m, n].plot(thet, VonMises.vonmises(thet, vonmises_params), label=na.condition, c=color_list[k])
                plot_wiskers(theta, r, axs[m, n], label='', color=color_list[k])

                # axs[m, n].axvline(x=vonmises_params[0])

            axs[m, n].set_title('Cell #%i, loc %i' % (cell, na.pref_loc[cell]))
            # axs[m, n].axis('off')
            m += 1
            if m % size == 0:
                n += 1
                m = 0
            if n and n % size == 0:
                n = 0
                handles, labels = axs[0, 0].get_legend_handles_labels()
                fig.legend(handles, labels, loc='lower right')
                save_fig(fig, '%stuning_curve/' % self.figpath, '%s_tuningCurve%i' % (self.day, int(self.vis_lat*100)))
                fig, axs = plt.subplots(size, size, sharex=True, sharey=False)
        if fig:
            handles, labels = axs[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower right')
            save_fig(fig, '%stuning_curve/' % self.figpath, '%s_tuningCurve%i' % (self.day, int(self.vis_lat * 100)))

    def plot_pop_tuning(self, arg='ovr'):

        center = self.n_ort // 2
        vis_lat_idx = int(np.argmin(np.abs(self.edges - self.vis_lat)))

        if arg is 'ovr':
            fig, axs = plt.subplots(1, 1)
        elif arg is 'all':
            fig, axs = plt.subplots(len(self.NA_list), sharey=True)

        thet = np.linspace(0, np.pi, 100)

        for k, na in enumerate(self.NA_list):
            fr = []
            ort = []
            pref_ort = na.get_pref_ort()
            for j, cell in enumerate(self.good_cells):
                offset = center - pref_ort[cell]
                r = na.get_fr(cell=cell, times=vis_lat_idx, normal='std')
                theta = na.get_theta(offset=offset, cell=cell)
                fr.append(r)
                ort.append(theta)

            if arg is 'ovr':
                r = np.zeros((1,))
                theta = np.zeros((1,))
                for j in range(len(fr)):
                    r = np.hstack((r, fr[j]))
                    theta = np.hstack((theta, ort[j]))

                r = r[1:]
                theta = theta[1:]
                print(r.shape)
                # r= np.reshape(r.T, (r.size,))
                # print('normal', np.min(r), np.max(r))
                # theta = np.reshape(theta.T, (theta.size,))
                vonmises_params = VonMises.fit(r, theta)
                axs.plot(thet, VonMises.vonmises(thet, vonmises_params), label=na.condition, c=color_list[k])
                plot_wiskers(theta, r, axs, label='', color=color_list[k])

            elif arg is 'all':
                cmap = plt.get_cmap('jet')
                colors = cmap(np.linspace(0, 1.0, len(self.good_cells)))
                for j, cell in enumerate(self.good_cells):
                    vonmises_params = VonMises.fit(fr[j], ort[j])
                    # print(vonmises_params)
                    # vonmises_params[3] = 0
                    # vonmises_params[0] = 1

                    axs[k].plot(thet, VonMises.vonmises(thet, vonmises_params), label=na.condition, c=colors[j])
                    plot_wiskers(ort[j], fr[j], axs[k], label='', color=colors[j])

                axs[k].set_ylim([0, 1])
                axs[k].set_title('%s' % na.condition)

        if arg is 'ovr':
            handles, labels = axs.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower right')

        if not os.path.exists('%stuning_curve/' % self.figpath):
            os.makedirs('%stuning_curve/' % self.figpath)

        filepath = '%stuning_curve/%s_tuningCurve%i' % (self.figpath, self.day, int(self.vis_lat*100))
        i = 0
        while glob.glob('%s%i.*' % (filepath, i)):
            i += 1

        filepath = '%s%i' % (filepath, i)

        fig.savefig(filepath, dpi=300)
        plt.close(fig)



