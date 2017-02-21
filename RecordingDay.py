import numpy as np
import glob
import matplotlib.pyplot as plt
import os, glob

from NeuronArray import NeuronArray as NA
from HelperFun import *
from fittingFun import VonMises
from sklearn.cross_validation import cross_val_score, StratifiedShuffleSplit, StratifiedKFold


color_list = ['blue', 'black', 'red', 'green', 'orange', 'magenta']
good_cell_dic = {'p128': [15, 16, 30, 32, 34, 35, 40, 66, 80, 82, 83, 86, 87, 88, 89],
                 'p131': [34, 35, 40, 52, 67, 69, 71, 84, 87, 88, 89],
                 'p132': [15, 34, 40, 88, 89]
                 }


class RecordingDay:
    def __init__(self, day, conditions, alpha):
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

        # Good_cells
        self.good_cells = np.arange(96)
        if day in good_cell_dic.keys():
            self.good_cells = good_cell_dic[day]

        # Initialize
        n_condition = len(conditions)
        self.NA_list = []
        for i in range(n_condition):
            assert conditions[i] in data.keys()
            self.NA_list.append(NA(data, conditions[i]))

        # self.trial_select([-0.25, -0.025])
        self.cell_select(alpha)

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

    def aligned_ort(self, center, time, arg='first'):
        """
        For all conditions, align all cell to the the same prefered orientation, according to the first condition prefered orientation.
        :return:
        """
        pref_ort = self.NA_list[0].get_pref_ort(time)
        fr = []
        ort = []
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
            r = na.get_fr(times=np.argmin(np.abs(self.edges - time)))
            theta = np.zeros((na.n_trial, na.n_cell))
            for k, cell in enumerate(self.good_cells):
                offset = center - pref_ort[cell]
                theta[:, k] = na.get_theta(offset)
            fr.append(r)
            ort.append(theta)

        return fr, ort

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

    def firing_rate(self, normal):
        for na in self.NA_list:
            if na.exist_data('fr%s' % normal):
                if self.save is 'over':
                    na.firing_rate_data(normal)
            else:
                na.firing_rate_data(normal)

    def tuning(self, time):
        for na in self.NA_list:
            if na.exist_data('tuning%i' % time):
                if self.save is 'over':
                    na.tuning(time)
            else:
                na.tuning(time)

    ### Plotting Fun ###
    def plot_decoding_time_course(self, name):
        # plot the result
        fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)

        for k, na in enumerate(self.NA_list):
            decoding_tc = na.data_dict['decode']['decoding_tc']
            decoding_tc_err = na.data_dict['decode']['decoding_tc_err']

            l1, = axs.plot(na.edges, decoding_tc, label=na.condition, c=color_list[k], linewidth=.5)
            axs.fill_between(na.edges, decoding_tc + decoding_tc_err, decoding_tc - decoding_tc_err,
                             facecolor=color_list[k], alpha=0.25)

        axs.axhline(y=1. / self.n_ort)
        axs.set_ylabel('Accuracy')
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

    def plot_firing_rate(self, normal, name, null_orientation=True, savemat=False):
        """ find the average and std err firing rate for prefered and null orientation for all times"""

        if null_orientation:
            n_plot = 2
        else:
            n_plot = 1

        fig, axs = plt.subplots(n_plot, 1, sharex=True)

        y_min = np.inf
        y_max = -np.inf

        for k, na in enumerate(self.NA_list):

            pref_fr = na.data_dict['fr%s' % normal]['pref_fr']
            null_fr = na.data_dict['fr%s' % normal]['null_fr']

            mean_pref_fr = pref_fr.mean(axis=1)
            std_pref_fr = pref_fr.std(axis=1, ddof=1) / np.sqrt(na.n_cell)
            mean_null_fr = null_fr.mean(axis=1)
            std_null_fr = null_fr.std(axis=1, ddof=1) / np.sqrt(na.n_cell)
        #
        #     if savemat:
        #         dict[na.condition] = {'Prefered_Orientation': pref_fr, 'Null_Orientation': null_fr}

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

    def plot_tuning_curves(self, vis_lat, aligned=False, size=None):
        """
        Plot the tuning curve for all the good cells, for all conditions (each cell is a separate subplot, with every
        conditions). Plot comprise of the average firing rate for each orientation (with std error whiskers) with a von
        mises fit overlay
        """

        if aligned:
            center = self.n_ort // 2
            fr, ort = self.aligned_ort(center, vis_lat)
            print('aligned cells', self.good_cells)

        pref_ort = []
        for na in self.NA_list:
            pref_ort.append(na.get_pref_ort(vis_lat))

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
                    r = fr[k][:, j]
                    theta = ort[k][:, j]
                    vonmises_params = VonMises.fit(r, theta)
                    # vonmises_params[3] = 0
                    # vonmises_params[0] = 1
                    axs[m, n].axvline(x=np.radians(self.angles[center]), c=color_list[k])

                else:
                    r = na.data_dict['tuning']['fr'][:, j]
                    theta = na.data_dict['tuning']['ort']
                    vonmises_params = na.data_dict['tuning']['vonmises_param'][:, j]
                    axs[m, n].axvline(x=np.radians(self.angles[pref_ort[k][cell]]), c=color_list[k])

                # plot best fit + data
                thet = np.linspace(0, np.pi, 100)
                axs[m, n].plot(thet, VonMises.vonmises(thet, vonmises_params), label=na.condition, c=color_list[k])
                plot_wiskers(theta, r, axs[m, n], label='', color=color_list[k])

                # axs[m, n].axvline(x=vonmises_params[0])

            axs[m, n].set_title('Cell #%i' % (cell))
            # axs[m, n].axis('off')
            m += 1
            if m % size == 0:
                n += 1
                m = 0
            if n and n % size == 0:
                n = 0
                handles, labels = axs[0, 0].get_legend_handles_labels()
                fig.legend(handles, labels, loc='lower right')
                save_fig(fig, '%stuning_curve/' % self.figpath, '%s_tuningCurve%i' % (self.day, int(vis_lat*100)))
                fig, axs = plt.subplots(size, size, sharex=True, sharey=False)
        if fig:
            handles, labels = axs[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower right')
            save_fig(fig, '%stuning_curve/' % self.figpath, '%s_tuningCurve%i' % (self.day, int(vis_lat * 100)))

    def plot_pop_tuning(self, vis_lat, normal=True, arg='ovr'):

        center = self.n_ort // 2
        fr, ort = self.aligned_ort(center, vis_lat)

        if arg is 'ovr':
            fig, axs = plt.subplots(1, 1)
        elif arg is 'all':
            fig, axs = plt.subplots(len(self.NA_list), sharey=True)

        thet = np.linspace(0, np.pi, 100)

        for k, na in enumerate(self.NA_list):
            r, theta = fr[k], ort[k]

            if normal:
                for j, cell in enumerate(self.good_cells):
                    mini = np.inf
                    maxi = -np.inf
                    for o in np.unique(theta[:, j]):
                        mu = r[theta[:, j] == o, j].mean()
                        if mu < mini:
                            mini = mu
                        if mu > maxi:
                            maxi = mu
                    # r[:, j] = (r[:, j] - np.min(r[:, j]))/(np.max(r[:, j])-np.min(r[:, j]))
                    # temp = np.maximum(r[:, j] - mini, np.zeros(r[:, j].shape))
                    r[:, j] = (r[:, j] - mini)/(maxi-mini)

            if arg is 'ovr':
                r = np.reshape(r.T, (r.size,))
                # print('normal', np.min(r), np.max(r))
                theta = np.reshape(theta.T, (theta.size,))
                vonmises_params = VonMises.fit(r, theta)
                axs.plot(thet, VonMises.vonmises(thet, vonmises_params), label=na.condition, c=color_list[k])
                plot_wiskers(theta, r, axs, label='', color=color_list[k])

            elif arg is 'all':
                cmap = plt.get_cmap('jet')
                colors = cmap(np.linspace(0, 1.0, len(self.good_cells)))
                for j, cell in enumerate(self.good_cells):
                    vonmises_params = VonMises.fit(r[:, j], theta[:, j])
                    # print(vonmises_params)
                    # vonmises_params[3] = 0
                    # vonmises_params[0] = 1

                    axs[k].plot(thet, VonMises.vonmises(thet, vonmises_params), label=na.condition, c=colors[j])
                    plot_wiskers(theta[:, j], r[:, j], axs[k], label='', color=colors[j])

                axs[k].set_ylim([0, 1])
                axs[k].set_title('%s' % na.condition)

        if arg is 'ovr':
            handles, labels = axs.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower right')

        if not os.path.exists('%stuning_curve/' % self.figpath):
            os.makedirs('%stuning_curve/' % self.figpath)

        filepath = '%stuning_curve/%s_tuningCurve%i' % (self.figpath, self.day, int(vis_lat*100))
        i = 0
        while glob.glob('%s%i.*' % (filepath, i)):
            i += 1

        filepath = '%s%i' % (filepath, i)

        fig.savefig(filepath, dpi=300)
        plt.close(fig)



