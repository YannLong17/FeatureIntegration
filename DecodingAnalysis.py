import os
import scipy.io as sio
import matplotlib.pyplot as plt
import glob
import numpy as np

# Sklearn
from NaiveBayes import PoissonNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from NeuronArray import NeuronArray as NA
from RecordingDay import RecordingDay as RD
from HelperFun import load


def main(args, files, conditions):
    for file in files:
        figpath = 'results/Decoding Analysis/%s/' % file  # where you want to save the figures

        alpha = 0.01

        rd = RD(file, conditions)

        rd.cell_select(alpha)

        for na in rd.NA_list:
            print(na.condition, na.visual_latency.mean(), na.n_cell)
            print(na.good_cells)

        if 'savemat' in args:
            mydict = {}

        if 'firing rate' in args:
            normal = 'raw'
            if 'pink' in args:
                normal = 'pink'
            elif 'sub' in args:
                normal = 'sub'

            if 'savemat' in args:
                mydict['firingRate'] = rd.plot_firing_rate(normal=normal, savemat=True)
            else:
                rd.plot_firing_rate(normal=normal)

        if 'tuning curve' in args:

            tempdict = {}
            for na in rd.NA_list:
                tempdict[na.condition] = na.plot_tuning_curves(figpath, file)

            if 'savemat' in args:
                mydict['tuningParam'] = tempdict

        if 'decoding' in args:
            # choose the learner
            # Uncomment the learner you want to use
            learner = ExtraTreesClassifier(n_estimators=5000, bootstrap=True, class_weight='balanced_subsample')
            # learner = SVC(kernel='linear', C=0.00002, class_weight='balanced', decision_function_shape='ovr')
            # learner = LogisticRegression(penalty='l2', multi_class='multinomial', solver='lbfgs', C=7.75)
            # learner = PoissonNB()
            name = 'ET'  # Will appear in title and file name

            # choose the scorer
            from sklearn.metrics import accuracy_score, make_scorer
            scorer = make_scorer(accuracy_score, greater_is_better=True)

            rd.equalize_trials()
            name += '_eq'

            # smoothin param
            tau = 0.1

            if 'smooth' in args:
                name += '_tau%i' % int(tau * 1000)
                smooth = tau
            else: smooth = False

            if 'jumble' in args:
                name += '_JB'
                for na in rd.NA_list:
                    na.jumble()

            for na in rd.NA_list:
                na.decoding(learner, scorer, smooth)

            rd.plot_decoding_time_course(figpath, name)

            if 'savemat' in args:
                tempdict = {}
                for na in rd.NA_list:
                    tempdict[na.condition] = {'accuracy': na.decoding_tc, 'std_err': na.decoding_tc_err}
                tempdict['info'] = {'learner': name, 'smoothing time constant': tau}
                mydict['decoding'] = tempdict

        if 'orientation bias' in args:
            rd.plot_orientation_bias()

        if 'savemat' in args:
            if os.path.isfile('%s%s_data.mat' % (figpath, file)):
                print('old exists ')
                olddict = sio.loadmat('%s%s_data.mat' % (figpath, file))
                mydict = {**olddict, **mydict}

            tempdict = {}
            for na in rd.NA_list:
                tempdict[na.condition] = {'time': na.edges.ravel(), 'visual_latency': na.visual_latency, 'good_cells': na.good_cells}

            mydict['info'] = tempdict
            # print(mydict)

            sio.savemat('%s%s_data.mat' % (figpath, file), mdict=mydict)

        if 'write' in args:
            text_file = open("%soutput_alpha%i.txt" % (rd.day, int(100*alpha)), "a")
            text_file.write('%s \n' % rd.figpath)
            text_file.write('file, condition, visual_latency, n_good_cells, n_trials \n')
            for na in rd.NA_list:
                text_file.write('%s, %s, %f, %i, %i \n' % (file, na.condition, na.visual_latency.mean(), na.n_cell, na.n_trial))
            text_file.close()

if __name__ == '__main__':
    # choose the condition to analyse
    conditions = []
    # uncomment the conditions you want
    #
    conditions += ['presac', 'postsac', 'postsac_change']

    # Choose the file to analyse
    files = []
    files += ['p121']

    # Cell selection
    kosher = False

    # Main: Args
        # 'decoding' -- plot the decoding time course for conditions
            # 'smooth': causal filter on firing rate
            # 'jumble': remove the correlation stucture

        # 'firing rate' -- plot the firing rate time course for conditions
            # 'raw': no baseline correction
            # 'pink': percentage increase with respect to baseline
            # 'sub': substract baseline

        # 'tuning curve' -- plot the tuning curve for each good cell at visual latency

        # 'savemat': saves the graph data to a matlab file

        # 'write': output basic information to a text file

    args = []
    # args += ['decoding', 'smooth']
    # args += ['firing rate', 'raw']
    # args += ['tuning curve']
    # args += ['savemat']
    args += ['orientation bias']

    main(args, files, conditions)


