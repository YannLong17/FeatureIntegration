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
        signature = ''
        figpath = 'results/Decoding Analysis/%s/' % file  # where you want to save the figures

        alpha = 0.01

        rd = RD(file, conditions)



        if 'early trial' in args:
            bounds = [-0.15, -0.075]
            signature += 'early150_75'
        elif 'mid trial' in args:
            bounds = [-0.1, -0.05]
            signature += 'mid100_50'
        elif 'good trial' in args:
            bounds = [-0.075, -0.025]
            signature += 'sac75_25'
        elif 'late trial' in args:
            bounds = [-0.01, 0.05]
            signature += 'late'
        else:
            bounds = [-np.inf, np.inf]

        rd.trial_select(bounds)

        rd.cell_select(alpha)

        for na in rd.NA_list:
            print(signature, na.condition, ' Visual Latency: ',  round(na.visual_latency.mean(), 3), ' n cell: ',
            na.n_cell, 'n trial: ', na.n_trial)

        if 'savemat' in args:
            if 'load' in args:
                rd.set_save('load')
            elif 'overwrite' in args:
                rd.set_save('overwrite')
            else:
                rd.set_save()

        if 'firing rate' in args:
            normal = 'raw'
            if 'pink' in args:
                normal = 'pink'
            elif 'sub' in args:
                normal = 'sub'

            rd.firing_rate(normal)

            rd.plot_firing_rate(normal, signature)


        if 'tuning curve' in args:

            tempdict = {}
            for na in rd.NA_list:
                tempdict[na.condition] = na.plot_tuning_curves(figpath, file)

            # if 'savemat' in args:
            #     mydict['tuningParam'] = tempdict

        if 'decoding' in args:
            # choose the learner
            # Uncomment the learner you want to use
            # learner = SVC(kernel='linear', C=0.00002, class_weight='balanced', decision_function_shape='ovr')

            if 'quick' in args:
                learner = LogisticRegression(penalty='l2', multi_class='multinomial', solver='lbfgs', C=7.75)
                name = 'LR'
                n_fold = 5

            else:
                learner = ExtraTreesClassifier(n_estimators=5000, bootstrap=True, class_weight='balanced_subsample')
                name = 'ET'
                n_fold = 'max'

            # choose the scorer
            from sklearn.metrics import accuracy_score, make_scorer
            scorer = make_scorer(accuracy_score, greater_is_better=True)

            if 'equal' in args:
                rd.equalize_trials()
                name += '_eq'

            # smoothin param
            tau = 0.1

            if 'smooth' in args:
                name += '_tau%i' % int(tau * 1000)
                smooth = True
                rd.set_tau(tau)

            else: smooth = False

            if 'jumble' in args:
                name += '_JB'
                for na in rd.NA_list:
                    na.jumble()

            # Boothstrapping sample number
            if 'boothstrap' in args:
                booth = 250
                name += 'booth%i' % booth
            else: booth = 0

            rd.decode(learner, scorer, smooth, name, n_folds=n_fold, booth=booth)

            rd.plot_decoding_time_course(name)

            # if 'savemat' in args:
            #     tempdict = {}
            #     for na in rd.NA_list:
            #         tempdict[na.condition] = {'accuracy': na.decoding_tc, 'std_err': na.decoding_tc_err}
            #     tempdict['info'] = {'learner': name, 'smoothing time constant': tau}
            #     mydict['decoding'] = tempdict

        if 'orientation bias' in args:
            rd.plot_orientation_bias()

        if 'savemat' in args:
            rd.save_mat()

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
    conditions += ['presac', 'postsac']

    # Choose the file to analyse
    files = []
    files += ['p128']

    # Cell selection
    kosher = False

    # Main: Args
        # 'decoding' -- plot the decoding time course for conditions
            # 'smooth': causal filter on firing rate
            # 'jumble': remove the correlation stucture
            # 'equal': equalize the number of trials between conditions
            # 'boothstrap': boothstrap estimate for post condition

        # 'firing rate' -- plot the firing rate time course for conditions
            # 'raw': no baseline correction
            # 'pink': percentage increase with respect to baseline
            # 'sub': substract baseline

        # 'tuning curve' -- plot the tuning curve for each good cell at visual latency

        # 'savemat': saves the graph data to a matlab file
        #         'Overwrite' - replace the data on file for given conditions
        #         'load' - load the data on file if exists

        # 'write': output basic information to a text file

    args = []
    args += ['decoding', 'boothstrap']
    args += ['firing rate', 'sub']
    # args += ['tuning curve']
    args += ['savemat', 'load']
    # args += ['orientation bias']
    args += ['good trial']
    # args += ['quick']

    main(args, files, conditions)


