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
from HelperFun import load


def main(args, files, conditions):
    for file in files:
        path = glob.glob('data/%s*' % file)
        print(path[0])
        path = path[0]                                    # path to the data file
        figpath = 'results/Decoding Analysis/%s/' % file  # where you want to save the figures

        alpha = 0.01

        data = load(path)
        neuron_array_list = []
        for i in range(len(conditions)):
            neuron_array = NA(data, conditions[i], colors[i], location=location, n_locations=n_locations)
            neuron_array.cell_selection_kosher(alpha=0.05)
            print(neuron_array.condition, neuron_array.visual_latency.mean(), neuron_array.n_cell)
            neuron_array_list.append(neuron_array)

        if 'savemat' in args:
            mydict = {}

        if 'firing rate' in args:
            normal = 'raw'
            if 'pink' in args:
                normal = 'pink'
            elif 'sub' in args:
                normal = 'sub'

            if 'savemat' in args:
                mydict['firingRate'] = NA.plot_firing_rate(neuron_array_list, figpath, file, normal=normal, savemat=True)
            else:
                NA.plot_firing_rate(neuron_array_list, figpath, file, normal=normal)

        if 'tuning curve' in args:

            tempdict = {}
            for na in neuron_array_list:
                tempdict[na.condition] = na.plot_tuning_curves(figpath, file)

            if 'savemat' in args:
                mydict['tuningParam'] = tempdict

        if 'decoding' in args:
            # choose the learner
            # Uncomment the learner you want to use
            learner = ExtraTreesClassifier(n_estimators=500, bootstrap=True, class_weight='balanced_subsample')
            # learner = SVC(kernel='linear', C=0.00002, class_weight='balanced', decision_function_shape='ovr')
            # learner = LogisticRegression(penalty='l2', multi_class='multinomial', solver='lbfgs', C=7.75)
            # learner = PoissonNB()
            name = 'ET'  # Will appear in title and file name

            # choose the scorer
            from sklearn.metrics import accuracy_score, make_scorer
            scorer = make_scorer(accuracy_score, greater_is_better=True)

            NA.equalize_trials(neuron_array_list)
            name += '_eq'

            # smoothin param
            tau = 0.1

            if 'smooth' in args:
                name += '_tau%i' % int(tau * 1000)
                for na in neuron_array_list:
                    na.smooth(tau)

            if 'jumble' in args:
                name += '_JB'
                for na in neuron_array_list:
                    na.jumble()

            for na in neuron_array_list:
                na.decoding(learner, scorer, n_folds='max')

            NA.plot_decoding_time_course(neuron_array_list, figpath, file, name)

            if 'savemat' in args:
                tempdict = {}
                for na in neuron_array_list:
                    tempdict[na.condition] = {'accuracy': na.decoding_tc, 'std_err': na.decoding_tc_err}
                tempdict['info'] = {'learner': name, 'smoothing time constant': tau}
                mydict['decoding'] = tempdict

        if 'savemat' in args:
            if os.path.isfile('%s%s_data.mat' % (figpath, file)):
                print('old exists ')
                olddict = sio.loadmat('%s%s_data.mat' % (figpath, file))
                mydict = {**olddict, **mydict}

            tempdict = {}
            for na in neuron_array_list:
                tempdict[na.condition] = {'time': na.edges.ravel(), 'visual_latency': na.visual_latency, 'good_cells': na.good_cells}

            mydict['info'] = tempdict
            # print(mydict)

            sio.savemat('%s%s_data.mat' % (figpath, file), mdict=mydict)

        if 'write' in args:
            text_file = open("%soutput_alpha%i.txt" % (figpath, int(100*alpha)), "a")
            text_file.write('%s \n' % path)
            text_file.write('file, condition, visual_latency, n_good_cells, n_trials \n')
            for na in neuron_array_list:
                text_file.write('%s, %s, %f, %i, %i \n' % (file, na.condition, na.visual_latency.mean(), na.n_cell, na.n_trial))
            text_file.close()

if __name__ == '__main__':
    # choose the condition to analyse
    conditions = []
    # uncomment the conditions you want
    #
    conditions += ['presac', 'postsac', 'postsac_change']

    colors = ['blue', 'black', 'red']

    # Choose the file to analyse
    files = []
    files += ['p112']

    # Number of location, location of interest
    n_locations = 1
    location = 0

    # Main: Options
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

    main(['decoding', 'smooth', 'savemat'], files, conditions)

    main(['firing rate', 'raw', 'savemat'], files, conditions)

    main(['write'], files, conditions)

    main(['tuning curve', 'savemat'], files, conditions)
