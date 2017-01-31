import numpy as np
import glob
from NeuronArray import NeuronArray as NA
from HelperFun import load


color_list = ['blue', 'black', 'red', 'green']


class RecordingDay:
    def __init__(self, day, conditions):
        path = glob.glob('data/%s*' % day)
        print(path[0])
        path = path[0]
        data = load(path)

        # Time axis
        self.edges = np.ravel(data['edges'])
        self.n_time = self.edges.shape[0]

        # Orientations
        self.angles = np.ravel(data['makeStim']['ort'][0, 0])
        self.n_ort = self.angles.shape[0]

        # Initialize
        n_condition = len(conditions)
        self.NA_list = []
        for i in n_condition:
            assert conditions[i] in data.keys()
            self.NA_list.append(NA(data, conditions[i], self.n_time, self.n_ort))

        if 'presac_retino_only' in data.keys():
            self.NA_fix = NA(data, 'presac_retino_only', self.n_time, self.n_ort)

        if 'presac_only' in data.keys():
            self.NA_remap = NA(data, 'presac_only', self.n_time, self.n_ort)



