import numpy as np
from scipy.stats import ks_2samp
import scipy.io as sio
import matplotlib.pyplot as plt
import os, glob



# Helper Function
def load(file):
    """
    :param file: path to a .mat file
    :return: data structure located at file
    """
    dat = sio.loadmat(file, squeeze_me=False)
    return dat


def build_static(dat, condition, times, location=0, n_locations=1, noProbe=False):
    """
    Forms a firing rate matrix X (n_trials, n_cells, n_times), along with a the corresponding orientation Y (n_trials,)
    :param dat: data structure, data['condition'][n_cells, n_orientations * n_location] [n_times * n_trials]
    :param condition: string, what condition you want
    :param times: list of times you want
    :param location: scalar, int, location of the stimulus
    :param n_locations: scalar, int, total number of location
    :return: numpy arrray X (n_trials, n_cells, n_times) and Y (n_trials, )
    """
    assert location in np.arange(n_locations)
    n_times = np.asarray(times).shape[0]
    n_orientations = (dat['presac'].shape[1] - 1)//n_locations
    X = np.zeros((1, 96, n_times))
    y = np.zeros((1, ))
    probe_latency = np.zeros((1, ))

    if condition is 'presac':
        lat_info = 'prbonset1'
    elif condition is 'postsac':
        lat_info = 'prbonset2'
    elif condition is 'postsac_change':
        lat_info = 'prbonset3'
    elif condition is 'presac_only':
        lat_info = 'prbonset4'
    else:
        lat_info = False

    if lat_info not in dat.keys():
        lat_info = False

    if noProbe:
        n_trials = dat[condition][0, -1][times, :].shape[1]
        X = np.zeros((n_trials, 96, n_times))
        for channel in range(96):
            X[:, channel, :] = dat[condition][channel, -1][times, :].T

        y = np.hstack((y, (np.full((n_trials, ), -1, dtype='int32'))))
        if lat_info:
            probe_latency = np.hstack((probe_latency, dat[lat_info][0, -1][0,:]))

    else:
        for i, orientation in enumerate(np.arange(location, n_orientations*n_locations, n_locations)):
            n_trials = dat[condition][0, orientation][times, :].shape[1]
            temp = np.zeros((n_trials, 96, n_times))
            for channel in range(96):
                temp[:, channel, :] = dat[condition][channel, orientation][times, :].T

            X = np.vstack((X, temp))
            y = np.hstack((y, (np.full((n_trials,), i, dtype='int32'))))
            if lat_info:
                probe_latency = np.hstack((probe_latency, dat[lat_info][0, orientation][0,:]))

    if n_times == 1:
        X, y, probe_latency = X[1:, :, 0], y[1:, ], probe_latency[1:, ]
    else:
        X, y, probe_latency = X[1:, ...], y[1:, ], probe_latency[1:, ]

    return X, y, probe_latency



def ks_test(X, X_no):
    _, n_cell, n_time = X.shape
    P_val = np.ones((n_cell, n_time))

    for t in range(n_time):
        for i in range(n_cell):
            D, pval = ks_2samp(X_no[:, i, t], X[:, i, t])
            P_val[i, t] = pval

    return P_val


def ks_test_baseline(X, baseline_mask):
    baseline = X[:, :, baseline_mask].mean(axis=2)
    _, n_cell, n_time = X.shape
    P_val = np.ones((n_cell, n_time))

    for t in range(n_time):
        for i in range(n_cell):
            D, pval = ks_2samp(baseline[:, i], X[:, i, t])
            P_val[i, t] = pval

    return P_val

def find_peak(x, y):
    max = 0
    pref_or = 0

    for angle in np.unique(y):
        temp = np.mean(x[y == angle])
        if temp > max:
            max = temp
            pref_or = angle

    return pref_or


def find_closest(array, target):
    """ returns the idx in the array closest to the target angle (rad)
    :param array: numpy array
    :param target: scalar
    :return: idx of element closest to the target
    """
    target = target % np.pi                                 # our angles are always in range [0, pi]!
    idx = np.argmin(np.abs(array - target))
    # return array[idx]
    return idx


def find_null(array, target):
    """
    returns the idx in the array closest to the target angle + pi/2 (rad)
    :param array: numpy array
    :param target: scalar
    :return: idx of element closest to the target + pi/2
    """
    target = (target + np.pi/2) % np.pi                    # our angles are always in range [0, pi]!
    idx = np.argmin(np.abs(array - target))
    # return array[idx]
    return idx


def max_folds(X, y):
    """
    The maximum number of cross validation folds is equal to the number of trials in the least populated orientation
    (because we need at least one trial from each orientation in each fold)
    :param X: numpy array (n_trials, ...), firing rate
    :param y: numpy array (n_trials, ), stimulus orientation
    :return: scalar, int, maximum number of cross validation folds
    """
    n_folds = np.inf

    for i in np.unique(y):
        n_trials = X[y == i, ...].shape[0]
        if n_trials < n_folds:
            n_folds = n_trials

    return n_folds


def min_square(n):
    """ Return the smallest square qrid that can fit n
    """
    return int(np.ceil(np.sqrt(n)))


def plot_wiskers(theta, r, ax, label, color):
    unique = np.unique(theta)
    n_angles = unique.shape[0]

    mean = np.zeros((n_angles,))
    std = np.zeros((n_angles,))

    for i, angle in enumerate(unique):
        y = r[theta == angle]
        mean[i,] = np.mean(y)
        std[i,] = np.std(y, ddof=1) / np.sqrt(len(y))

    ax.errorbar(unique, mean, std, linestyle='None', marker='.', label=label, color=color)


def save_fig(fig, directory, name):
    if not os.path.exists(directory):
        os.makedirs(directory)

    filepath = '%s%s' % (directory, name)
    i = 0
    while glob.glob('%s%i.*' % (filepath, i)):
        i += 1

    filepath = '%s%i' % (filepath, i)

    fig.savefig(filepath, dpi=300)
    plt.close(fig)

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict