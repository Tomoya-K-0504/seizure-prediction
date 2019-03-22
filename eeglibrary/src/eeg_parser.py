from eeglibrary import eeg_loader
import numpy as np
import scipy.signal


windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}


class EEGParser:
    def __init__(self, eeg_conf, normalize=False, augment=False):
        self.mat_col = eeg_conf['mat_col']
        self.window_stride = eeg_conf['window_stride']
        self.window_size = eeg_conf['window_size']
        self.sample_rate = eeg_conf['sample_rate']
        self.wave_split_sec = eeg_conf['wave_split_sec']
        self.window = windows.get(eeg_conf['window'], windows['hamming'])
        self.normalize = normalize
        self.augment = augment

    def parse_eeg(self, eeg_path) -> np.array:
        if self.augment:
            raise NotImplementedError
        else:
            eeg = eeg_loader.from_mat(eeg_path, mat_col='')
            eegs = eeg.split(self.wave_split_sec)
            y = [eeg.split(self.window_size, self.window_stride) for eeg in eegs]

        return y
