from eeglibrary.src import eeg_loader
import numpy as np
import scipy.signal
import librosa
import torch


windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}


class EEGParser:
    def __init__(self, eeg_conf, spect=False, normalize=False, augment=False):
        self.mat_col = eeg_conf['mat_col']
        self.sample_rate = eeg_conf['sample_rate']
        self.spect = spect
        if self.spect:
            self.window_stride = eeg_conf['window_stride']
            self.window_size = eeg_conf['window_size']
            self.window = windows.get(eeg_conf['window'], windows['hamming'])
        self.normalize = normalize
        self.augment = augment

    def parse_eeg(self, eeg_path) -> np.array:
        if self.augment:
            raise NotImplementedError
        else:
            eeg = eeg_loader.from_mat(eeg_path, mat_col='')

        if self.spect:
            y = self.to_spect(eeg)

        return y

    def to_spect(self, eeg):
        n_fft = int(eeg.sample_rate * self.window_size)
        win_length = n_fft
        hop_length = int(eeg.sample_rate * self.window_stride)
        # STFT
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=self.window)
        spect, phase = librosa.magphase(D)
        spect = torch.FloatTensor(spect)
        if self.normalize:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)

        return spect
