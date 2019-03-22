import os, sys, pdb, time
from pathlib import Path
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pickle


FILE_FORMAT = ['.mat']


class EEG:
    """
    要件: - もともとのファイル形式に関係なく、eegに関するデータにアクセスできること
            - 波形データ np.array: values
            - チャンネル名 list(str): channel_list
            - チャンネル数 int: num_channel
            - 波形長 float: len_sec
            - サンプリング周波数 sr: float
            -
            -
            -
    """
    def __init__(self, values, channel_list, len_sec, sr, header=None):
        self.values = values
        self.channel_list = channel_list
        self.len_sec = len_sec
        self.sr = sr
        self.header = header

    def info(self):
        print('Data: \n {}'.format(self.values))
        print('Data Shape: \t {}'.format(self.values.shape))
        print('Data length sec: \t {}'.format(self.len_sec))
        print('Data sampling frequency: \t {}'.format(self.sr))
        print('All channels: \n {}'.format(self.channel_list))
        print('Number of channels: \t {}'.format(len(self.channel_list)))
        # print('Data sequense: \t {}'.format(data['sequence'][0][0]))

    @classmethod
    def load_pkl(cls, file_path):
        with open(file_path, mode='rb') as f:
            eeg = pickle.load(f)
        return eeg

    def __repr__(self):
        self.info()
        return ""

    def to_pkl(self, file_path):
        with open(file_path, mode='wb') as f:
            pickle.dump(self, f)

    def split(self, window_size=0.5, window_stride=0.0, padding=0.0) -> list:
        n_eeg = (self.len_sec - window_size) // window_stride
        if padding == 'same':
            padding = (self.len_sec - (n_eeg * window_stride + window_size)) / 2
        else:
            n_eeg = (self.len_sec + padding * 2 - window_size) // window_stride

        # add padding
        padded_waves = self.values

        splitted_eegs = []

        for i in range(n_eeg):
            eeg = copy.deepcopy(self)
            eeg.values = self.values[i * self.sr * window_size:(i + 1) * self.sr * window_size]
            eeg.len_sec = window_size
            splitted_eegs.append(eeg)

        return splitted_eegs


if __name__ == '__main__':
    from pathlib import Path
    from eeglibrary import eeg_parser
    data_dir = Path('../input')
    eeg_conf = dict(sample_rate=16000,
                    window_size=0.02,
                    window_stride=0.01,
                    window='hamming',
                    wave_split_sec=2.0,
                    noise_dir=None,
                    noise_prob=0.4,
                    noise_levels=(0.0, 0.5))

    eeg = eeg_parser.EEGParser(eeg_conf).parse_eeg('/home/tomoya/workspace/kaggle/seizure-prediction/input/Dog_1/train')
