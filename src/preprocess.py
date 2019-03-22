import os, sys, pdb, time
from pathlib import Path
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pickle
import argparse
from tqdm import tqdm
subject_dir_names = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
partial_name_list = ['train', 'val', 'test']
class_names = ['preictal', 'interictal']

from eeglibrary import EEG
from eeglibrary import from_mat


def set_args():
    parser = argparse.ArgumentParser(description='DeepSpeech training')
    parser.add_argument('--out-dir', metavar='DIR',
                        help='directory to save splitted data', default='input/splitted')
    parser.add_argument('--patients-dir', metavar='DIR',
                        help='directory where patients data placed', default='input/splitted')
    parser.add_argument('--duration', type=float,
                        help='duration of one splitted wave', default=1.0)

    return parser.parse_args()


def data_split(args, wave) -> list:
    """
    1. load eeg
    2. split
    3. save to out_dir
    4. return path where file is saved
    :return:
    """
    mat_col = wave.name[:-8] + str(int(wave.name[-8:-4]))
    eeg = from_mat(wave, mat_col)
    out_paths = []

    splitted_eeg = eeg.split(window_size=1.0, window_stride='same')
    for i, eeg in enumerate(splitted_eeg):
        # print(eeg.values.shape[1])
        # assert eeg.values.shape[1] == 399, "eeg don't have enough length, which is 399"
        patient_dir = Path(args.out_dir) / '_'.join(mat_col.split('_')[:2])
        Path(patient_dir).mkdir(exist_ok=True)
        out_file = Path(args.out_dir) / patient_dir / '{}_{}.pkl'.format(mat_col, i*eeg.sr)
        eeg.to_pkl(out_file)
        out_paths.append(out_file.resolve())

    return out_paths


def make_manifest(args, wave_paths) -> None:
    """
    1. wave_pathsからtrainとtestを分ける
    2. csvにしてout_dirに保存
    :return:
    """
    for part in list(wave_paths.keys()):
        pd.DataFrame(wave_paths[part]).to_csv('{}/{}_manifest.csv'.format(args.out_dir, part))


def preprocess(args):
    """
    1. for each wave data in each patient,
       data split with args.length and save to args.out_dir
    2. make manifessts
    TODO - multi processing
    """
    def initialize_folders(args):
        assert Path(args.patients_dir).is_dir()
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    def remove_mac_folder(paths):
        return [p for p in paths if not p.name.startswith('.')]

    initialize_folders(args)
    wave_paths = {part: [] for part in partial_name_list}
    for patient in remove_mac_folder(list(Path(args.patients_dir).iterdir())):
        partials = remove_mac_folder(list(Path(patient).iterdir()))
        for part in partials:
            waves = remove_mac_folder(list(Path(part).iterdir()))
            for wave in tqdm(waves):
                wave_paths[part.name].extend(data_split(args, wave))

    make_manifest(args, wave_paths)


if __name__ == '__main__':
    args = set_args()
    preprocess(args)
