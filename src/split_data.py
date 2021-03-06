from multiprocessing import Process
from pathlib import Path

import pandas as pd

subject_dir_names = ['Dog_1']#, 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
partial_name_list = ['train', 'val', 'test']
class_names = ['preictal', 'interictal']

from eeglibrary import from_mat
from eeglibrary.utils import split_args


def data_split(wave, out_dir, duration=10.0) -> list:
    """
    1. load eeg
    2. split
    3. save to out_dir
    4. return path where file is saved
    :return:
    """
    mat_col = wave.name[:-8] + str(int(wave.name[-8:-4]))
    Path(out_dir / mat_col).mkdir(exist_ok=True)
    eeg = from_mat(wave, mat_col)
    out_paths = []

    splitted_eeg = eeg.split(window_size=duration, window_stride='same')
    for i, eeg in enumerate(splitted_eeg):
        # print(eeg.values.shape[1])
        # assert eeg.values.shape[1] == 399, "eeg don't have enough length, which is 399"
        out_file = Path(out_dir / mat_col / '{}.pkl'.format(i * int(duration) * eeg.sr))
        eeg.to_pkl(out_file)
        out_paths.append(out_file.resolve())

    return out_paths


def make_manifest(out_dir, paths) -> None:
    """
    1. pathsからtrainとtestを分ける
    2. csvにしてout_dirに保存
    :return:
    """
    train_paths = paths['preictal'][:len(paths['preictal'])*8//10] + paths['interictal'][:len(paths['interictal'])*8//10]
    val_paths = paths['preictal'][len(paths['preictal'])*8//10:] + paths['interictal'][len(paths['interictal'])*8//10:]
    pd.DataFrame(train_paths).to_csv(out_dir / 'train_manifest.csv', header=None, index=False)
    pd.DataFrame(val_paths).to_csv(out_dir / 'val_manifest.csv', header=None, index=False)
    pd.DataFrame(paths['test']).to_csv(out_dir / 'test_manifest.csv', header=None, index=False)


def preprocess(args, patient_path, label_func):
    """
    1. for each wave data in each patient,
       data split with args.length and save to args.out_dir
    2. make manifessts
    TODO - multi processing
    """
    def initialize_folders(args, patient_name):
        assert Path(args.patients_dir).is_dir()
        Path(args.out_dir + '/' + patient_name).mkdir(parents=True, exist_ok=True)
        return Path(args.out_dir + '/' + patient_name)

    out_dir = initialize_folders(args, patient_path.name)

    wave_paths = {'preictal': [], 'interictal': [], 'test': []}
    print('{} is now processing...'.format(patient_path.name))
    phases = Path(patient_path).iterdir()
    for phase in phases:
        for wave in Path(phase).iterdir():
            class_name = label_func(wave.name)
            wave_paths[class_name].extend(data_split(wave, out_dir, duration=args.duration))
        make_manifest(out_dir, wave_paths)


def remove_mac_folder(path):
    # dirがあればremove_mac_folderを呼び、すべてファイルならば .で始まるファイルを削除
    folders = [p for p in Path(path).iterdir() if p.is_dir()]
    if folders:
        [remove_mac_folder(p) for p in folders]

    [p.unlink() for p in Path(path).iterdir() if p.name.startswith('.')]


def compile_manifests(args):
    patient_folders = [p for p in Path(args.out_dir).iterdir() if p.name[0] in ['D', 'P']]
    Path('{}/manifests'.format(args.out_dir)).mkdir(exist_ok=True)

    for part in partial_name_list:
        df = pd.DataFrame()
        part_manifest_paths = [p / '{}_manifest.csv'.format(part) for p in patient_folders]
        for part_manifest_path in part_manifest_paths:
            df = df.append(pd.read_csv(part_manifest_path, header=None))
        df.to_csv('{}/manifests/{}_manifest.csv'.format(args.out_dir, part), header=None, index=False)


def label_func(file_name):
    return file_name.split('_')[2]


if __name__ == '__main__':
    args = split_args().parse_args()
    Path(args.out_dir).mkdir(exist_ok=True, parents=True)
    remove_mac_folder(args.patients_dir)
    proc_list = []
    for patient in Path(args.patients_dir).iterdir():
        p = Process(target=preprocess, args=(args, patient, label_func, ))
        p.start()
        proc_list.append(p)

    [p.join() for p in proc_list]

    # compile_manifests(args)
