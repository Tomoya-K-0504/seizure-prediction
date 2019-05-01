from __future__ import print_function, division

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import argparse

from eeglibrary.models.CNN import *
from eeglibrary.models.RNN import *
from args import test_args
from eeglibrary import EEGDataSet, EEGDataLoader, make_weights_for_balanced_classes, EEG
from utils import AverageMeter, init_seed, set_eeg_conf, set_model, init_device

supported_rnns = {
    'lstm': nn.LSTM,
    'rnn': nn.RNN,
    'gru': nn.GRU
}

supported_rnns_inv = dict((v, k) for k, v in supported_rnns.items())

subject_dir_names = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
partial_name_list = ['train', 'val', 'test']
class_names = ['interictal', 'preictal']


def set_dataloaders(args, eeg_conf, device='cpu'):
    dataset = EEGDataSet(args.test_manifest, eeg_conf, return_path=True)
    dataloader = EEGDataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                      pin_memory=True, shuffle=False)
    return dataloader


def test(args, device, class_names):

    eeg_conf = set_eeg_conf(args)
    dataloader = set_dataloaders(args, eeg_conf, device)
    model = set_model(args, eeg_conf, device, class_names)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    pred_list = []
    path_list = []

    for i, (inputs, paths) in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs = inputs.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        pred_list.extend(preds)
        # Transpose paths, but I don't know why dataloader outputs aukward
        path_list.extend([pd.DataFrame(paths).iloc[:, i].values for i in range(len(paths[0]))])

    def ensemble_preds(pred_list, path_list, sub_df, thresh):
        # もともとのmatファイルごとに振り分け直す
        patient_name = path_list[0][0].split('/')[-3]
        orig_mat_list = sub_df[sub_df['clip'].apply(lambda x: '_'.join(x.split('_')[:2])) == patient_name]
        ensembled_pred_list = []
        for orig_mat_name in orig_mat_list['clip']:
            seg_number = int(orig_mat_name[-8:-4])
            one_segment_preds = [pred for path, pred in zip(path_list[0], pred_list) if
                                 int(path.split('/')[-2].split('_')[-1]) == seg_number]
            ensembled_pred = int(sum(one_segment_preds) >= len(one_segment_preds) * thresh)
            ensembled_pred_list.append(ensembled_pred)
        orig_mat_list['preictal'] = ensembled_pred_list
        return orig_mat_list

    # preds to csv
    # sub_df = pd.read_csv('../output/sampleSubmission.csv')
    sub_df = pd.read_csv(args.sub_path, engine='python')
    thresh = args.thresh  # 1の割合がthreshを超えたら1と判断
    pred_df = ensemble_preds(pred_list, path_list, sub_df, thresh)
    sub_df.loc[pred_df.index, 'preictal'] = pred_df['preictal']
    sub_df.to_csv(args.sub_path, index=False)

    for subject in subject_dir_names:
        subject_df = sub_df.loc[sub_df['clip'].apply(lambda x: subject in x), 'preictal']
        print(subject, '\n', subject_df.value_counts(normalize=True))


if __name__ == '__main__':
    args = test_args().parse_args()
    init_seed(args)
    device = init_device(args)

    class_names = ['interictal', 'preictal']

    test(args, device, class_names)
