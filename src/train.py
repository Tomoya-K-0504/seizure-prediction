from __future__ import print_function, division
import os, sys, pdb, time
from pathlib import Path
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pickle
from tqdm import tqdm
import torch
from torch import nn
import torchvision
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
from sklearn import metrics
from random import shuffle
import random
random.seed(seed)
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, transforms, utils

from eeglibrary import EEG, EEGDataSet, EEGDataLoader, make_weights_for_balanced_classes
from eeglibrary.models.CNN import *
from eeglibrary.models.RNN import *
from args import train_args
from utils import AverageMeter


supported_rnns = {
    'lstm': nn.LSTM,
    'rnn': nn.RNN,
    'gru': nn.GRU
}
supported_rnns_inv = dict((v, k) for k, v in supported_rnns.items())

subject_dir_names = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
partial_name_list = ['train', 'val', 'test']
class_names = ['interictal', 'preictal']


def init_seed(args):
    # Set seeds for determinism
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def set_model(args, eeg_conf):
    if args.model_name == 'cnn_1_16_399':
        model = cnn_1_16_399(n_labels=len(class_names))
    if args.model_name == 'cnn_16_751_751':
        model = cnn_16_751_751(n_labels=len(class_names))
    if args.model_name == 'rnn_16_751_751':
        cnn, out_ftrs = cnn_ftrs_16_751_751(n_labels=len(class_names))
        model = RNN(cnn, out_ftrs, args.batch_size, args.rnn_type, class_names, eeg_conf=eeg_conf)
    else:
        raise NotImplementedError

    print(model)

    return model


def set_eeg_conf(args):
    eeg_conf = dict(spect=args.spect,
                    window_size=args.window_size,
                    window_stride=args.window_stride,
                    window='hamming',
                    sample_rate=args.sample_rate)
    return eeg_conf


def set_dataloaders(args, eeg_conf):
    manifests = [args.train_manifest, args.val_manifest, args.test_manifest]

    dataloaders = {}
    for part, manifest in zip(partial_name_list, manifests):
        if part in ['train', 'val']:
            dataset = EEGDataSet(manifest, eeg_conf, class_names)
            weights = make_weights_for_balanced_classes(dataset.labels_index(), len(class_names))
            sampler = WeightedRandomSampler(weights, args.batch_size)
            dataloaders[part] = EEGDataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              pin_memory=True, shuffle=False, sampler=sampler)
        else:
            dataset = EEGDataSet(manifest, eeg_conf, return_path=True)
            dataloaders[part] = EEGDataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              pin_memory=True, shuffle=False)
    return dataloaders


if __name__ == '__main__':

    args = train_args()
    init_seed(args)

    device = torch.device("cuda" if args.cuda else "cpu")

    Path(args.model_dir).mkdir(exist_ok=True)

    start_epoch, start_iter, optim_state = 0, 0, None
    best_loss, best_auc, losses, aucs = {}, {}, {}, {}
    for phase in ['train', 'val']:
        best_loss[phase], best_auc[phase], losses[phase], aucs[phase] = 1000, 0, AverageMeter(), AverageMeter()

    eeg_conf = set_eeg_conf(args)
    model = set_model(args, eeg_conf)
    model = model.to(device)
    dataloaders = set_dataloaders(args, eeg_conf)

    parameters = model.parameters()
    optimizer = torch.optim.SGD(parameters, lr=args.lr,
                                momentum=args.momentum, nesterov=True, weight_decay=1e-5)

    criterion = nn.BCELoss()
    batch_time = AverageMeter()

    for epoch in range(start_epoch, args.epochs):
        break
        end = time.time()
        start_epoch_time = time.time()

        for phase in ['train', 'val']:
            for i, (inputs, labels) in tqdm(enumerate(dataloaders[phase]), total=len(dataloaders[phase])):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    pred_prob = outputs[:, 1].float()
                    loss = criterion(pred_prob, labels.float())

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                losses[phase].update(loss.item() / inputs.size(0), inputs.size(0))
                aucs[phase].update(
                    metrics.auc(labels.cpu(), pred_prob.cpu().detach().numpy()) / inputs.size(0),
                    inputs.size(0)
                )

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                if not args.silent:
                    if phase == 'val':
                        print('validation results')
                    print('Epoch: [{0}][{1}/{2}] \tTime {batch_time.val:.3f} ({batch_time.avg:.3f}) \t'
                          'AUC {auc.val:.3f} ({auc.avg:.3f}) \tLoss {loss.val:.4f} ({loss.avg:.4f}) \t'.format(
                            epoch, (i + 1), len(dataloaders[phase]), batch_time=batch_time,
                            auc=aucs[phase], loss=losses[phase]))

            if losses[phase].avg < best_loss[phase]:
                best_loss[phase] = losses[phase].avg

            if aucs[phase].avg < best_auc[phase]:
                best_auc[phase] = aucs[phase].avg
                if phase == 'val':
                    print("Found better validated model, saving to %s" % args.model_path)
                    torch.save(
                        RNN.serialize(model, optimizer=optimizer, epoch=epoch, loss_results=loss_results,
                                             wer_results=wer_results, cer_results=cer_results)
                        , args.model_path)

                    # if not args.no_shuffle:
                    #     print("Shuffling batches...")
                    #     train_sampler.shuffle(epoch)

            # anneal lr
            param_groups = optimizer.param_groups
            for g in param_groups:
                g['lr'] = g['lr'] / args.learning_anneal
            print('Learning rate annealed to: {lr:.6f}'.format(lr=g['lr']))

            losses[phase].reset()
            aucs[phase].reset()

    # test phase
    pred_list = []
    path_list = []
    for i, (inputs, paths) in tqdm(enumerate(dataloaders['test']), total=len(dataloaders['test'])):
        inputs = inputs.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        pred_list.extend(preds)
        path_list.extend(paths)
        break

    def ensemble_preds(pred_list, path_list, sub_df):
        # もともとのmatファイルごとに振り分け直す
        patient_name = path_list[0].split('/')[-3]
        orig_mat_list = sub_df[sub_df['clip'].apply(lambda x: '_'.join(x.split('_')[:2])) == patient_name]
        ensembled_pred_list = []
        for orig_mat_name in orig_mat_list['clip']:
            _ = int(path_list[0].split('/')[-2].split('_')[-1])
            seg_number = int(orig_mat_name[-8:-4])
            one_segment_preds = [pred for path, pred in zip(path_list, pred_list) if
                                 int(path.split('/')[-2].split('_')[-1]) == seg_number]
            ensembled_pred = int(sum(one_segment_preds) >= len(one_segment_preds) / 2)
            ensembled_pred_list.append(ensembled_pred)
        orig_mat_list['preictal'] = ensembled_pred_list
        return orig_mat_list

    # preds to csv
    sub_df = pd.read_csv('../output/sampleSubmission.csv')
    pred_df = ensemble_preds(pred_list, path_list, sub_df)
    sub_df.loc[pred_df.index, 'preictal'] = pred_df['preictal']
    pd.DataFrame(pred_list)
