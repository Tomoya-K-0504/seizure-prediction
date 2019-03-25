from __future__ import print_function, division
import os, sys, pdb, time
from pathlib import Path
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pickle

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
                                              shuffle=False, sampler=sampler)
        else:
            dataset = EEGDataSet(manifest, eeg_conf)
            dataloaders[part] = EEGDataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              pin_memory=True, shuffle=False)
    return dataloaders

    # data_transforms = {
    #     'train': transforms.Compose([
    #         transforms.ToTensor(),
    #     ]),
    #     'val': transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ]),
    # }


if __name__ == '__main__':

    args = train_args()
    init_seed(args)

    device = torch.device("cuda" if args.cuda else "cpu")

    Path(args.model_dir).mkdir(exist_ok=True)

    roc_results, prec_results, rec_results = torch.Tensor(args.epochs), torch.Tensor(args.epochs), torch.Tensor(
        args.epochs)
    best_loss = 100000.0
    avg_loss, avg_auc, start_epoch, start_iter, optim_state = 0, 0, 0, 0, None

    eeg_conf = set_eeg_conf(args)
    model = set_model(args, eeg_conf)
    model = model.to(device)
    dataloaders = set_dataloaders(args, eeg_conf)

    parameters = model.parameters()
    optimizer = torch.optim.SGD(parameters, lr=args.lr,
                                momentum=args.momentum, nesterov=True, weight_decay=1e-5)

    criterion = nn.BCELoss()
    batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

    # for a in dataloaders['train']:
    #     b = ''

    for epoch in range(start_epoch, args.epochs):
        end = time.time()
        start_epoch_time = time.time()

        for phase in ['train', 'val']:
            #         if phase == 'train':
            #             scheduler.step()

            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                # measure data loading time
                data_time.update(time.time() - end)
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

                avg_loss += loss.item()
                if device == 'cuda':
                    pred_prob = pred_prob.cpu()
                avg_auc += metrics.auc(labels.cpu(), pred_prob.cpu().detach().numpy())
                losses.update(loss.item(), inputs.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                if True:  # not args.silent:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        (epoch + 1), (i + 1), len(dataloaders[phase]), batch_time=batch_time, data_time=data_time, loss=losses))

            # deep copy the model
            if phase == 'val' and losses.avg >= best_loss:
                best_acc = losses.avg
                best_model_wts = copy.deepcopy(model.state_dict())


