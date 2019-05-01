from __future__ import print_function, division

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
import random
random.seed(seed)
from torch.utils.data.sampler import WeightedRandomSampler

from eeglibrary import EEGDataSet, EEGDataLoader, make_weights_for_balanced_classes, EEG
from eeglibrary import TensorBoardLogger
from eeglibrary.models.CNN import *
from eeglibrary.models.RNN import *
from args import train_args
from test import test
from utils import AverageMeter, init_seed, set_eeg_conf, init_device, set_model
from eeglibrary import recall_rate, false_detection_rate


def set_dataloaders(args, eeg_conf, device='cpu'):
    manifests = [args.train_manifest, args.val_manifest]

    dataloaders = {}
    for part, manifest in zip(['train', 'val'], manifests):
        dataset = EEGDataSet(manifest, eeg_conf, class_names, device=device)
        weights = make_weights_for_balanced_classes(dataset.labels_index(), len(class_names))
        sampler = WeightedRandomSampler(weights, int(len(dataset) * args.epoch_rate))
        dataloaders[part] = EEGDataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                          pin_memory=True, sampler=sampler, drop_last=True)
    return dataloaders


def train_all():
    subject_dir_names = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
    pass


if __name__ == '__main__':

    class_names = ['interictal', 'preictal']

    args = train_args().parse_args()
    init_seed(args)
    Path(args.model_path).parent.mkdir(exist_ok=True)

    if args.tensorboard:
        tensorboard_logger = TensorBoardLogger(args.log_id, args.log_dir, args.log_params)

    start_epoch, start_iter, optim_state = 0, 0, None
    # far; False alarm rate = 1 - specificity
    best_loss, best_far, losses, far, recall = {}, {}, {}, {}, {}
    for phase in ['train', 'val']:
        best_loss[phase], best_far[phase] = 1000, 1.0
        losses[phase], recall[phase], far[phase] = (AverageMeter() for i in range(3))

    # init setting
    device = init_device(args)
    eeg_conf = set_eeg_conf(args)
    model = set_model(args, eeg_conf, device, class_names)
    dataloaders = set_dataloaders(args, eeg_conf, device)

    parameters = model.parameters()
    optimizer = torch.optim.SGD(parameters, lr=args.lr)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, args.pos_loss_weight]).to(device))
    batch_time = AverageMeter()
    execute_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        start_epoch_time = time.time()

        for phase in ['train', 'val']:
            print('\n{} phase started.'.format(phase))

            epoch_preds = torch.empty((len(dataloaders[phase])*args.batch_size, 1), dtype=torch.int64, device=device)
            epoch_labels = torch.empty((len(dataloaders[phase])*args.batch_size, 1), dtype=torch.int64, device=device)

            start_time = time.time()
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs, labels = inputs.to(device), labels.to(device)
                break
                data_load_time = time.time() - start_time
                # print('data loading time', data_load_time)

                optimizer.zero_grad()

                # feature scaling
                if args.scaling:
                    inputs = (inputs - 100).div(600)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # print('forward calculation time', time.time() - (data_load_time + start_time))
                    _, preds = torch.max(outputs, 1)
                    epoch_preds[i*args.batch_size:(i+1)*args.batch_size, 0] = preds
                    epoch_labels[i*args.batch_size:(i+1)*args.batch_size, 0] = labels
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # save loss and recall in one batch
                losses[phase].update(loss.item() / inputs.size(0), inputs.size(0))
                recall[phase].update(recall_rate(labels, preds).item())
                far[phase].update(false_detection_rate(labels, preds).item())

                # measure elapsed time
                batch_time.update(time.time() - start_time)

                if not args.silent:
                    print('Epoch: [{0}][{1}/{2}] \tTime {batch_time.val:.3f} \t'
                          'recall {recall.val:.3f} far {far.val:.3f} '
                          '\tLoss {loss.val:.4f} ({loss.avg:.4f}) \t'.format(
                        epoch, (i + 1), len(dataloaders[phase]), batch_time=batch_time,
                        recall=recall[phase], far=far[phase], loss=losses[phase]))

                start_time = time.time()

            if losses[phase].avg < best_loss[phase]:
                best_loss[phase] = losses[phase].avg
                if phase == 'val':
                    print("Found better validated model, saving to %s" % args.model_path)
                    torch.save(model.state_dict(), args.model_path)

            if far[phase].avg < best_far[phase]:
                best_far[phase] = far[phase].avg

            if args.tensorboard:
                if args.log_params:
                    raise NotImplementedError
                values = {
                    phase + '_loss': losses[phase].avg,
                    phase + '_recall': recall[phase].avg,
                    phase + '_far': far[phase].avg,
                }
                tensorboard_logger.update(epoch, values, model.named_parameters())

            # anneal lr
            if phase == 'train':
                param_groups = optimizer.param_groups
                for g in param_groups:
                    g['lr'] = g['lr'] / args.learning_anneal
                print('Learning rate annealed to: {lr:.6f}'.format(lr=g['lr']))

            losses[phase].reset()
            recall[phase].reset()
            recall[phase].reset()

    print('execution time was {}'.format(time.time() - execute_time))

    if args.test:
        # test phase
        test(args, device, class_names)
