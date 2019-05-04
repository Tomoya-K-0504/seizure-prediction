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
from sklearn.metrics import log_loss
from eeglibrary import EEGDataSet, EEGDataLoader, make_weights_for_balanced_classes, EEG
from eeglibrary import TensorBoardLogger
from eeglibrary.models.CNN import *
from eeglibrary.models.RNN import *
from args import train_args
from test import test
from utils import *
from eeglibrary import recall_rate, false_detection_rate


def train_model(model, inputs, labels, phase, optimizer, criterion, type='nn'):

    if type == 'nn':
        optimizer.zero_grad()
        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            # print('forward calculation time', time.time() - (data_load_time + start_time))
            loss = criterion(outputs, labels)

            if phase == 'train':
                loss.backward()
                optimizer.step()

            _, preds = torch.max(outputs, 1)
    else:
        inputs, labels = inputs.data.numpy(), labels.data.numpy()
        if phase == 'train':
            model.partial_fit(inputs, labels)
        preds = model.predict(inputs)
        loss = criterion(labels, preds, labels=[0, 1])  # logloss of skearn is reverse argment order compared with pytorch criterion

    return preds, loss


def save_model(model, model_path, numpy):
    if numpy:
        model.save_model(model_path)
    else:
        torch.save(model.state_dict(), model_path)


def train(args):
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
    model = set_model(args, eeg_conf, device)
    dataloaders = {phase: set_dataloader(args, eeg_conf, phase, device='cpu') for phase in ['train', 'val']}

    if 'nn' in args.model_name:
        parameters = model.parameters()
        optimizer = torch.optim.SGD(parameters, lr=args.lr)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, args.pos_loss_weight]).to(device))
        numpy = False
    else:
        optimizer = None
        criterion = log_loss
        numpy = True

    batch_time = AverageMeter()
    execute_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        start_epoch_time = time.time()

        for phase in ['train', 'val']:
            print('\n{} phase started.'.format(phase))

            epoch_preds = torch.empty((len(dataloaders[phase])*args.batch_size, 1), dtype=torch.int64, device=device)
            epoch_labels = torch.empty((len(dataloaders[phase])*args.batch_size, 1), dtype=torch.int64, device=device)

            if numpy:
                epoch_preds, epoch_labels = epoch_preds.data.numpy(), epoch_labels.data.numpy()

            start_time = time.time()
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs, labels = inputs.to(device), labels.to(device)
                # break
                data_load_time = time.time() - start_time
                # print('data loading time', data_load_time)

                # feature scaling
                if args.scaling:
                    inputs = (inputs - 100).div(600)

                preds, loss_value = train_model(model, inputs, labels, phase, optimizer, criterion, args.model_name)

                epoch_preds[i * args.batch_size:(i + 1) * args.batch_size, 0] = preds
                epoch_labels[i * args.batch_size:(i + 1) * args.batch_size, 0] = labels

                # save loss and recall in one batch
                losses[phase].update(loss_value / inputs.size(0), inputs.size(0))
                recall[phase].update(recall_rate(preds, labels, numpy))
                far[phase].update(false_detection_rate(preds, labels, numpy))

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
                    save_model(model, args.model_path, numpy)

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
                tensorboard_logger.update(epoch, values)

            # anneal lr
            if phase == 'train' and (not numpy):
                param_groups = optimizer.param_groups
                for g in param_groups:
                    g['lr'] = g['lr'] / args.learning_anneal
                print('Learning rate annealed to: {lr:.6f}'.format(lr=g['lr']))

            losses[phase].reset()
            recall[phase].reset()
            far[phase].reset()

    print('execution time was {}'.format(time.time() - execute_time))

    if args.test:
        # test phase
        test(args, numpy, device)


if __name__ == '__main__':
    args = train_args().parse_args()
    if args.train_manifest == 'all':
        for sub_name in subject_dir_names:
            args = arrange_paths(args, sub_name)
            train(args)
    else:
        train(args)