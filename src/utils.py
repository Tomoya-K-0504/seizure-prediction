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
from args import train_args, add_test_args
from eeglibrary import recall_rate, false_detection_rate


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def init_seed(args):
    # Set seeds for determinism
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def init_device(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    if args.cuda:
        torch.cuda.set_device(args.gpu_id)
    return device


def set_eeg_conf(args):
    manifest_path = [value for key, value in vars(args).items() if 'manifest' in key][0]
    one_eeg_path = pd.read_csv(manifest_path).values[0][0]
    n_elect = len(EEG.load_pkl(one_eeg_path).channel_list)
    eeg_conf = dict(spect=args.spect,
                    n_elect=n_elect,
                    duration=args.duration,
                    window_size=args.window_size,
                    window_stride=args.window_stride,
                    window='hamming',
                    sample_rate=args.sample_rate)
    return eeg_conf


def set_model(args, eeg_conf, device, class_names):
    if args.model_name == 'cnn_1_16_399':
        model = cnn_1_16_399(eeg_conf, n_labels=len(class_names))
    elif args.model_name == 'cnn_16_751_751':
        model = cnn_16_751_751(eeg_conf, n_labels=len(class_names))
    elif args.model_name == 'rnn_16_751_751':
        cnn, out_ftrs = cnn_ftrs_16_751_751(eeg_conf)
        model = RNN(cnn, out_ftrs, args.batch_size, args.rnn_type, class_names, eeg_conf=eeg_conf)
    elif args.model_name == 'cnn_1_16_751_751':
        model = cnn_1_16_751_751(eeg_conf, n_labels=len(class_names))
    else:
        raise NotImplementedError

    model = model.to(device)
    print(model)

    return model
