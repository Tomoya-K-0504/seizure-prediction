from __future__ import print_function, division

import pandas as pd
from args import baseline_args
from eeglibrary import EEGDataSet, EEGDataLoader
from eeglibrary.models.RNN import *
from tqdm import tqdm
from utils import AverageMeter, init_seed, set_eeg_conf, init_device, set_model, set_dataloader
from test import test
from eeglibrary.models.toolbox import model_baseline
from sklearn.metrics import log_loss


def calc_criterions(y_pred, y_true):
    metric_names = ['log_loss', 'recall', 'far']
    results = {key: 0 for key in metric_names}

    results['log_loss'] = log_loss(y_true, y_pred)
    results['recall'] = recall_rate(preds, labels).item()
    results['far'] = false_detection_rate(preds, labels).item()

    return results


if __name__ == '__main__':
    args = baseline_args().parse_args()
    init_seed(args)

    eeg_conf = set_eeg_conf(args)
    dataloaders = {phase: set_dataloader(args, eeg_conf, phase, device='cpu') for phase in ['train', 'val', 'test']}
    models = None

    for phase in ['train', 'val']:
        for i, (inputs, labels) in tqdm(enumerate(dataloaders[phase]), total=len(dataloaders[phase])):
            models, preds = model_baseline(models, inputs, labels)

    if args.test:
        # test phase
        test(args, device='cpu')
