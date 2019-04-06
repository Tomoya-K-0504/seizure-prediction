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
import sklearn.metrics as metrics
import random
random.seed(seed)
from torch.utils.data.sampler import WeightedRandomSampler
import torch.optim as optim

from eeglibrary import EEGDataSet, EEGDataLoader, make_weights_for_balanced_classes, EEG
from eeglibrary import TensorBoardLogger
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

    print(model)

    return model


def set_eeg_conf(args):
    one_eeg_path = pd.read_csv(args.train_manifest).values[0][0]
    n_elect = len(EEG.load_pkl(one_eeg_path).channel_list)
    eeg_conf = dict(spect=args.spect,
                    n_elect=n_elect,
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
            sampler = WeightedRandomSampler(weights, int(len(dataset)*args.epoch_rate))
            dataloaders[part] = EEGDataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              pin_memory=True, sampler=sampler)
        else:
            dataset = EEGDataSet(manifest, eeg_conf, return_path=True)
            dataloaders[part] = EEGDataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              pin_memory=True, shuffle=False)
    return dataloaders


if __name__ == '__main__':

    args = train_args()
    init_seed(args)

    device = torch.device("cuda" if args.cuda else "cpu")
    if args.cuda:
        torch.cuda.set_device(args.gpu_id)

    Path(args.model_path).parent.mkdir(exist_ok=True)

    if args.tensorboard:
        tensorboard_logger = TensorBoardLogger(args.id, args.log_dir, args.log_params)

    start_epoch, start_iter, optim_state = 0, 0, None
    best_loss, best_auc, losses, aucs, recall_0, recall_1 = {}, {}, {}, {}, {}, {}
    for phase in ['train', 'val']:
        best_loss[phase], best_auc[phase] = 1000, 0
        losses[phase], recall_0[phase], recall_1[phase], aucs[phase] = (AverageMeter() for i in range(4))
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
        start_epoch_time = time.time()

        for phase in ['train', 'val']:
            print('\n{} phase started.'.format(phase))

            epoch_preds = []
            epoch_labels = []

            start_time = time.time()
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                data_load_time = time.time() - start_time
                # print('data loading time', data_load_time)
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # print('forward calculation time', time.time() - (data_load_time + start_time))
                    _, preds = torch.max(outputs, 1)
                    epoch_preds.extend(preds.cpu().numpy())
                    epoch_labels.extend(labels.cpu().numpy())
                    pred_prob = outputs[:, 1].float()
                    loss = criterion(pred_prob, labels.float())

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # save loss and recall in one batch
                losses[phase].update(loss.item() / inputs.size(0), inputs.size(0))
                _, recall, _, _ = metrics.precision_recall_fscore_support(labels.cpu(), preds.cpu())
                if len(recall) == 2:
                    recall_0[phase].update(recall[0])
                    recall_1[phase].update(recall[1])
                else:
                    recall_0[phase].update(recall[0]) if not labels.sum() else recall_1[phase].update(recall[0])

                # measure elapsed time
                batch_time.update(time.time() - start_time)

                if not args.silent:
                    print('Epoch: [{0}][{1}/{2}] \tTime {batch_time.val:.3f} ({batch_time.avg:.3f}) \t'
                          'rec_0 {rec_0.val:.3f} rec_1 {rec_1.val:.3f} \tLoss {loss.val:.4f} ({loss.avg:.4f}) \t'.format(
                        epoch, (i + 1), len(dataloaders[phase]), batch_time=batch_time,
                        rec_0=recall_0[phase], rec_1=recall_1[phase], loss=losses[phase]))

                start_time = time.time()

            aucs[phase].update(metrics.roc_auc_score(epoch_labels, epoch_preds))

            if losses[phase].avg < best_loss[phase]:
                best_loss[phase] = losses[phase].avg

            if aucs[phase].avg > best_auc[phase]:
                best_auc[phase] = aucs[phase].avg
                if phase == 'val':
                    print("Found better validated model, saving to %s" % args.model_path)
                    torch.save(model.state_dict(), args.model_path)

            if args.tensorboard:
                if args.log_params:
                    raise NotImplementedError
                values = {
                    phase + 'loss': losses[phase].avg,
                    phase + 'rec_0': recall_0[phase].avg,
                    phase + 'rec_1': recall_1[phase].avg,
                    phase + 'auc': aucs[phase].avg,
                }
                tensorboard_logger.update(epoch, values, model.named_parameters())

            # anneal lr
            param_groups = optimizer.param_groups
            for g in param_groups:
                g['lr'] = g['lr'] / args.learning_anneal
            print('Learning rate annealed to: {lr:.6f}'.format(lr=g['lr']))

            losses[phase].reset()
            recall_0[phase].reset()

    if args.test:
        # test phase
        model.eval()
        pred_list = []
        path_list = []
        for i, (inputs, paths) in tqdm(enumerate(dataloaders['test']), total=len(dataloaders['test'])):
            inputs = inputs.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            pred_list.extend(preds)
            path_list.extend(paths)
            # break

        def ensemble_preds(pred_list, path_list, sub_df, thresh):
            # もともとのmatファイルごとに振り分け直す
            patient_name = path_list[0].split('/')[-3]
            orig_mat_list = sub_df[sub_df['clip'].apply(lambda x: '_'.join(x.split('_')[:2])) == patient_name]
            ensembled_pred_list = []
            for orig_mat_name in orig_mat_list['clip']:
                seg_number = int(orig_mat_name[-8:-4])
                one_segment_preds = [pred for path, pred in zip(path_list, pred_list) if
                                     int(path.split('/')[-2].split('_')[-1]) == seg_number]
                ensembled_pred = int(sum(one_segment_preds) >= len(one_segment_preds) * thresh)
                ensembled_pred_list.append(ensembled_pred)
            orig_mat_list['preictal'] = ensembled_pred_list
            return orig_mat_list

        # preds to csv
        # sub_df = pd.read_csv('../output/sampleSubmission.csv')
        sub_df = pd.read_csv(args.sub_path, engine='python')
        thresh = args.thresh    # 1の割合がthreshを超えたら1と判断
        pred_df = ensemble_preds(pred_list, path_list, sub_df, thresh)
        sub_df.loc[pred_df.index, 'preictal'] = pred_df['preictal']
        sub_df.to_csv(args.sub_path, index=False)

        for subject in subject_dir_names:
            subject_df = sub_df.loc[sub_df['clip'].apply(lambda x: subject in x), 'preictal']
            print(subject, '\n', subject_df.value_counts(normalize=True))
