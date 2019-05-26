from __future__ import print_function, division

import pandas as pd
from eeglibrary.src import train
from eeglibrary.src import Metric
from eeglibrary.utils import train_args, add_adda_args
from utils import class_names, subject_dir_names, arrange_paths


def voting(args, pred_list, path_list):
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
    sub_df = pd.read_csv(args.sub_path, engine='python')
    thresh = args.thresh  # 1の割合がthreshを超えたら1と判断
    pred_df = ensemble_preds(pred_list, path_list, sub_df, thresh)
    sub_df.loc[pred_df.index, 'preictal'] = pred_df['preictal']
    sub_df.to_csv(args.sub_path, index=False)


def label_func(path):
    return path.split('/')[-2].split('_')[2]


if __name__ == '__main__':
    args = add_adda_args(train_args()).parse_args()

    metrics = [
        Metric('loss', initial_value=10000, inequality='less', save_model=True),
        Metric('accuracy', initial_value=0, inequality='more'),
        Metric('far', initial_value=1000, inequality='less')]

    if args.train_manifest == 'all':
        for sub_name in subject_dir_names:
            args = arrange_paths(args, sub_name)
            train(args, class_names, label_func, metrics)
    elif args.inference:
        pred_list, path_list = train(args, class_names, label_func, metrics)
        voting(args, pred_list, path_list)
    else:
        train(args, class_names, label_func, metrics)

