from __future__ import print_function, division

import pandas as pd
from eeglibrary.src import test
from eeglibrary.utils import test_args

from src.utils import arrange_paths, class_names, subject_dir_names


def print_preds(sub_df):
    for subject in subject_dir_names:
        subject_df = sub_df.loc[sub_df['clip'].apply(lambda x: subject in x), 'preictal']
        print(subject, '\n', subject_df.value_counts(normalize=True))


if __name__ == '__main__':
    args = test_args().parse_args()

    if args.only_results:
        sub_df = pd.read_csv(args.sub_path, engine='python')
        print_preds(sub_df)
        exit()

    if args.test_manifest == 'all':
        for sub_name in subject_dir_names:
            args = arrange_paths(args, sub_name)
            test(args)
    else:
        test(args)
