from __future__ import print_function, division

from eeglibrary.src import train
from eeglibrary.utils import train_args
from utils import class_names, subject_dir_names, arrange_paths


def label_func(path):
    return path.split('/')[-2].split('_')[2]


if __name__ == '__main__':
    args = train_args().parse_args()

    if args.train_manifest == 'all':
        for sub_name in subject_dir_names:
            args = arrange_paths(args, sub_name)
            train(args, class_names, label_func)
    else:
        train(args, class_names, label_func)