import argparse
import subprocess
import copy
from tqdm import tqdm


def preprocess_args():
    parser = argparse.ArgumentParser(description='preprocess arguments')
    parser.add_argument('--out-dir', metavar='DIR',
                        help='directory to save splitted data', default='input/splitted')
    parser.add_argument('--patients-dir', metavar='DIR',
                        help='directory where patients data placed', default='input/splitted')
    parser.add_argument('--duration', type=str,
                        help='duration of one splitted wave', default=1.0)

    return parser.parse_args()


def search_args():
    parser = argparse.ArgumentParser(description='training arguments')
    parser.add_argument('--sub-path', default='../output/sth.csv', type=str, help='submission file save folder name')
    parser.add_argument('--model-path', metavar='DIR', help='directory to save models', default='../model/sth.pth')
    parser.add_argument('--train-manifest', type=str, help='manifest file for training', default='input/train_manifest.csv')
    parser.add_argument('--val-manifest', type=str, help='manifest file for validation', default='input/val_manifest.csv')
    parser.add_argument('--test-manifest', type=str, help='manifest file for test', default='input/test_manifest.csv')
    parser.add_argument('--log-dir', type=str, help='tensorboard log dir', default='../log/tensorboard/')
    parser.add_argument('--epochs', default=30, type=int, help='Number of training epochs')
    parser.add_argument('--gpu-id', default=0, type=int, help='ID of GPU to use')

    parser.add_argument('--model-name', default='cnn_16_751_751', type=str, help='network model name')
    parser.add_argument('--batch-size', default='32', type=str, help='Batch size for training')
    parser.add_argument('--epoch-rate', default='1.0', type=str, help='Data rate to to use in one epoch')
    parser.add_argument('--window-size', default='1.0', type=str, help='Window size for spectrogram in seconds')
    parser.add_argument('--window-stride', default='0.05', type=str, help='Window stride for spectrogram in seconds')
    parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
    # parser.add_argument('--rnn-type', default='gru', help='Type of the RNN. rnn|gru|lstm are supported')
    parser.add_argument('--lr', '--learning-rate', default='3e-2', type=str, help='initial learning rate')
    parser.add_argument('--momentum', default='0.9', type=str, help='momentum')
    parser.add_argument('--learning-anneal', default='1.1', type=str,
                        help='Annealing applied to learning rate every epoch')
    parser.add_argument('--sample-rate', default='1500', type=str, help='Sample rate')

    return parser.parse_args()


if __name__ == '__main__':
    # TODO multiprocessing対応にして、CPU・GPUの余りに応じてプロセス増やす機能

    one_args = ['sub_path', 'model_path', 'train_manifest', 'val_manifest', 'test_manifest', 'log_dir', 'epochs', 'gpu_id']
    str_args = ['model_name', 'window', 'rnn_type']
    int_args = ['batch_size', 'sample_rate']

    args = search_args()
    args_dict = {}
    for arg, value in args.__dict__.items():
        if arg in one_args:
            args_dict[arg] = value
        elif arg in str_args:
            args_dict[arg] = value.split(',')
        elif arg in int_args:
            args_dict[arg] = list(map(int, value.split(',')))
        else:
            args_dict[arg] = list(map(float, value.split(',')))

    def make_params_gridlike(grid_params, arg, value_list):
        new_grid_params = []
        if grid_params:
            for value in value_list:
                for dic in grid_params:
                    dic[arg] = value
                    new_grid_params.append(copy.deepcopy(dic))
        else:
            dic = {}
            for value in value_list:
                dic[arg] = value
                new_grid_params.append(dic)
        return new_grid_params

    grid_params = []
    for arg, value in args_dict.items():
        if isinstance(value, list):
            grid_params = make_params_gridlike(grid_params, arg, value)
    base_exec_str = 'python /home/tomoya/workspace/kaggle/seizure-prediction/src/train.py --tensorboard --cuda '
    base_exec_str += ' '.join(['--{} {}'.format(one_arg.replace('_', '-'), args_dict[one_arg]) for one_arg in one_args])
    for params_dict in tqdm(grid_params):
        exec_str = base_exec_str + ' ' + ' '.join(['--{} {}'.format(arg.replace('_', '-'), value) for arg, value in params_dict.items()])
        id_name = '_'.join(list(map(str, params_dict.values())))
        exec_str += ' --id {}'.format(id_name)
        _ = subprocess.check_output(exec_str, shell=True)
