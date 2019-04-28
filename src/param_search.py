import argparse
import subprocess
import copy
from tqdm import tqdm
from pathlib import Path
from args import search_args


if __name__ == '__main__':
    # TODO multiprocessing対応にして、CPU・GPUの余りに応じてプロセス増やす機能

    one_args = ['sub_path', 'model_path', 'train_manifest', 'val_manifest', 'test_manifest', 'log_dir', 'epochs', 'gpu_id']
    str_args = ['model_name', 'window', 'rnn_type']
    int_args = ['batch_size', 'sample_rate', 'hidden_layers', 'hidden_size']

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
    base_exec_str = 'python {}/train.py --tensorboard --cuda '.format(Path(__file__).parent)
    base_exec_str += ' '.join(['--{} {}'.format(one_arg.replace('_', '-'), args_dict[one_arg]) for one_arg in one_args])
    for params_dict in tqdm(grid_params):
        exec_str = base_exec_str + ' ' + ' '.join(['--{} {}'.format(arg.replace('_', '-'), value) for arg, value in params_dict.items()])
        exec_str += ' --spect' if params_dict['model_name'] != 'cnn_1_16_399' else ''
        id_name = '_'.join(list(map(str, params_dict.values())))
        exec_str += ' --id {}'.format(id_name)
        _ = subprocess.check_output(exec_str, shell=True)
        # print(_)
