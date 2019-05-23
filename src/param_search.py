import argparse
import subprocess
import copy
from tqdm import tqdm
from pathlib import Path
from eeglibrary.utils import search_args
import optuna

one_args = ['sub_path', 'model_path', 'train_manifest', 'val_manifest', 'test_manifest', 'log_dir', 'epochs', 'gpu_id']
str_args = ['model_name', 'window', 'rnn_type', 'loss_weight']
int_args = ['batch_size', 'sample_rate', 'rnn_n_layers', 'rnn_hidden_size']


def choose_args(trial, arg_name, arg_values):
    return trial.suggest_categorical(arg_name, arg_values)


def objective(trial=optuna.trial.Trial(optuna.create_study(direction='maximize'), 0)):
    # TODO multiprocessing対応にして、CPU・GPUの余りに応じてプロセス増やす機能
    args_dict = {}
    log_dir = 'log/'
    log_id = ''
    exec_str = '~/.conda/envs/brain/bin/python {}/train.py --tensorboard --cuda --silent '.format(Path(__file__).parent)
    args = parse_args()

    for arg_name, value in args.items():
        if isinstance(value, list):
            value = choose_args(trial, arg_name, value)
            log_dir = log_dir + arg_name + '/'
            log_id =  log_id + value + '/'
        exec_str += '--{} {} '.format(arg_name.replace('_', '-'), value)
        args_dict[arg_name] = value

    exec_str += '--spect ' if args_dict['model_name'] != '2dcnn_1' else ''
    exec_str += '--log-dir {} --log-id'.format(log_dir, log_id)
    res = subprocess.check_output(exec_str, shell=True)
    # print(res)
    return float(res.decode().split('\n')[-2])


def parse_args():

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
    return args_dict


if __name__ == '__main__':
    # objective()
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=1, n_jobs=1)
    print('Number of finished trials: ', len(study.trials))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: ', trial.value)

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    import pickle
    with open('study_2.pkl', 'wb') as f:
        pickle.dump(study, f)

    with open('study_2.pkl', 'rb') as f:
        study = pickle.load(f)
