import argparse


def preprocess_args():
    parser = argparse.ArgumentParser(description='preprocess arguments')
    parser.add_argument('--out-dir', metavar='DIR',
                        help='directory to save splitted data', default='input/splitted')
    parser.add_argument('--patients-dir', metavar='DIR',
                        help='directory where patients data placed', default='input/splitted')
    parser.add_argument('--duration', type=float,
                        help='duration of one splitted wave', default=1.0)

    return parser.parse_args()


def train_args():
    parser = argparse.ArgumentParser(description='training arguments')
    parser.add_argument('--model-name', default='cnn_16_751_751', type=str, help='network model name')
    parser.add_argument('--sub-path', default='../output/', type=str, help='submission file save folder name')
    parser.add_argument('--seed', default=0, type=int, help='Seed to generators')
    parser.add_argument('--thresh', default=0.5, type=float, help='Threshold in ensemble')
    parser.add_argument('--model-path', metavar='DIR', help='directory to save models', default='../model/')
    parser.add_argument('--train-manifest', type=str, help='manifest file for training', default='input/train_manifest.csv')
    parser.add_argument('--val-manifest', type=str, help='manifest file for validation', default='input/val_manifest.csv')
    parser.add_argument('--test-manifest', type=str, help='manifest file for test', default='input/test_manifest.csv')
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size for training')
    parser.add_argument('--epoch-rate', default=1.0, type=float, help='Data rate to to use in one epoch')
    parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in data-loading')
    parser.add_argument('--duration', default=1.0, type=float, help='Duration of one EEG dataset')
    parser.add_argument('--window-size', default=4.0, type=float, help='Window size for spectrogram in seconds')
    parser.add_argument('--window-stride', default=2.0, type=float, help='Window stride for spectrogram in seconds')
    parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
    parser.add_argument('--hidden-size', default=800, type=int, help='Hidden size of RNNs')
    parser.add_argument('--hidden-layers', default=5, type=int, help='Number of RNN layers')
    parser.add_argument('--rnn-type', default='gru', help='Type of the RNN. rnn|gru|lstm are supported')
    parser.add_argument('--pos-loss-weight', default=1.0, type=float, help='The weights of positive class loss')
    parser.add_argument('--epochs', default=70, type=int, help='Number of training epochs')
    parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
    parser.add_argument('--lr', '--learning-rate', default=3e-2, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--max-norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
    parser.add_argument('--learning-anneal', default=1.1, type=float,
                        help='Annealing applied to learning rate every epoch')
    parser.add_argument('--silent', dest='silent', action='store_true', help='Turn off progress tracking per iteration')
    parser.add_argument('--checkpoint', dest='checkpoint', action='store_true',
                        help='Enables checkpoint saving of model')
    parser.add_argument('--checkpoint-per-batch', default=0, type=int,
                        help='Save checkpoint per batch. 0 means never save')
    parser.add_argument('--augment', dest='augment', action='store_true',
                        help='Use random tempo and gain perturbations.')
    parser.add_argument('--no-bidirectional', dest='bidirectional', action='store_false', default=True,
                        help='Turn off bi-directional RNNs, introduces lookahead convolution')
    parser.add_argument('--spect', dest='spect', action='store_true', help='Use spectrogram as input')
    parser.add_argument('--sample-rate', default=1500, type=int, help='Sample rate')

    parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='Turn on tensorboard graphing')
    parser.add_argument('--log-dir', default='visualize/', help='Location of tensorboard log')
    parser.add_argument('--log-params', dest='log_params', action='store_true',
                        help='Log parameter values and gradients')
    parser.add_argument('--id', default='Seizure prediction training', help='Identifier for tensorboard run')
    parser.add_argument('--gpu-id', default=0, type=int, help='ID of GPU to use')
    parser.add_argument('--test', dest='test', action='store_true', help='Test phase after training or not')

    # parser.add_argument('--visdom', dest='visdom', action='store_true', help='Turn on visdom graphing')
    # parser.add_argument('--continue-from', default='', help='Continue from checkpoint model')
    # parser.add_argument('--finetune', dest='finetune', action='store_true',
    #                     help='Finetune the model from checkpoint "continue_from"')
    # parser.add_argument('--noise-dir', default=None,
    #                     help='Directory to inject noise into audio. If default, noise Inject not added')
    # parser.add_argument('--noise-prob', default=0.4, help='Probability of noise being added per sample')
    # parser.add_argument('--noise-min', default=0.0,
    #                     help='Minimum noise level to sample from. (1.0 means all noise, not original signal)',
    #                     type=float)
    # parser.add_argument('--noise-max', default=0.5,
    #                     help='Maximum noise levels to sample from. Maximum 1.0', type=float)
    # parser.add_argument('--no-shuffle', dest='no_shuffle', action='store_true',
    #                     help='Turn off shuffling and sample from dataset based on sequence length (smallest to largest)')

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
    parser.add_argument('--rnn-type', default='gru', help='Type of the RNN. rnn|gru|lstm are supported')
    parser.add_argument('--lr', '--learning-rate', default='3e-2', type=str, help='initial learning rate')
    parser.add_argument('--momentum', default='0.9', type=str, help='momentum')
    parser.add_argument('--learning-anneal', default='1.1', type=str,
                        help='Annealing applied to learning rate every epoch')
    parser.add_argument('--sample-rate', default='1500', type=str, help='Sample rate')
    parser.add_argument('--hidden-layers', default='2', type=str, help='Number of RNN layers')
    parser.add_argument('--hidden-size', default='400', type=str, help='Hidden size of RNNs')
    parser.add_argument('--pos-loss-weight', default='1.0', type=str, help='The weights of positive class loss')
    parser.add_argument('--duration', default='1.0', type=str, help='Duration of one EEG dataset')

    return parser.parse_args()
