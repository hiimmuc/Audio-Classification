import math
import argparse
import importlib
import torch.nn as nn
from torchsummary import summary
from utils import read_config


parser = argparse.ArgumentParser(description="Model plot")
if __name__ == '__main__':
    parser.add_argument('--config', type=str, default=None)
    sys_args = parser.parse_args()
    if sys_args.config is not None:
        args = read_config(sys_args.config, sys_args)
        args = argparse.Namespace(**args)

    model_options = args.model
    if isinstance(model_options['name'], str):
        SpeakerNetModel = importlib.import_module(
            'models.encoder.' + model_options['name']).__getattribute__('MainModel')
        model = SpeakerNetModel(
            nOut=model_options['nOut'], **vars(args)).to('cpu')
    else:
        del args.features
        model = importlib.import_module('models.encoder.' + 'Mixed_model').__getattribute__(
            'MainModel')(model_options=model_options, **vars(args)).to('cpu')

    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("nb_params:{}".format(nb_params))
    print("features:", str(args.features))
    if str(args.features) != 'raw':
        max_audio = int(args.audio_spec['sample_rate']
                        * args.audio_spec['sentence_len'])
        hop_frames = int(
            args.audio_spec['sample_rate'] * args.audio_spec['hop_len'])
        win_frames = int(
            args.audio_spec['sample_rate'] * args.audio_spec['win_len'])
        overlapped = win_frames - hop_frames
        max_frames = math.ceil((max_audio - overlapped) / hop_frames) + 2
        input_dim = (int(args.n_mels), max_frames)
    else:
        input_dim = (
            int(args.audio_spec['sample_rate'] * args.audio_spec['sentence_len']),)

    summary(model, input_dim, device='cpu', depth=3, col_width=16, col_names=[
            "input_size", "output_size", "num_params", "mult_adds"])
