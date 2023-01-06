import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib

from models.encoder.import ECAPA_TDNN, TitaNet


class Raw_ECAPA(nn.Module):
    """
    Refactored RawNet2 combined with ECAPA architecture.
    Reference:
    [1] RawNet2 : https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1011.pdf
    [2] ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in
    TDNN Based Speaker Verification" (https://arxiv.org/abs/2005.07143)
    """

    def __init__(self, nOut=512, **kwargs):
        super(Raw_ECAPA, self).__init__()
        # SiLU, ReLU, GELU
        self.ECAPA_TDNN = ECAPA_TDNN.MainModel(nOut=192,
                                               activation=torch.nn.GELU,
                                               channels=[
                                                   512, 512, 512, 512, 1536],
                                               input_norm=True, **kwargs)
        self.titaNet = TitaNet.MainModel(
            nOut=nOut-192, model_size='m', n_mega_blocks=None, **kwargs)
        # self.rawnet2v2 = RawNet2v2.MainModel(nOut=nOut-192,**kwargs)
        features = 'melspectrogram'
        Features_extractor = importlib.import_module(
            'models.encoder.FeatureExtraction.feature').__getattribute__(f"{features}")
        self.compute_features = Features_extractor(**kwargs)

    def forward(self, x):
        #####

        # #####
        # # forward model 1
        # #####
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x_spec = self.compute_features(x)
        out1 = self.ECAPA_TDNN(x_spec)
        # #####
        # # forward model 2
        # #####
        out2 = self.titaNet(x_spec)
        #
        out = torch.cat([out1, out2], dim=-1)
#         out = torch.mean(out, dim=-1)
        return out


def MainModel(nOut=512, **kwargs):
    model = Raw_ECAPA(nOut=nOut, **kwargs)
    return model


if __name__ == "__main__":
    from torchsummary import summary

    model = MainModel()
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("nb_params:{}".format(nb_params))

    summary(model, (16240,), batch_size=2)
