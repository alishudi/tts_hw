import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from hw_tts.utils import ROOT_PATH

class Transpose(nn.Module):
    def __init__(self, dim_1, dim_2):
        super().__init__()
        self.dim_1 = dim_1
        self.dim_2 = dim_2

    def forward(self, x):
        return x.transpose(self.dim_1, self.dim_2)

class EnergyPredictor(nn.Module):
    """ Energy Predictor """

    def __init__(self, model_config):
        super(EnergyPredictor, self).__init__()

        self.input_size = model_config['encoder_dim']
        self.filter_size = model_config['duration_predictor_filter_size']
        self.kernel = model_config['duration_predictor_kernel_size']
        self.conv_output_size = model_config['duration_predictor_filter_size']
        self.dropout = model_config['variance_dropout']

        self.conv_net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(
                self.input_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            Transpose(-1, -2),
            nn.Conv1d(
                self.filter_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)
        self.relu = nn.ReLU()

    def forward(self, encoder_output):
        encoder_output = self.conv_net(encoder_output)
            
        out = self.linear_layer(encoder_output)
        out = self.relu(out)
        out = out.squeeze()
        if not self.training:
            out = out.unsqueeze(0)
        return out


class Energy(nn.Module):
    def __init__(self, model_config):
        super(Energy, self).__init__()
        self.embedding = nn.Embedding(model_config['duration_predictor_filter_size'], model_config['encoder_dim'])
        self.energy_predictor = EnergyPredictor(model_config)
        energy_min = np.load(ROOT_PATH / "small data" / "energy_min.npy")
        energy_max = np.load(ROOT_PATH / "small data" / "energy_max.npy")
        self.bins = torch.Tensor(np.linspace(energy_min - 1e-3, energy_max + 1e-3, num = 256))

    def forward(self, x, alpha_e=1.0, target=None):
        energy_predictor_output = self.energy_predictor(x)
        if target is not None:
            energy_quantized = torch.bucketize(target, self.bins.to(x.device))
            energy_embedding = self.embedding(energy_quantized)
            return energy_embedding, energy_predictor_output
        else:
            energy_predictor_output = ((energy_predictor_output + 0.5) * alpha_e).int()
            energy_quantized = torch.bucketize(energy_predictor_output, self.bins.to(x.device))
            energy_embedding = self.embedding(energy_quantized)
            return energy_embedding, energy_predictor_output
