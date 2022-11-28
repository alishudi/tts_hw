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

class PitchPredictor(nn.Module):
    """ Pitch Predictor """

    def __init__(self, model_config):
        super(PitchPredictor, self).__init__()

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


class Pitch(nn.Module):
    def __init__(self, model_config):
        super(Pitch, self).__init__()
        self.embedding = nn.Embedding(model_config['duration_predictor_filter_size'], model_config['encoder_dim'])
        self.pitch_predictor = PitchPredictor(model_config)
        pitch_min = np.load(ROOT_PATH / "small data" / "pitch_min.npy")
        pitch_max = np.load(ROOT_PATH / "small data" / "pitch_max.npy")
        self.bins = torch.Tensor(np.linspace(pitch_min - 1e-3, pitch_max + 1e-3, num = 256)) # pitches are already in log scale
 
    def forward(self, x, alpha_p=1.0, target=None):
        pitch_predictor_output = self.pitch_predictor(x)
        if target is not None:
            pitch_quantized = torch.bucketize(target, self.bins.to(x.device))
            pitch_embedding = self.embedding(pitch_quantized)
            return pitch_embedding, pitch_predictor_output
        else:
            pitch_predictor_output = ((pitch_predictor_output + 0.5) * alpha_p).int()
            pitch_quantized = torch.bucketize(pitch_predictor_output, self.bins.to(x.device))
            pitch_embedding = self.embedding(pitch_quantized)
            return pitch_embedding, pitch_predictor_output
