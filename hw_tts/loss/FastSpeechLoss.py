import torch
import torch.nn as nn


class FastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, mel, duration_predicted, energy_predicted, energy, pitch_predicted, pitch, mel_target, duration, **batch):
        mel_loss = self.mse_loss(mel, mel_target)

        duration_predictor_loss = self.l1_loss(duration_predicted,
                                               duration.float())

        energy_predictor_loss = self.mse_loss(energy_predicted, energy)

        pitch_predictor_loss = self.mse_loss(pitch_predicted, pitch)

        return mel_loss, duration_predictor_loss, energy_predictor_loss, pitch_predictor_loss
