from torch import nn
import torch

from hw_tts.base import BaseModel
from hw_tts.model.duration_predictor import LengthRegulator
from hw_tts.model.fftblock import FFTBlock
from hw_tts.model.energy_predictor import Energy
from hw_tts.model.pitch_predictor import Pitch


def get_non_pad_mask(model_config, seq):
    assert seq.dim() == 2
    return seq.ne(model_config['PAD']).type(torch.float).unsqueeze(-1)

def get_attn_key_pad_mask(model_config, seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''
    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(model_config['PAD'])
    padding_mask = padding_mask.unsqueeze(1)
    padding_mask = padding_mask.expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


class Encoder(nn.Module):
    def __init__(self, model_config):
        super(Encoder, self).__init__()
        
        self.model_config = model_config
        len_max_seq=model_config['max_seq_len']
        n_position = len_max_seq + 1
        n_layers = model_config['encoder_n_layer']

        self.src_word_emb = nn.Embedding(
            model_config['vocab_size'],
            model_config['encoder_dim'],
            padding_idx=model_config['PAD']
        )

        self.position_enc = nn.Embedding(
            n_position,
            model_config['encoder_dim'],
            padding_idx=model_config['PAD']
        )

        self.layer_stack = nn.ModuleList([FFTBlock(
            model_config['encoder_dim'],
            model_config['encoder_conv1d_filter_size'],
            model_config['encoder_head'],
            model_config['encoder_dim'] // model_config['encoder_head'],
            model_config['encoder_dim'] // model_config['encoder_head'],
            dropout=model_config['dropout']
        ) for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(self.model_config, seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(self.model_config, src_seq)
        
        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
        

        return enc_output, non_pad_mask


class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, model_config):

        super(Decoder, self).__init__()

        self.model_config = model_config
        len_max_seq=model_config['max_seq_len']
        n_position = len_max_seq + 1
        n_layers = model_config['decoder_n_layer']

        self.position_enc = nn.Embedding(
            n_position,
            model_config['encoder_dim'],
            padding_idx=model_config['PAD'],
        )

        self.layer_stack = nn.ModuleList([FFTBlock(
            model_config['encoder_dim'],
            model_config['encoder_conv1d_filter_size'],
            model_config['encoder_head'],
            model_config['encoder_dim'] // model_config['encoder_head'],
            model_config['encoder_dim'] // model_config['encoder_head'],
            dropout=model_config['dropout']
        ) for _ in range(n_layers)])

    def forward(self, enc_seq, enc_pos, energy_emb, pitch_emb, return_attns=False):

        dec_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(self.model_config, seq_k=enc_pos, seq_q=enc_pos)
        non_pad_mask = get_non_pad_mask(self.model_config, enc_pos)

        # -- Forward
        dec_output = enc_seq + self.position_enc(enc_pos) + energy_emb + pitch_emb

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output


def get_mask_from_lengths(lengths, max_len=None):
    if max_len == None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len, 1, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool()

    return mask


class FastSpeech2(BaseModel):
    """ FastSpeech """

    def __init__(self, model_config):
        super(FastSpeech2, self).__init__()

        self.encoder = Encoder(model_config)
        self.length_regulator = LengthRegulator(model_config)
        self.energy_predictor = Energy(model_config)
        self.pitch_predictor = Pitch(model_config)
        self.decoder = Decoder(model_config)

        self.mel_linear = nn.Linear(model_config['decoder_dim'], model_config['num_mels'])

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(self, src_seq, src_pos, mel_pos=None, mel_max_len=None, duration=None, energy=None, pitch=None, alpha=1.0, alpha_e=1, alpha_p=1, **batch):
        x, non_pad_mask = self.encoder(src_seq, src_pos)
        if self.training:
            output, duration_predictor_output, alignment = self.length_regulator(x, alpha, duration, mel_max_len)
            energy_embedding, energy_predictor_output = self.energy_predictor(output, alpha_e=alpha_e, target=energy)
            pitch_embedding, pitch_predictor_output = self.pitch_predictor(output, alpha_p=alpha_p, target=pitch)
            output = self.decoder(output, mel_pos, energy_embedding, pitch_embedding)
            output = self.mask_tensor(output, mel_pos, mel_max_len)
            output = self.mel_linear(output)
            return output, duration_predictor_output, energy_predictor_output, pitch_predictor_output
        else:
            output, mel_pos, alignment = self.length_regulator(x, alpha)
            energy_embedding, _ = self.energy_predictor(output, alpha_e=alpha_e)
            pitch_embedding, _ = self.pitch_predictor(output, alpha_p=alpha_p)
            output = self.decoder(output, mel_pos, energy_embedding, pitch_embedding)
            output = self.mel_linear(output)
            return output
