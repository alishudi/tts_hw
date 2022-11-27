import logging
import os

from hw_tts.utils import ROOT_PATH
from tqdm import tqdm

import torch
import numpy as np
import time
from text import text_to_sequence
from torch.utils.data import Dataset
from hw_tts.utils.parse_config import ConfigParser
from hw_tts.preprocessing import TacotronSTFT
import torchaudio

logger = logging.getLogger(__name__)

class LJspeechDataset(Dataset):
    def __init__(self, config_parser: ConfigParser):
        self.data_dir = ROOT_PATH / "data" / "LJSpeech-1.1"
        self.waw_paths = self.get_wav_paths()
        self.buffer = self.get_data_to_buffer()
        self.length_dataset = len(self.buffer)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        return self.buffer[idx]

    def process_text(self, train_text_path):
            with open(train_text_path, "r", encoding="utf-8") as f:
                txt = []
                for line in f.readlines():
                    txt.append(line)

                return txt

    def get_wav_paths(self):
        waw_paths = []
        csv_path = self.data_dir / 'metadata.csv'
        with csv_path.open() as f:
            for line in f:
                waw_id = line.split('|')[0]
                waw_paths.append(self.data_dir / 'wavs' / f"{waw_id}.wav")
        return waw_paths


    def get_data_to_buffer(self):
        buffer = list()
        text = self.process_text('./data/train.txt')

        start = time.perf_counter()
        print("Loading data to the buffer")

        energy_min, energy_max = np.inf, -np.inf

        for i in tqdm(range(len(text))):

            mel_gt_name = os.path.join(
                "./data/mels", "ljspeech-mel-%05d.npy" % (i+1))
            mel_gt_target = np.load(mel_gt_name)
            duration = np.load(os.path.join(
                "./data/alignments", str(i)+".npy"))
            character = text[i][0:len(text[i])-1]
            character = np.array(
                text_to_sequence(character, ['english_cleaners']))

            STFT = TacotronSTFT()
            audio_tensor, sr = torchaudio.load(self.waw_paths[i])
            audio_tensor = audio_tensor[0:1, :]
            energy = STFT.calc_energy(audio_tensor)
            if torch.min(energy) < energy_min:
                energy_min = torch.min(energy)
            if torch.max(energy) > energy_max:
                energy_max = torch.max(energy)

            character = torch.from_numpy(character)
            duration = torch.from_numpy(duration)
            mel_gt_target = torch.from_numpy(mel_gt_target)

            buffer.append({"src_seq": character, "duration_predictor_target": duration,
                        "mel_target": mel_gt_target, "energy": energy})

        end = time.perf_counter()
        print("cost {:.2f}s to load all data into buffer.".format(end-start))
        np.save(ROOT_PATH / "data" / "energy_min.npy", energy_min)
        np.save(ROOT_PATH / "data" / "energy_max.npy", energy_max)

        return buffer