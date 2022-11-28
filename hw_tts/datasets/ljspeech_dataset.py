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

logger = logging.getLogger(__name__)

class LJspeechDataset(Dataset):
    def __init__(self, config_parser: ConfigParser):
        self.data_dir = ROOT_PATH / "data" / "LJSpeech-1.1"
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

    def get_data_to_buffer(self):
        buffer = list()
        text = self.process_text('./data/train.txt')

        start = time.perf_counter()
        print("Loading data to the buffer")

        for i in tqdm(range(len(text))):

            mel_gt_name = os.path.join(
                "./data/mels", "ljspeech-mel-%05d.npy" % (i+1))
            mel_gt_target = np.load(mel_gt_name)
            duration = np.load(os.path.join(
                "./data/alignments", str(i)+".npy"))
            character = text[i][0:len(text[i])-1]
            character = np.array(
                text_to_sequence(character, ['english_cleaners']))
            energy = np.load(os.path.join(
                "./data/energies", str(i)+".npy"))
            energy = energy.squeeze(0)
            pitch = np.load(os.path.join(
                "./data/pitches", str(i)+".npy"))

            character = torch.from_numpy(character)
            duration = torch.from_numpy(duration)
            mel_gt_target = torch.from_numpy(mel_gt_target)
            energy = torch.from_numpy(energy)
            pitch = torch.from_numpy(pitch)

            buffer.append({"src_seq": character, "duration_predictor_target": duration,
                        "mel_target": mel_gt_target, "energy": energy, "pitch": pitch})

        end = time.perf_counter()
        print("cost {:.2f}s to load all data into buffer.".format(end-start))

        return buffer