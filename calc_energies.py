import torch
from hw_tts.preprocessing import TacotronSTFT
import torchaudio
import numpy as np
from hw_tts.utils import ROOT_PATH
from tqdm import tqdm
import os

DATA_DIR = ROOT_PATH / "data" / "LJSpeech-1.1"

def get_wav_paths():
        waw_paths = []
        csv_path = DATA_DIR / 'metadata.csv'
        with csv_path.open() as f:
            for line in f:
                waw_id = line.split('|')[0]
                waw_paths.append(DATA_DIR / 'wavs' / f"{waw_id}.wav")
        return waw_paths

def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        txt = []
        for line in f.readlines():
            txt.append(line)

        return txt

def calc_energies():
    os.makedirs(ROOT_PATH / "data" / "energies", exist_ok=True)
    waw_paths = get_wav_paths()
    text = process_text('./data/train.txt')
    print("Precaclulating energies")

    energy_min, energy_max = np.inf, -np.inf

    STFT = TacotronSTFT()
    for i in tqdm(range(len(text))):
        audio_tensor, sr = torchaudio.load(waw_paths[i])
        audio_tensor = audio_tensor[0:1, :]
        energy = STFT.calc_energy(audio_tensor)
        np.save(ROOT_PATH / "data" / "energies" / f'{i}.npy', energy)

        if torch.min(energy) < energy_min:
            energy_min = torch.min(energy)
        if torch.max(energy) > energy_max:
            energy_max = torch.max(energy)

    np.save(ROOT_PATH / "small data" / "energy_min.npy", energy_min)
    np.save(ROOT_PATH / "small data" / "energy_max.npy", energy_max)


if __name__ == "__main__":
    calc_energies()