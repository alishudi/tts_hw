import torch
import torchaudio
import numpy as np
from hw_tts.utils import ROOT_PATH
from tqdm import tqdm
import os
import pyworld as pw

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

def interpolation(arr):
    x = np.where(arr == 0)[0]
    xp = np.where(arr != 0)[0]
    fp = arr[xp]
    interp = np.interp(x, xp, fp)
    res = arr.copy()
    res[x] = interp
    return res

def calc_pitches():
    os.makedirs(ROOT_PATH / "data" / "pitches", exist_ok=True)
    waw_paths = get_wav_paths()
    text = process_text('./data/train.txt')
    print("Precaclulating pitches")

    pitch_min, pitch_max = np.inf, -np.inf

    for i in tqdm(range(len(text))):
        audio_tensor, sr = torchaudio.load(waw_paths[i])
        audio_tensor = audio_tensor[0:1, :].numpy().squeeze(0).astype('double')
        _f0, t = pw.dio(audio_tensor, sr)   # raw pitch extractor
        pitch = pw.stonemask(audio_tensor, _f0, t, sr) # pitch refinement
        pitch = interpolation(pitch) #we use linear interpolation to fill the unvoiced frame in pitch contour
        pitch = np.log(pitch) # we transform the resulting pitch contour to logarithmic scale
        np.save(ROOT_PATH / "data" / "pitches" / f'{i}.npy', pitch)

        if np.min(pitch) < pitch_min:
            pitch_min = np.min(pitch)
        if np.max(pitch) > pitch_max:
            pitch_max = np.max(pitch)

    np.save(ROOT_PATH / "data" / "pitch_min.npy", pitch_min)
    np.save(ROOT_PATH / "data" / "pitch_max.npy", pitch_max)



if __name__ == "__main__":
    calc_pitches()