import logging
import os

from hw_tts.utils import ROOT_PATH
from tqdm import tqdm

import torch
import numpy as np
import time
from FastSpeech.text import text_to_sequence
from torch.utils.data import Dataset
from hw_tts.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)

class LJspeechDataset(Dataset):
    def __init__(self, config_parser: ConfigParser):
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

            character = torch.from_numpy(character)
            duration = torch.from_numpy(duration)
            mel_gt_target = torch.from_numpy(mel_gt_target)

            buffer.append({"text": character, "duration_predictor_target": duration,
                        "mel_target": mel_gt_target})

        end = time.perf_counter()
        print("cost {:.2f}s to load all data into buffer.".format(end-start))

        return buffer






# class LJspeechDataset(BaseDataset):
#     def __init__(self, part, data_dir=None, *args, **kwargs):
#         if data_dir is None:
#             data_dir = ROOT_PATH / "data" / "datasets" / "ljspeech"
#             data_dir.mkdir(exist_ok=True, parents=True)
#         self._data_dir = data_dir
#         # index = self._get_or_load_index(part)

#         super().__init__(index, *args, **kwargs)


#     def _load_dataset(self):
#         arch_path = self._data_dir / "LJSpeech-1.1.tar.bz2"
#         print(f"Loading LJSpeech")
#         download_file(URL_LINKS["dataset"], arch_path)
#         shutil.unpack_archive(arch_path, self._data_dir)
#         for fpath in (self._data_dir / "LJSpeech-1.1").iterdir():
#             shutil.move(str(fpath), str(self._data_dir / fpath.name))
#         os.remove(str(arch_path))
#         shutil.rmtree(str(self._data_dir / "LJSpeech-1.1"))

    

    # def _get_or_load_index(self, part):
    #     index_path = self._data_dir / f"{part}_index.json"
    #     if index_path.exists():
    #         with index_path.open() as f:
    #             index = json.load(f)
    #     else:
    #         index = self._create_index(part)
    #         with index_path.open("w") as f:
    #             json.dump(index, f, indent=2)
    #     return index

    # def _create_index(self, part):
    #     index = []
    #     split_dir = self._data_dir / part
    #     if not split_dir.exists():
    #         self._load_dataset()

    #     wav_dirs = set()
    #     for dirpath, dirnames, filenames in os.walk(str(split_dir)):
    #         if any([f.endswith(".wav") for f in filenames]):
    #             wav_dirs.add(dirpath)
    #     for wav_dir in tqdm(
    #             list(wav_dirs), desc=f"Preparing ljspeech folders: {part}"
    #     ):
    #         wav_dir = Path(wav_dir)
    #         trans_path = list(self._data_dir.glob("*.csv"))[0]
    #         with trans_path.open() as f:
    #             for line in f:
    #                 w_id = line.split('|')[0]
    #                 w_text = " ".join(line.split('|')[1:]).strip()
    #                 wav_path = wav_dir / f"{w_id}.wav"
    #                 if not wav_path.exists(): # elem in another part
    #                     continue
    #                 t_info = torchaudio.info(str(wav_path))
    #                 length = t_info.num_frames / t_info.sample_rate
    #                 if w_text.isascii():
    #                     index.append(
    #                         {
    #                             "path": str(wav_path.absolute().resolve()),
    #                             "text": w_text.lower(),
    #                             "audio_len": length,
    #                         }
    #                     )
    #     return 