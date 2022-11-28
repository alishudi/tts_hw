from torch.utils.data import Sampler
import torch
from random import shuffle


class GroupLengthBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, batches_per_group=20, sort_groups=False):
        #if sort_groups=True then groups are iterated from shortest to longest, like SortaGrad in DeepSpeech2 
        super().__init__(data_source)
        
        self.data_source = data_source
        self.batch_size = batch_size
        self.batches_per_group = batches_per_group
        self.sort_groups = sort_groups

        self.group_size = self.batch_size * self.batches_per_group
        self.n_groups = len(self.data_source) // self.group_size

        lengths = []
        for sample in data_source:
            lengths.append(sample['spectrogram'].shape[-1])
        self.idxs = torch.argsort(torch.tensor(lengths[:self.n_groups * self.group_size]))
        

    def __iter__(self):
        groups_order = list(range(self.n_groups))
        if not self.sort_groups:
            shuffle(groups_order)
        for group_idx in groups_order:
            group = self.idxs[group_idx * self.group_size: (group_idx + 1) * self.group_size]
            shuffle(group)
            for i in range(self.batches_per_group):
                yield(group[i * self.batch_size: (i + 1) * self.batch_size])

    def __len__(self):
        return len(self.data_source) // self.batch_size
