from torch.utils.data import Dataset
import torch

class TimeseriesDataset(Dataset):
    def __init__(self, raw, input_length, output_length):
        self.raw = raw
        self.input_length = input_length
        self.output_length = output_length

    def __len__(self):
        return len(self.raw) - (self.input_length + self.output_length) + 1

    def __getitem__(self, index):
        return (self.raw[index:index+self.input_length],
                self.raw[index+self.input_length:index+self.input_length+self.output_length])

class TimeseriesDatasetUnsqueeze(Dataset):
    def __init__(self, raw, input_length, output_length):
        self.raw = raw
        self.input_length = input_length
        self.output_length = output_length

    def __len__(self):
        return len(self.raw) - (self.input_length + self.output_length) + 1

    def __getitem__(self, index):
        history = torch.unsqueeze(torch.tensor(self.raw[index:index+self.input_length]), 1)
        prediction = torch.unsqueeze(torch.tensor(self.raw[index+self.input_length:index+self.input_length+self.output_length]), 1)
        return history, prediction

class AlternativeTimeseriesDataset(Dataset):
    def __init__(self, raw, input_length, output_length):
        self.raw = raw
        self.input_length = input_length
        self.output_length = output_length

    def __len__(self):
        return int(len(self.raw) / (self.input_length + self.output_length))

    def __getitem__(self, index):
        start = index * (self.input_length + self.output_length)
        return (self.raw[start:start+self.input_length],
                self.raw[start+self.input_length:start+self.input_length+self.output_length])