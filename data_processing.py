import numpy as np
import pickle
import random
import torch
from torch.utils.data import Dataset

# Utility functions
def random_perturb(feature_len, length):
    return np.linspace(0, feature_len, length + 1, dtype=np.uint16)

def sampling(feature, length, d):
    new_feature = np.zeros((length, d))
    sample_index = random_perturb(feature.shape[0], length)
    for i in range(len(sample_index) - 1):
        if sample_index[i] == sample_index[i + 1]:
            new_feature[i, :] = feature[sample_index[i], :]
        else:
            new_feature[i, :] = feature[sample_index[i]:sample_index[i + 1], :].mean(0)
    return new_feature

# Dataset class
class MultiModalDataset(Dataset):
    def __init__(self, audio_data, text_data, vision_data, labels, ids):
        self.audio_data = audio_data
        self.text_data = text_data
        self.vision_data = vision_data
        self.labels = labels
        self.ids = ids

    def __len__(self):
        return len(self.audio_data)

    def __getitem__(self, idx， length):
        audio = self.audio_data[idx]
        audio = sampling(audio, length, audio.shape[1])
        audio += np.random.normal(0, 3, audio.shape)

        text = self.text_data[idx]
        text = sampling(text, length, text.shape[1])
        text += np.random.normal(0, 3, text.shape)

        vision = self.vision_data[idx]
        vision = sampling(vision, length, vision.shape[1])
        vision += np.random.normal(0, 3, vision.shape)

        label = self.labels[idx]
        id = self.ids[idx]
        return audio, text, vision, label, id

# Data loading function
def load_data(file_path):
    with open(file_path, 'rb') as f:
        aligned_data = pickle.load(f)
    return aligned_data
