import sys
import os
import pickle
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime
from mamba_ssm.modules.mamba_simple import MutiMamba
from src.utils import MetricsTop
from src.utils.functions import dict_to_str

logger = logging.getLogger('MMSA')

def set_random_seed(seed):
    print("Random seed:", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def random_perturb(length, segments):
    return np.linspace(0, length, segments + 1, dtype=np.uint16)

def sampling(features, length, dim):
    sampled = np.zeros((length, dim))
    indices = random_perturb(features.shape[0], length)
    for i in range(len(indices) - 1):
        start, end = indices[i], indices[i + 1]
        sampled[i, :] = features[start:end].mean(0) if start != end else features[start, :]
    return sampled

class MultiModalDataset(Dataset):
    def __init__(self, audio, text, vision, labels, ids):
        self.audio = audio
        self.text = text
        self.vision = vision
        self.labels = labels
        self.ids = ids

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, idx):
        audio = sampling(self.audio[idx], length, self.audio[idx].shape[1])
        text = sampling(self.text[idx], length, self.text[idx].shape[1])
        vision = sampling(self.vision[idx], length, self.vision[idx].shape[1])
        label = self.labels[idx]
        return audio, text, vision, label, self.ids[idx]

class MambaTrainer:
    def __init__(self, metric_key="MAE", mode="regression"):
        self.metric_key = metric_key
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics = MetricsTop(mode).getMetics('MOSEI')
        self.early_stop = 10
        self.update_epochs = 10
        self.start_time = datetime.now()

    def do_train(self, model, train_loader, valid_loader, test_loader, return_results=True):
        model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        best_valid = float('inf') if self.metric_key == 'MAE' else -float('inf')
        epoch_results = {'train': [], 'valid': [], 'test': []} if return_results else None

        for epoch in range(1, epoch):
            model.train()
            y_pred, y_true, train_loss = [], [], 0.0
            for batch in tqdm(train_loader):
                optimizer.zero_grad()
                audio, text, vision, labels, _ = [x.to(self.device).float() if isinstance(x, torch.Tensor) else x for x in batch]
                outputs = model(audio.size(0), audio, vision, text, False)
                labels = labels.view(-1, 1).float()
                loss = F.l1_loss(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                y_pred.append(outputs)
                y_true.append(labels)

            scheduler.step()
            train_loss /= len(train_loader)
            train_metrics = self.metrics(torch.cat(y_pred), torch.cat(y_true))
            train_metrics["Loss"] = train_loss
            logger.info(f"Train: {dict_to_str(train_metrics)}")

            val_metrics = self.evaluate(model, valid_loader)
            if ((self.metric_key == 'MAE' and val_metrics[self.metric_key] < best_valid) or
                (self.metric_key != 'MAE' and val_metrics[self.metric_key] > best_valid)):
                best_valid = val_metrics[self.metric_key]
                torch.save(model.cpu().state_dict(), f"your path")
                model.to(self.device)

            if return_results:
                epoch_results['train'].append(train_metrics)
                epoch_results['valid'].append(val_metrics)
                epoch_results['test'].append(self.evaluate(model, test_loader))

        return epoch_results

    def evaluate(self, model, loader):
        model.eval()
        y_pred, y_true, total_loss = [], [], 0.0
        with torch.no_grad():
            for batch in tqdm(loader):
                audio, text, vision, labels, _ = [x.to(self.device).float() if isinstance(x, torch.Tensor) else x for x in batch]
                outputs = model(audio.size(0), audio, vision, text, False)
                labels = labels.view(-1, 1).float()
                loss = F.l1_loss(outputs, labels)
                total_loss += loss.item()
                y_pred.append(outputs)
                y_true.append(labels)

        metrics = self.metrics(torch.cat(y_pred), torch.cat(y_true))
        metrics['Loss'] = round(total_loss / len(loader), 4)
        return metrics

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 202
    set_random_seed(seed)

    with open('path to dataset', 'rb') as f:
        data = pickle.load(f)

    datasets = {split: MultiModalDataset(
        data[split]['audio'], data[split]['text'], data[split]['vision'],
        data[split]['regression_labels'], data[split]['id']) for split in ['train', 'valid', 'test']
    }

    loaders = {split: DataLoader(datasets[split], batch_size=1024, shuffle=True, num_workers=3)
               for split in datasets}

    trainer = MambaTrainer()
    model = MutiMamba(128, layer=3, audio_dim=74, vision_dim=35, text_dim=768).to(device)
    results = trainer.do_train(model, loaders['train'], loaders['valid'], loaders['test'])
