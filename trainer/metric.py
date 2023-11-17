import pandas as pd
from torchmetrics.classification import BinaryAUROC, BinaryF1Score, BinaryAveragePrecision
from torchmetrics.classification import BinaryRecall, BinarySpecificity, BinaryPrecision
from torchmetrics.classification import BinaryAccuracy
import torch

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

class FullMetricTracker:
    """
    A metric tracker of Trainer, only record the train & val results from each epoch
    Usecase:
        >>> tracker = FullMetricTracker()
        >>> tracker.update({'train_loss': 0.1, 'val_loss': 0.2})
        >>> tracker.update({'train_loss': 0.2, 'val_loss': 0.3})
        >>> tracker.get_data()
    
    """
    def __init__(self):
        self._data = None
    
    def update(self, dict_input):
        if self._data is None:
            self._data = pd.DataFrame(columns=list(dict_input.keys()))
        self._data.loc[len(self._data.index)] = dict_input.values()
        
    def reset(self):
        if self._data is not None:
            self._data = pd.DataFrame(columns=self._data.columns)
            
    def get_data(self):
        return self._data
    
def precision(output, target, device):
    with torch.no_grad():
        return BinaryPrecision().to(device)(output, target).item()

def accuracy(output, target, device):
    with torch.no_grad():
        return BinaryAccuracy().to(device)(output, target).item()

def sensitivity(output, target, device):
    with torch.no_grad():
        return BinaryRecall().to(device)(output, target).item()

def specificity(output, target, device):
    with torch.no_grad():
        return BinarySpecificity().to(device)(output, target).item()

def f1_score(output, target, device):
    with torch.no_grad():
        return BinaryF1Score().to(device)(output, target).item()

def auroc(output, target, device):
    with torch.no_grad():
        return BinaryAUROC().to(device)(output, target).item()

def auprc(output, target, device):
    with torch.no_grad():
        return BinaryAveragePrecision().to(device)(output, target.int()).item()