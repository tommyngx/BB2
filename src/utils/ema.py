import torch
import torch.nn as nn
from copy import deepcopy


class ModelEMA:
    """Exponential Moving Average for model parameters"""

    def __init__(self, model, decay=0.999):
        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        self.decay = decay

        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            for ema_param, param in zip(
                self.ema_model.parameters(), model.parameters()
            ):
                ema_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)
