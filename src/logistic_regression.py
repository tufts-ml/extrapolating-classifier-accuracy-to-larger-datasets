import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(in_features=768, out_features=1)
        
    def forward(self, x, slices):
        proba = torch.sigmoid(self.fc(x))
        pooled_proba = torch.stack([torch.mean(split, dim=0) for split in torch.split(proba, slices)])
        return pooled_proba