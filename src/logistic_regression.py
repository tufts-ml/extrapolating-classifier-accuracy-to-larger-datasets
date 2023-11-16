import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dimension=1, num_heads=1):
        self.hidden_dimension = hidden_dimension
        self.num_heads = num_heads
        super().__init__()
        self.attention_pooling = nn.ModuleList([nn.Linear(in_features=self.hidden_dimension+1, out_features=1)\
                                                for head in range(self.num_heads)])

    def forward(self, x, slices):
        b, d = x.shape
        positional_encoding = torch.concat([torch.arange(0, num_slices)/num_slices for num_slices in slices]).to(x.device)
        Phi = torch.hstack([positional_encoding[:, None], x])
        weights = torch.hstack([self.attention_pooling[head](Phi) for head in range(self.num_heads)])
        attention_weights = torch.vstack([F.softmax(split, dim=0) for split in torch.split(weights, slices)])
        context_vector = attention_weights*x
        pooled = torch.vstack([torch.sum(split, dim=0) for split in torch.split(context_vector, slices)])
        return pooled

class LogisticRegression(nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.attention = attention
        self.attention_pooling = AttentionPooling(hidden_dimension=1, num_heads=1)
        self.fc = nn.LazyLinear(out_features=1)
        
    def forward(self, x, slices):
        b, c, d = x.shape
        x = x.view(b, -1)
        proba = torch.sigmoid(self.fc(x))
        if not self.attention:
            # Mean pooling
            pooled_proba = torch.stack([torch.mean(split, dim=0) for split in torch.split(proba, slices)])
        else:
            pooled_proba = self.attention_pooling(proba, slices)
        return pooled_proba