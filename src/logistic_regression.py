import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LogisticRegression(nn.Module):
    def __init__(self, attention, num_classes):
        self.attention = attention
        self.num_classes = num_classes
        super().__init__()
        self.fc = nn.Linear(in_features=768, out_features=self.num_classes)
        self.attention_pooling = nn.ModuleList([nn.Linear(in_features=112, out_features=1)\
                                                for class_index in range(self.num_classes)])

    def forward(self, x, slices):
        x = torch.sigmoid(self.fc(x))
        # Mean Pooling
        if not self.attention:
            pooled = torch.stack([torch.mean(split, dim=0) for split in torch.split(x, slices)])
        # Attention Pooling
        else:
            # x.shape = (num_slices_in_batch, num_classes)
            # padded_x.shape = (num_slices_in_batch, 111, num_classes)
            padded_x = torch.vstack([split[torch.linspace(start=0, end=split.shape[0]-1, steps=111, dtype=torch.long)]
                                           .expand(split.shape[0], 111, self.num_classes)\
                                           for split in torch.split(x, slices)])
            # Phi.shape = (num_slices_in_batch, num_classes, 374)
            Phi = torch.cat([torch.unsqueeze(x, -1),
                             torch.permute(padded_x, (0, 2, 1))], dim=-1)
            # Pass per-slice perdicted probabilty of WMD and SBI through 
            # seperate linear layers.
            # weights.shape = (num_slices_in_batch, num_classes)
            weights = torch.hstack([self.attention_pooling[class_index](Phi[:,class_index,:])\
                                    for class_index in range(self.num_classes)])
            # F.softmax(split, dim=0) where split.shape = (num_slices_in_scan, num_classes)
            # attention_weights.shape = (num_slices_in_batch, num_classes)
            attention_weights = torch.vstack([F.softmax(split, dim=0)\
                                              for split in torch.split(weights, slices)])
            context_vector = attention_weights*x
            pooled = torch.vstack([torch.sum(split, dim=0)\
                                   for split in torch.split(context_vector, slices)])
            #print(attention_weights)
        return pooled