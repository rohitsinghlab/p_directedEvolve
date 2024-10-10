import torch
from torch import nn
from einops.layers.torch import Rearrange

class PredHeadESM2(nn.Module):
    def __init__(self):
        super(PredHeadESM2, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1280, 1),
            nn.ReLU(),
            Rearrange("b l h -> b (l h)"),# remove the last dim
            nn.Linear(171, 3),
        )
        return
    
    def forward(self, x):
        return self.layers(x)
    
    
class PredHeadRaygun(nn.Module):
    def __init__(self):
        super(PredHeadRaygun, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1280, 1),
            nn.ReLU(),
            Rearrange("b l h -> b (l h)"), # b, 50
            nn.Linear(50, 3),
        )
        
    def forward(self, x):
        return self.layers(x)

    
class PredHeadESM2_3b(nn.Module):
    def __init__(self):
        super(PredHeadESM2_3b, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2560, 1),
            nn.ReLU(),
            Rearrange("b l h -> b (l h)"),# remove the last dim
            nn.Linear(171, 3),
        )
        return
    
    def forward(self, x):
        return self.layers(x)
