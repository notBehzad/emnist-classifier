import torch.nn as nn

class EMNISTClassifier(nn.Module):
    def __init__(self): 
        super().__init__()
        self.Flatten = nn.Flatten()
        
        self.Layers = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 47)
        )
        
    def forward(self, x):
        x = self.Flatten(x)
        x = self.Layers(x)
        return x