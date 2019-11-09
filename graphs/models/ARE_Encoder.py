import torch.nn as nn
from graphs.weights_initializer import weights_init

class MSER(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # define layers
        self.relu = nn.ReLU(inplace=True)

        # initialize weights
        self.apply(weights_init)
        raise NotImplementedError("This mode is not implemented YET")
    
    def forward(self, x):
        out = self.relu(x)
        raise NotImplementedError("This mode is not implemented YET")
        return out
