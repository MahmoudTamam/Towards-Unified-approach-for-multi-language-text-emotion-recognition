import torch.nn as nn
from graphs.weights_initializer import weights_init
import torch

class Text_Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        #Embedding
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.embedding_dim)
        # define layers

        self.Gru_model = nn.GRU(input_size = self.config.embedding_dim, 
                                hidden_size = self.config.hiddenUnits, 
                                num_layers = self.config.n_layers,
                                bidirectional = self.config.bidirectional,
                                bias=self.config.bias)
        
        self.n_directions = 2 if self.config.bidirectional else 1

        self.fc = nn.Linear(self.n_directions * self.config.hiddenUnits, self.config.num_classes)
        self.dropout = nn.Dropout(p=0.5)
        self.softmax =  nn.LogSoftmax(dim=1)
        # initialize weights
        self.apply(weights_init)
        
        #raise NotImplementedError("This mode is YET Finalized, Word embeddeding, Attention, Agent, NLL Loss, json, Agent")
    def initialize_hidden_state(self, batch_sz, device):
        return torch.zeros((1, self.config.batch_size, self.n_directions *self.config.hiddenUnits)).to(device)
    
    def forward(self, x):
        x = self.embedding(x)
        #self.hidden = self.initialize_hidden_state(x.shape[0],self.config.device)
        x, _ = self.Gru_model(x)
        x = x[-1, :, :]
        x = self.dropout(x)
        x = self.fc(x)
        x = self.softmax(x)
        #raise NotImplementedError("This mode is YET Finalized, previous hidden")
        return x    
