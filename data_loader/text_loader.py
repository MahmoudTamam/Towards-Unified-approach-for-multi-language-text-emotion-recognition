import torch
import torchvision.utils as v_utils
from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset, Dataset

class TextDataLoader(data.Dataset):
    def __init__(self, config):
        """
        :param config:
        """
        self.config = config

        if config.data_mode == "Text":

            self.train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True)
            self.valid_loader = DataLoader(valid, batch_size=config.batch_size, shuffle=False)
        
        elif config.data_mode == "Speech":
            raise NotImplementedError("This mode is not implemented YET")

        elif config.data_mode == "Multi_Speech_Text":
            raise NotImplementedError("This mode is not implemented YET")

        else:
            raise Exception("Please specify in the json a specified mode in data_mode")
        
        raise NotImplementedError("This mode is not implemented YET")
    
    def __getitem__(self, index):
        raise NotImplementedError("This mode is not implemented YET")

    def __len__(self):
        raise NotImplementedError("This mode is not implemented YET")

