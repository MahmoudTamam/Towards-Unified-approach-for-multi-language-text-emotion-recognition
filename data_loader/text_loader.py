import torch
import torchvision.utils as v_utils
from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset, Dataset
#from torchtext.data.utils import get_tokenizer
import spacy
import pickle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torchvision.transforms as standard_transforms


class SENTEMO_Data(Dataset):
    def __init__(self, X, y, input_transform= None, target_transform = None):
        self.data = X
        self.target = y
        self.length = [ np.sum(1 - np.equal(x, 0)) for x in X]
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        x_len = self.length[index]
        return x, y, x_len
    
    def __len__(self):
        return len(self.data)

class TextDataLoader(data.Dataset):
    def __init__(self, config):
        """
        :param config:
        """
        self.config = config

        if config.data_mode == "Text":
            #Init
            self.word2idx = {}
            self.idx2word = {}
            self.vocab = set()
            #Read Data
            if self.config.data_type == 'SENTEMO':

                if self.config.mode == 'test':
                    self.word2idx   =   pickle.load(open(self.config.out_dir+'word2idx.pkl',"rb"))
                    self.idx2word   =   pickle.load(open(self.config.out_dir+'idx2word.pkl',"rb"))
                    self.vocab      =   pickle.load(open(self.config.out_dir+'vocab.pkl',"rb"))
                    vocab_size      =   pickle.load(open(self.config.out_dir+'vocab_size.pkl',"rb"))
                    self.config.vocab_size = vocab_size['embedded_dim']

                    test_data = np.load(self.config.out_dir+'test_data.npy')
                    test_labels = np.load(self.config.out_dir+'test_labels.npy')
                    test = SENTEMO_Data(test_data, test_labels)
                    self.test_loader = DataLoader(test, batch_size=config.batch_size, shuffle=True, drop_last=True)
                    self.test_iterations = (len(test) + self.config.batch_size) // self.config.batch_size

                else:
                    data = self.load_from_pickle(directory=self.config.SENT_EMO_Path)
                    data["token_size"] = data["text"].apply(lambda x: len(x.split(' ')))
                    data = data.loc[data['token_size'] < 70].copy()
                    # sampling
                    data = data.sample(n=50000)
                    # construct vocab and indexing
                    self.create_index(data["text"].values.tolist())
                    # vectorize to tensor
                    input_tensor = [[self.word2idx[s] for s in es.split(' ')]  for es in data["text"].values.tolist()]
                    max_length_inp = self.max_length(input_tensor)
                    # inplace padding
                    input_tensor = [self.pad_sequences(x, max_length_inp) for x in input_tensor]
                    ### convert targets to one-hot encoding vectors
                    emotions = list(set(data.emotions.unique()))
                    # binarizer
                    mlb = preprocessing.MultiLabelBinarizer()
                    data_labels =  [set(emos) & set(emotions) for emos in data[['emotions']].values]
                    bin_emotions = mlb.fit_transform(data_labels)
                    target_tensor = np.array(bin_emotions.tolist()) 
                    # Creating training and validation sets using an 80-20 split
                    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

                    # Split the validataion further to obtain a holdout dataset (for testing) -- split 50:50
                    input_tensor_val, input_tensor_test, target_tensor_val, target_tensor_test = train_test_split(input_tensor_val, target_tensor_val, test_size=0.5)

                    #for Infernce
                    self.test_data = input_tensor_test
                    self.test_labels = target_tensor_test

                    #Init Transforms
                    self.input_transform = standard_transforms.Compose([
                        standard_transforms.ToTensor(),
                    ])

                    self.target_transform = standard_transforms.Compose([
                        standard_transforms.ToTensor(),
                    ])
                    #Creeate Datasets
                    train = SENTEMO_Data(input_tensor_train, target_tensor_train)#, input_transform=self.input_transform, target_transform=self.target_transform)
                    valid = SENTEMO_Data(input_tensor_val, target_tensor_val)#, input_transform=self.input_transform, target_transform=self.target_transform)
                    test = SENTEMO_Data(input_tensor_test, target_tensor_test)#, input_transform=self.input_transform, target_transform=self.target_transform)

                    self.train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True, drop_last=True,)
                    self.valid_loader = DataLoader(valid, batch_size=config.batch_size, shuffle=True, drop_last=True,)
                    self.test_loader = DataLoader(test, batch_size=config.batch_size, shuffle=True, drop_last=True,)

                    self.train_iterations = (len(train) + self.config.batch_size) // self.config.batch_size
                    self.valid_iterations = (len(valid) + self.config.batch_size) // self.config.batch_size
                    self.test_iterations = (len(test) + self.config.batch_size) // self.config.batch_size
                    
                    self.config.vocab_size = len(self.word2idx)
            
            elif self.config.data_type == 'IEMOCAP':
                raise NotImplementedError("This mode is not implemented YET")
                #utterances, videoSpeakers, videoLabels, videoText, videoAudio, videoVisual, transcripts, scripts, testVid = self.load_from_pickle(directory=self.config.pickle_path, encoding=self.config.pickle_encoding)
                #Create Tokenizer
                #self.tokenizer = spacy.load('en_core_web_sm')
                #Loop through all data and do tokenization
                #self.data_seq_len = []
                #self.data_text = []
                #for vid in scripts:
                #    self.data_seq_len.append(len(utterances[vid]))
                #    self.data_text.append(transcripts[vid])
                #Create Vocab

                #Padding

        elif config.data_mode == "Speech":
            raise NotImplementedError("This mode is not implemented YET")

        elif config.data_mode == "Multi_Speech_Text":
            raise NotImplementedError("This mode is not implemented YET")

        else:
            raise Exception("Please specify in the json a specified mode in data_mode")
        
        #raise NotImplementedError("This mode is not implemented YET")

    def tokenize_en(self, text):
        # tokenizes the english text into a list of strings(tokens)
        return [tok.text for tok in self.tokenizer.tokenizer(text)]

    def load_from_pickle(self, directory, encoding = None):
        if encoding is None:
            return pickle.load(open(directory,"rb"))
        return pickle.load(open(directory,"rb"), encoding=encoding)
   
    def convert_to_pickle(self, item, directory):
        pickle.dump(item, open(directory,"wb"))

    def create_index(self, sentences):
        for s in sentences:
            # update with individual tokens
            self.vocab.update(s.split(' '))
            
        # sort the vocab
        self.vocab = sorted(self.vocab)

        # add a padding token with index 0
        self.word2idx['<pad>'] = 0
        
        # word to index mapping
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1 # +1 because of pad token
        
        # index to word mapping
        for word, index in self.word2idx.items():
            self.idx2word[index] = word  
    
    def max_length(self, tensor):
        return max(len(t) for t in tensor)

    def pad_sequences(self, x, max_len):
        padded = np.zeros((max_len), dtype=np.int64)
        if len(x) > max_len: padded[:] = x[:max_len]
        else: padded[:len(x)] = x
        return padded

    def finalize(self):
        if self.config.data_type == 'SENTEMO':
            #Save Dicts for inference
            if self.config.mode == 'train':
                self.convert_to_pickle(self.word2idx, self.config.out_dir+'word2idx.pkl')
                self.convert_to_pickle(self.idx2word, self.config.out_dir+'idx2word.pkl')
                self.convert_to_pickle(self.vocab, self.config.out_dir+'vocab.pkl')
                vocab_size = {'embedded_dim':self.config.vocab_size}
                self.convert_to_pickle(vocab_size, self.config.out_dir+'vocab_size.pkl')
                np.save(self.config.out_dir+'test_data.npy',self.test_data,allow_pickle=True)
                np.save(self.config.out_dir+'test_labels.npy',self.test_labels,allow_pickle=True)