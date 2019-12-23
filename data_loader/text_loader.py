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
import pandas as pd
import torchvision.transforms as standard_transforms
from nltk.stem.snowball import SnowballStemmer
import random

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

        if config.data_type == "SENTEMO":
            #Init
            self.word2idx = {}
            self.idx2word = {}
            self.vocab = set()
            #Read Data
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
            

        elif config.data_type == "SEM_EVAL_OC":
            #Init
            self.word2idx = {}
            self.idx2word = {}
            self.vocab = set()

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
                anger0_x, anger0_y          = self.parse_oc(self.config.Train_OC_Anger)
                anger1_x, anger1_y          = self.parse_oc(self.config.Valid_OC_Anger)
                anger2_x, anger2_y          = self.parse_oc(self.config.Test_OC_Anger)
                fear0_x, fear0_y            = self.parse_oc(self.config.Train_OC_Fear)
                fear1_x, fear1_y            = self.parse_oc(self.config.Valid_OC_Fear)
                fear2_x, fear2_y            = self.parse_oc(self.config.Test_OC_Fear)
                joy0_x, joy0_y              = self.parse_oc(self.config.Train_OC_Joy)
                joy1_x, joy1_y              = self.parse_oc(self.config.Valid_OC_Joy)
                joy2_x, joy2_y              = self.parse_oc(self.config.Test_OC_Joy)
                sadness0_x, sadness0_y      = self.parse_oc(self.config.Train_OC_Sadness)
                sadness1_x, sadness1_y      = self.parse_oc(self.config.Valid_OC_Sadness)
                sadness2_x, sadness2_y      = self.parse_oc(self.config.Test_OC_Sadness)
                
                #Do Splitting later followed by sampling
                pd_anger =  pd.DataFrame({"emotions": anger0_y + anger1_y + anger2_y})
                pd_anger["text"] = anger0_x + anger1_x + anger2_x
                pd_joy = pd.DataFrame({"emotions": joy0_y + joy1_y + joy2_y})
                pd_joy["text"] = joy0_x + joy1_x + joy2_x
                pd_fear = pd.DataFrame({"emotions": fear0_y + fear1_y + fear2_y})
                pd_fear["text"] = fear0_x + fear1_x + fear2_x
                pd_sad = pd.DataFrame({"emotions": sadness0_y + sadness1_y + sadness2_y})
                pd_sad["text"] = sadness0_x + sadness1_x + sadness2_x

                pd_anger["emotions"] = pd_anger["emotions"].apply(lambda x: x[1])
                pd_anger["emotions"] = pd_anger["emotions"][pd_anger["emotions"] > self.config.emo_threshold]
                pd_anger = pd_anger.dropna()
                pd_anger["emotions"] = pd_anger["emotions"].apply(lambda x: 0)

                pd_joy["emotions"] = pd_joy["emotions"].apply(lambda x: x[1])
                pd_joy["emotions"] = pd_joy["emotions"][pd_joy["emotions"] > self.config.emo_threshold]
                pd_joy = pd_joy.dropna()
                pd_joy["emotions"] = pd_joy["emotions"].apply(lambda x: 1)

                pd_fear["emotions"] = pd_fear["emotions"].apply(lambda x: x[1])
                pd_fear["emotions"] = pd_fear["emotions"][pd_fear["emotions"] > self.config.emo_threshold]
                pd_fear = pd_fear.dropna()
                pd_fear["emotions"] = pd_fear["emotions"].apply(lambda x: 2)

                pd_sad["emotions"] = pd_sad["emotions"].apply(lambda x: x[1])
                pd_sad["emotions"] = pd_sad["emotions"][pd_sad["emotions"] > self.config.emo_threshold]
                pd_sad = pd_sad.dropna()
                pd_sad["emotions"] = pd_sad["emotions"].apply(lambda x: 3)
                
                data = pd.concat([pd_anger, pd_joy, pd_fear, pd_sad], ignore_index=True)

                data = data.sample(frac=1).reset_index(drop=True)

                #data.to_csv('data_x.csv')
                
                # lowering case
                if self.config.lower_case == True:
                    data['text'] = data['text'].apply(lambda x: x.lower())
                
                # remove non-alpha 
                if self.config.remove_nonalpha:
                    data['text'] = data['text'].replace('[^a-zA-Z ]', '', regex=True)

                # stemming
                if self.config.stemming == True:
                    snowball_stemmer = SnowballStemmer(language='english')
                    data['text'] = data['text'].apply(lambda x: " ".join([snowball_stemmer.stem(word) for word in x.split(' ')]))
                

                #data.to_csv('data_x_2.csv')
                data["token_size"] = data["text"].apply(lambda x: len(x.split(' ')))

                data = data.loc[data['token_size'] < 80].copy()

                if self.config.remove_stopwords == True:
                    nlp = spacy.load('en_core_web_sm')
                    spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
                    data['text'].apply(lambda x: [item for item in x if item not in spacy_stopwords])
                    data.to_csv('data_x_stop.csv')

                
                if self.config.use_weighting_words == True:
                    self.weighting_words = pd.read_csv(self.config.weighting_words)
                    data = data.apply(self.add_weighting_words, axis=1)
                    data.to_csv('data_weight.csv')            
                
                    
                self.create_index(data["text"].values.tolist())
                input_tensor = [[self.word2idx[s] for s in es.split(' ')]  for es in data["text"].values.tolist()]
                max_length_inp = self.max_length(input_tensor)
                input_tensor = [self.pad_sequences(x, max_length_inp) for x in input_tensor]
                emotions = list(set(data.emotions.unique()))
                # binarizer
                mlb = preprocessing.MultiLabelBinarizer()
                data_labels =  [set(emos) & set(emotions) for emos in data[['emotions']].values]
                bin_emotions = mlb.fit_transform(data_labels)
                target_tensor = np.array(bin_emotions.tolist())

                input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)
                input_tensor_val, input_tensor_test, target_tensor_val, target_tensor_test = train_test_split(input_tensor_val, target_tensor_val, test_size=0.5)
                
                #for Infernce
                self.test_data = input_tensor_test
                self.test_labels = target_tensor_test

                train = SENTEMO_Data(input_tensor_train, target_tensor_train)
                valid = SENTEMO_Data(input_tensor_val, target_tensor_val)
                test = SENTEMO_Data(input_tensor_test, target_tensor_test)

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
        
        else:
            raise Exception("Please specify in the json a specified mode in data_mode")
    
    def add_weighting_words(self, row):
        row['text'] = row['text'] + ' ' + self.random_weighting_words(row['emotions'])
        return row
        
    def random_weighting_words(self, emotion, k=10):
        """
        0: anger
        1: joy
        2: fear
        3: sad
        """
        if emotion == 0:
            emotion = 'anger'
        elif emotion == 1:
            emotion = 'joy'
        elif emotion == 2:
            emotion = 'fear'
        elif emotion == 3: 
            emotion = 'sadness'

        df_words = self.weighting_words.loc[self.weighting_words['emotion'] == emotion, ['words']]
        words = random.choices(df_words.iloc[0][0].split(' '), k=k)
        return " ".join(words)
        
    def tokenize_en(self, text):
        # tokenizes the english text into a list of strings(tokens)
        return [tok.text for tok in self.tokenizer.tokenizer(text)]

    def parse_oc(self, data_file, label_format='tuple'):
        """
        Returns:
            X: a list of tweets
            y: a list of (affect dimension, v) tuples corresponding to
            the ordinal classification targets of the tweets
        """
        with open(data_file, 'r',encoding="utf8") as fd:
            data = [l.strip().split('\t') for l in fd.readlines()][1:]
        X = [d[1] for d in data]
        y = [(d[2], int(d[3].split(':')[0])) for d in data]
        if label_format == 'list':
            y = [l[1] for l in y]
        return X, y

    def parse_e_c(self, data_file):
        """
        Returns:
            X: a list of tweets
            y: a list of lists corresponding to the emotion labels of the tweets
        """
        with open(data_file, 'r',encoding="utf8") as fd:
            data = [l.strip().split('\t') for l in fd.readlines()][1:]
        X = [d[1] for d in data]
        # dict.values() does not guarantee the order of the elements
        # so we should avoid using a dict for the labels
        y = [[int(l) for l in d[2:]] for d in data]

        return X, y

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
        if self.config.data_type == 'SENTEMO' or self.config.data_type == 'SEM_EVAL_OC':
            #Save Dicts for inference
            if self.config.mode == 'train':
                self.convert_to_pickle(self.word2idx, self.config.out_dir+'word2idx.pkl')
                self.convert_to_pickle(self.idx2word, self.config.out_dir+'idx2word.pkl')
                self.convert_to_pickle(self.vocab, self.config.out_dir+'vocab.pkl')
                vocab_size = {'embedded_dim':self.config.vocab_size}
                self.convert_to_pickle(vocab_size, self.config.out_dir+'vocab_size.pkl')
                np.save(self.config.out_dir+'test_data.npy',self.test_data,allow_pickle=True)
                np.save(self.config.out_dir+'test_labels.npy',self.test_labels,allow_pickle=True)