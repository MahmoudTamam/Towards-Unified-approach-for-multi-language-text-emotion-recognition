import pandas as pd
import numpy as np

def parse_oc(data_file, label_format='tuple'):
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

data_x, data_y = parse_oc("EI-oc-En-anger-train.txt")
dato_x, dato_y = parse_oc("EI-oc-En-joy-train.txt")

data_y = data_y+dato_y
data_x = data_x+dato_x

data_y = pd.DataFrame({"emotion": data_y})
#data_y.emotion = data_y["emotion"].split()

data_y["text"] = data_x
data_y["emotion"] = data_y["emotion"].apply(lambda x: x[1])
data_y["emotion"] = data_y["emotion"][data_y["emotion"] > 1]
data_y = data_y.dropna()

data_y["token_size"] = data_y["text"].apply(lambda x: len(x.split(' ')))
data_y = data_y.sample(frac=1).reset_index(drop=True)

print(data_y)

print(np.unique(data_y["token_size"],return_counts=True))

exit(0)

def parse_e_c(data_file):
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


data_x, data_y = parse_e_c("EI-oc-En-anger-train")

converter = np.array([[0,1,0,0],[1,0,0,0],[0,0,0,1]])


print(data_y[0])

exit(0)
data_x = pd.DataFrame({"text": data_x})

data_x["token_size"] = data_x["text"].apply(lambda x: len(x.split(' ')))

print(np.unique(data_x["token_size"],return_counts=True))

#print (data_x["token_size"])