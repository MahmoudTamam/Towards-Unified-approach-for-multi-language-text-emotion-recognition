import pandas as pd
import numpy as np
from yandex_translate import YandexTranslate
import emoji
import re

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

translate = YandexTranslate('trnsl.1.1.20191221T010442Z.9d28b61382dc3f7b.232b2b2105b3e06035dce39c6729fefe8be1b167')

anger0_x, anger0_y          = parse_oc('SemEval2018/Spanish/EI-oc/training/2018-EI-oc-Es-anger-train.txt')
anger1_x, anger1_y          = parse_oc('SemEval2018/Spanish/EI-oc/development/2018-EI-oc-Es-anger-dev.txt')
anger2_x, anger2_y          = parse_oc('SemEval2018/Spanish/EI-oc/test-gold/2018-EI-oc-Es-anger-test-gold.txt')
fear0_x, fear0_y            = parse_oc('SemEval2018/Spanish/EI-oc/training/2018-EI-oc-Es-fear-train.txt')
fear1_x, fear1_y            = parse_oc('SemEval2018/Spanish/EI-oc/development/2018-EI-oc-Es-fear-dev.txt')
fear2_x, fear2_y            = parse_oc('SemEval2018/Spanish/EI-oc/test-gold/2018-EI-oc-Es-fear-test-gold.txt')
joy0_x, joy0_y              = parse_oc('SemEval2018/Spanish/EI-oc/training/2018-EI-oc-Es-joy-train.txt')
joy1_x, joy1_y              = parse_oc('SemEval2018/Spanish/EI-oc/development/2018-EI-oc-Es-joy-dev.txt')
joy2_x, joy2_y              = parse_oc('SemEval2018/Spanish/EI-oc/test-gold/2018-EI-oc-Es-joy-test-gold.txt')
sadness0_x, sadness0_y      = parse_oc('SemEval2018/Spanish/EI-oc/training/2018-EI-oc-Es-sadness-train.txt')
sadness1_x, sadness1_y      = parse_oc('SemEval2018/Spanish/EI-oc/development/2018-EI-oc-Es-sadness-dev.txt')
sadness2_x, sadness2_y      = parse_oc('SemEval2018/Spanish/EI-oc/test-gold/2018-EI-oc-Es-sadness-test-gold.txt')

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
pd_anger["emotions"] = pd_anger["emotions"][pd_anger["emotions"] > 0]
pd_anger = pd_anger.dropna()
pd_anger["emotions"] = pd_anger["emotions"].apply(lambda x: 0)

pd_joy["emotions"] = pd_joy["emotions"].apply(lambda x: x[1])
pd_joy["emotions"] = pd_joy["emotions"][pd_joy["emotions"] > 0]
pd_joy = pd_joy.dropna()
pd_joy["emotions"] = pd_joy["emotions"].apply(lambda x: 1)

pd_fear["emotions"] = pd_fear["emotions"].apply(lambda x: x[1])
pd_fear["emotions"] = pd_fear["emotions"][pd_fear["emotions"] > 0]
pd_fear = pd_fear.dropna()
pd_fear["emotions"] = pd_fear["emotions"].apply(lambda x: 2)

pd_sad["emotions"] = pd_sad["emotions"].apply(lambda x: x[1])
pd_sad["emotions"] = pd_sad["emotions"][pd_sad["emotions"] > 0]
pd_sad = pd_sad.dropna()
pd_sad["emotions"] = pd_sad["emotions"].apply(lambda x: 3)

data = pd.concat([pd_anger, pd_joy, pd_fear, pd_sad], ignore_index=True)

data = data.sample(frac=1).reset_index(drop=True)

emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                            "]+", flags=re.UNICODE)
#print(translator.translate('Hola Amigo').text)

#print(translate.translate(emoji.demojize(data_x['text'][0]),'es-en')['text'][0])

export_csv = data.to_csv (r'spanish_translated.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

data['text'] = data['text'].apply(lambda x: translate.translate(emoji_pattern.sub(r'', x),'es-en')['text'][0])

export_csv = data.to_csv (r'spanish_translated.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

print(data.head())



