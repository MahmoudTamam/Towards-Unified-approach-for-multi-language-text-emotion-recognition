import torch
from textblob import TextBlob

class EnglishTranslator():
    """
    Helper class to translate text to English
    """

    def __init__(self, languages=['de']):
        """
        Load the translation models. This process may take several minutes
        args:
            languages: list of languages to initialize their models, supperted languages are ['de', 'ru']

        """
        
        self.__trans_models = {lang: None for lang in languages }
        self.__load_torch_models()


    def translate(self, text):
        """
        Translate given text to English. Failback to google translator
        """

        trans_text = ''
        text_blob = TextBlob(text)
        lang = text_blob.detect_language()
        
        if lang in self.__trans_models:
            trans_text = self.__trans_models[lang].translate(text)
        else:
            trans_text = text_blob.translate(to='en')

        return trans_text


    def __load_torch_models(self):
        if 'de' in self.__trans_models:
            self.__trans_models['de'] = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')
        
        if 'ru' in self.__trans_models:
            self.__trans_models['ru'] = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.ru-en.single_model', tokenizer='moses', bpe='fastbpe')

