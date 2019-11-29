from transformer_arabic_to_english.translator import Translator as ArabicTranslator

class MultiTranslator():
    """
    Helper class to translate text of initialized input language to English.
    Translator supports Arabic-English, German-English, and French-English
    """

    def __init__(self, inputLanguage='ar'):
        """
        Args:
            inputLanguage: The language of feeded text to translate, value must be one of these 'fr', 'de', or 'ar'
        """
        self.__inputLanguage = inputLanguage

        if self.__inputLanguage == 'ar':
            self.translator = ArabicTranslator()
        else:
            raise NotImplementedError()



    def translate(self, text):
        """
        Translate the given string into the pre-initialized language
        args:
            text: the string value to be translated
        """
        return self.translator.translate(text)