from translator import Translator



# print(translate('بسم الله'))
# print(translate('الحمد الله'))

arabic_text = """بسم الله الرحمن الرحيم
                ان الله لا يغير ما بقوم حتى يغيروا ما بأنفسهم
                """

translator = Translator()
print(translator.translate(arabic_text))