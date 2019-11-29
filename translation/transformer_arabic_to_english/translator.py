from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import time

import tensorflow as tf
from utils import tokenizer
import transformer_main
from fast_predict import FastPredict


class Translator():
    __DECODE_BATCH_SIZE = 32
    __EXTRA_DECODE_LENGTH = 100
    __BEAM_SIZE = 4
    __ALPHA = 0.6

    def __init__(self):
        tf.logging.set_verbosity(tf.logging.INFO)

        params = transformer_main.PARAMS_MAP["tiny"]
        params["beam_size"] = self.__BEAM_SIZE
        params["alpha"] = self.__ALPHA
        params["extra_decode_length"] = self.__EXTRA_DECODE_LENGTH
        params["batch_size"] = self.__DECODE_BATCH_SIZE

        self.__tf_estimator = tf.estimator.Estimator(model_fn=transformer_main.model_fn, model_dir="./tiny-model/", params=params)

        self.__subtokenizer = tokenizer.Subtokenizer("./tiny-model/vocab.ende.32768")


    # load model
    def __encode_and_add_eos(self, line):
        """Encode line with subtokenizer, and add EOS id to the end."""
        return self.__subtokenizer.encode(line) + [tokenizer.EOS_ID]

    def __trim_and_decode(self, ids):
        """Trim EOS and PAD tokens from ids, and decode to return a string."""
        try:
            index = list(ids).index(tokenizer.EOS_ID)
            return self.__subtokenizer.decode(ids[:index])
        except ValueError:  # No EOS found in sequence
            return self.__subtokenizer.decode(ids)


    def __get_input_fn(self, input):
        encoded_txt = self.__encode_and_add_eos(input)
        def input_fn(generator):
            def inner_input_fn():
                ds = tf.data.Dataset.from_tensors(encoded_txt)
                ds = ds.batch(self.__DECODE_BATCH_SIZE)
                return ds
            return inner_input_fn
        return input_fn

    def __translate_text(self, fastEstimator, txt):
        """Translate a single string."""
        #fastEstimator.input_fn=get_input_fn(txt,self.subtokenizer)
        predictions = fastEstimator.predict([1])
        translation = predictions[0]["outputs"]
        translation = self.__trim_and_decode(translation)
        tf.logging.info("Translation of \"%s\": \"%s\"" % (txt, translation))
        return translation


    
    def translate(self, text):
        estimator = FastPredict(self.__tf_estimator, self.__get_input_fn(text))
        result = self.__translate_text(estimator, text)
        return result


