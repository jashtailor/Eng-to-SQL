# importing the necessary libraries
import streamlit as st

import tensorflow
from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

import pandas as pd
import numpy as np 
from numpy import array
from numpy import asarray
from numpy import zeros

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------

# loading the files
reconstructed_encoder = keras.models.load_model("encoder")

reconstructed_encoder.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

reconstructed_decoder = keras.models.load_model("decoder")

reconstructed_decoder.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

with open("dict1.json", "w") as outfile:
    json.dump(word2idx_outputs, outfile)

with open("dict2.json", "w") as outfile:
    json.dump(idx2word_target, outfile)

with open("dict3.json", "w") as outfile:
    json.dump(word2idx_inputs, outfile)
    
a = open("dict1.json")
word2idx_outputs_frontend = json.load(a)

b = open("dict2.json")
idx2word_target_frontend = json.load(b)

c = open("dict3.json")
word2idx_inputs_frontend = json.load(c)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------

def translate_sentence(input_seq):
    states_value = reconstructed_encoder.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = word2idx_outputs_frontend['<sos>']
    eos = word2idx_outputs_frontend['<eos>']
    output_sentence = []
    lst1 = []

    for _ in range(58):
        output_tokens, h, c = reconstructed_decoder.predict([target_seq] + states_value)
        idx = np.argmax(output_tokens[0, 0, :])

        if eos == idx:
            break

        word = ''

        if idx > 0:
            idx = str(idx)
            word = idx2word_target_frontend[idx]
            output_sentence.append(word)

        target_seq[0, 0] = idx
        states_value = [h, c]

    return ' '.join(output_sentence)

def preprocess(text):
  text1 = text.lower()
  lst1 = text1.split(' ')
  # print(lst1)
  txt_seq = []
  count = max(word2idx_inputs_frontend.values()) + 1
  for each in lst1:
    '''try:
      temp = word2idx_inputs_frontend[each]
    except:
      temp_lst = glove_vectors.most_similar(each)
      temp = temp_lst[0][0]'''
    if each not in word2idx_inputs_frontend.keys():

      word2idx_inputs_frontend[each] = count
      count = count + 1
    else:
      temp = word2idx_inputs_frontend[each]
    txt_seq.append(temp)
  # print(txt_seq)
  if len(txt_seq)<46:
    lst2 = [[0]*(46-len(txt_seq)) + txt_seq]
  # print(lst2)
  translation = translate_sentence(lst2)
  # print(translation)
  return translation
 
