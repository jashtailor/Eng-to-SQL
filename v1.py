# importing the necessary libraries
import streamlit as st

import tensorflow
from tensorflow import keras
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
import json
import sys
import time
from time import sleep
import requests

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

  txt_seq = []
  count = max(word2idx_inputs_frontend.values()) + 1
  for each in lst1:
 
    if each not in word2idx_inputs_frontend.keys():

      word2idx_inputs_frontend[each] = count
      count = count + 1
    else:
      temp = word2idx_inputs_frontend[each]
    txt_seq.append(temp)

  if len(txt_seq)<46:
    lst2 = [[0]*(46-len(txt_seq)) + txt_seq]

  translation = translate_sentence(lst2)
  st.write(translation)

  return translation

@st.cache(suppress_st_warning=True)
def transcribe(audio_file):
    # import argparse

    '''
    # Parse argument
    parser = argparse.ArgumentParser(prog='transcriber')
    parser.add_argument('-i', help='Enter the URL of YouTube video')
    args = parser.parse_args()
    '''

    # 1. Read API from text file
    # f = open("api.txt", "r")
    api_key = '7dceeb758cb7442481d1927aa97a4ad6'

    # print('1. API is read ...')

    

    # 3. Upload YouTube audio file to AssemblyAI

    

    # filename = audio_file

    
    def read_file(audio_file, chunk_size=5242880):
        with open(audio_file, 'rb') as _file:
            while True:
                data = _file.read(chunk_size)
                if not data:
                    break
                yield data
    
     
    headers = {'authorization': api_key}
    response = requests.post('https://api.assemblyai.com/v2/upload',
                             headers=headers,
                             data=read_file(audio_file))

    audio_url = response.json()['upload_url']

    # print('3. YouTube audio file has been uploaded to AssemblyAI')


    # 4. Transcribe uploaded audio file

    # import requests

    endpoint = "https://api.assemblyai.com/v2/transcript"

    json = {
      "audio_url": audio_url
    }

    headers = {
        "authorization": api_key,
        "content-type": "application/json"
    }

    transcript_input_response = requests.post(endpoint, json=json, headers=headers)

    # print('4. Transcribing uploaded file')


    # 5. Extract transcript ID

    transcript_id = transcript_input_response.json()["id"]

    print('5. Extract transcript ID')

    # 6. Retrieve transcription results
    endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
    headers = {
        "authorization": api_key,
    }

    transcript_output_response = requests.get(endpoint, headers=headers)

    # print('6. Retrieve transcription results')

    # Check if transcription is complete
    

    while transcript_output_response.json()['status'] != 'completed':
      sleep(5)
      print('Transcription is processing ...')
      transcript_output_response = requests.get(endpoint, headers=headers)


    #print(transcript_output_response.json()["status"])


    # 7. Print transcribed text

    print('----------\n')
    print('Output:\n')
    print(transcript_output_response.json()["text"])


    # 8. Save transcribed text to file
    '''
    # Save as TXT file
    yt_txt = open('yt.txt', 'w')
    yt_txt.write(transcript_output_response.json()["text"])
    yt_txt.close()

    # Save as SRT file
    srt_endpoint = endpoint + "/srt"
    srt_response = requests.get(srt_endpoint, headers=headers)

    with open("yt.srt", "w") as _file:
        _file.write(srt_response.text)
    '''
    
    return transcript_output_response.json()["text"]
    
def audio(audio_file):
    audio_bytes = audio_file.read()

    st.audio(audio_bytes, format='audio/ogg')
    
    return 0

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------

# design elements
st.header('''
English-To-SQL
''')

side_bar = ['None', 'Speech-to-SQL', 'English-to-SQL']
choice = st.sidebar.selectbox("Menu", side_bar)

if choice==side_bar[1]:
    url = "https://online-voice-recorder.com/"
    st.write("You can use this link to record an audio [link](%s)" % url)
    
    uploaded_files = st.file_uploader("Choose a .mp3 file", accept_multiple_files=True)
    
    for uploaded_file in uploaded_files:
        audio(uploaded_file)
        
    if st.button(label="Transcribe .mp3 file"):
        try:
            transcribed = transcribe(uploaded_files)
