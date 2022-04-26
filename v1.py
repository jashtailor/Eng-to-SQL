# IMPORTING THE NECESSARY LIBRARIES
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
