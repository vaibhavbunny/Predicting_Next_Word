import pickle
import streamlit as st 
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable CUDA if present
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # Suppress some logs

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # Disable Metal GPU on Apple Silicon


## Load the trained Model
model = load_model('next_word_lstm.h5')

## load the tokenizer 
with open('tokenizer.pickle','rb') as file:
    tokenizer = pickle.load(file)

def predict_next_word(model, tokenizer, text, max_sequence_len):
    try:
        token_list = tokenizer.texts_to_sequences([text])[0]
        if len(token_list) >= max_sequence_len:
            token_list = token_list[-(max_sequence_len - 1):]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')

        predicted = model.predict(token_list, verbose=0)

        predicted_word_index = np.argmax(predicted, axis=1)[0]

        reverse_word_index = dict((index, word) for word, index in tokenizer.word_index.items())
        return reverse_word_index.get(predicted_word_index, "Unknown")
    except Exception as e:
        return f"Error inside function: {e}"



## streamlit app
st.title("Next Word Prediction with LSTM and EarlyStopping")
input_text = st.text_input("Enter the sequence of words: ","To be or not to be")

if st.button("Predict Next Word"):
    try:
        max_sequence_len = model.input_shape[1] + 1
        
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        st.success(f"Next Word: {next_word}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
