from flask import Flask,render_template, request, jsonify
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import pandas as pd
# from transformers import pipeline
import streamlit as st


data = pd.read_csv('train.csv')
model = load_model('chatbotClassificationModel.h5')
tokenizer = pickle.load(open('chatbotTokinizer.pkl','rb'))
le = pickle.load(open('chatbotLabelencoder.pkl','rb'))

# chatbot = pipeline("text-generation", model="facebook/bart-base")

def chat(user_input):
    if user_input.lower() == 'exit':
        st.write("الي اللقاء سعيد بخدمتك")
    else:
        text = tokenizer.texts_to_sequences([user_input])
        text = pad_sequences(text,maxlen=7)
        result = model.predict(text)
        output_class = np.argmax(result)
        output = le.inverse_transform(np.array([output_class]))
        
        try:
            mask = data['label'] == output[0]
            if mask.any():
                random_index = np.random.choice(mask[mask].index)
                random_row = data.iloc[random_index]
                st.write(f"المساعد الشخصي: {random_row['answer']}")
            else:
                st.write(f"المساعد الشخصي: عفوا لم افهم السوال")
        except ValueError as e:
            st.write("Error:", e)
            st.write(f"المساعد الشخصي: عفوا لم افهم السوال")

st.title("Medica AI Assistant")

user_input = st.text_input("استفسارك : ")

if user_input:
    chat(user_input)
