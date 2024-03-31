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
    # Generate response using the chatbot library
    
    if user_input.lower() == 'exit':
        print("الي اللقاء سعيد بخدمتك")
    else:
        text = tokenizer.texts_to_sequences([user_input])
        text = pad_sequences(text,maxlen=20)
        result = model.predict(text)
        output_class = np.argmax(result)
        output = le.inverse_transform(np.array([output_class]))
        # predicted_answer = data[data['label'] == data]['answer'].iloc[0]
        mask = data['label'] == output[0]
        if mask.any():
          random_index = np.random.choice(mask[mask].index)
          random_row = data.iloc[random_index]
          # print(random_row['answer'])
          # print(output[0])
          # print("----------------")
          # print(random_row['answer'] )
          st.write(f"المساعد الشخصي: {random_row['answer']}")
        else:
          st.write(f"المساعد الشخصي: عفوا لم افهم السوال")

st.title("Medica AI Assistant")

user_input = st.text_input("استفسارك : ")

if user_input:
    chat(user_input)
