# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 18:13:19 2024

@author: HP
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import json
from PIL import Image
import cv2
import numpy as np
import os
import json
import tensorflow 
from gradio_client import Client
import gradio_client as gc
import brain_tumor_detection,heart_disease_prediction,llm_chatbot,parkinson_prediction

#from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline)

#loading saved models and transformations
def app():
    sc=pickle.load(open("C:/Users/HP/Desktop/multiple_disease_prediction/scaler.pkl",'rb'))
     
    diabetes_model=pickle.load(open("C:/Users/HP/Desktop/multiple_disease_prediction/diabetes_predictor.sav",'rb'))
    heart_disease_model=pickle.load(open("C:/Users/HP/Desktop/multiple_disease_prediction/heart_disease.sav",'rb'))
    parkinson_model=pickle.load(open("C:/Users/HP/Desktop/multiple_disease_prediction/parkinson.sav",'rb'))
    tumor_detection_model=tensorflow.keras.models.load_model("C:/Users/HP/Desktop/multiple_disease_prediction/tumor_detection.h5")
    st.session_state.consult_for_heart=False
    st.session_state.consult_for_parkinson=False
    #sidebar for navigation
    
    #diabetes prediction page
    st.title("Diabetes prediction using ml")
    col1,col2,col3=st.columns(3)
    
    
    
    with col1:    
        Pregnancies=st.text_input("Number of pregnancies")
        Glucose=st.text_input("glucose value")
        BloodPressure=st.text_input("blood pressure")
    with col2: 
        SkinThickness=st.text_input("skin thickness value")
        Insulin=st.text_input("insulin levels")
        BMI=st.text_input("bmi index")
    with col3:
        DiabetesPedigreeFunction=st.text_input("diabetes pedigree function value")
        Age=st.text_input("age of person")
    if 'consult_dr_llm' not in st.session_state:
        st.session_state.consult_dr_llm = False
    diag=''
    if st.button("Diabetics test result"):
        prediction=diabetes_model.predict(sc.transform([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]))
    
        if prediction[0]==1:
            diag="person is diabetic"
            st.session_state.consult_dr_llm=True
            
        else:
            diag="person not diabetic"
        st.success(diag)
    if st.session_state.consult_dr_llm :
        
        conversations=''
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history=[]
            conversations=''
        st.title('doctor advice')
        for message in st.session_state.chat_history:
            with st.markdown(message['role']):
                st.markdown(message['content'])
       
        
      
        
        
        #def add_auth_headers(headers=None):
            #if headers is None:
             #   headers = {}
            #api_key = os.getenv('API_KEY')
           # if api_key:
            #    headers['Authorization'] = f'Bearer {api_key}'
            #return headers
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Main interaction logic
        for message in st.session_state.chat_history:
           with st.chat_message(message['role']):
               st.markdown(message['content'])
        user_prompt = st.chat_input("enter message:")
        if user_prompt:
            st.chat_message('user').markdown(user_prompt)
            st.session_state.chat_history.append({'role': 'user', 'content': user_prompt})
            
            conversations = ' '.join([msg['content'] for msg in st.session_state.chat_history if msg['role'] == 'user'])
            
            
           # headers = add_auth_headers()
            
            
            client = Client("https://799d4b17baa2782337.gradio.live")

             
            result = client.predict(
            		conversations,
            		api_name="/predict"
            )
            st.session_state.chat_history.append({'role':'assistant','content':result})
            with st.chat_message('assistant'):
                st.markdown(result)
               