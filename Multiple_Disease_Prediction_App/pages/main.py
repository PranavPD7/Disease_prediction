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
from websockets.legacy.client import InvalidStatusCode
import logging
import brain_tumor_detection,heart_disease_prediction,llm_chatbot,parkinson_prediction,diabetes_prediction

#from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline)

#loading saved models and transformations

sc=pickle.load(open("C:/Users/HP/Desktop/multiple_disease_prediction/scaler.pkl",'rb'))
 
diabetes_model=pickle.load(open("C:/Users/HP/Desktop/multiple_disease_prediction/diabetes_predictor.sav",'rb'))
heart_disease_model=pickle.load(open("C:/Users/HP/Desktop/multiple_disease_prediction/heart_disease.sav",'rb'))
parkinson_model=pickle.load(open("C:/Users/HP/Desktop/multiple_disease_prediction/parkinson.sav",'rb'))
tumor_detection_model=tensorflow.keras.models.load_model("C:/Users/HP/Desktop/multiple_disease_prediction/tumor_detection.h5")

class MultiApp:
    def __init__(self):
        self.apps=[]
    def add_app(self,title,function):
        self.apps.append({'title':title,'function':function})
    def run():
        with st.sidebar:
            app=option_menu("multiple disease prediction system",
                                 ['diabetes_prediction',
                                  'heart_disease_prediction',
                                  'parkinson_prediction',
                                  'brain_tumor_detection'],
                                 icons=['activity','heart','person','hospital-fill'],
                                 default_index=0)
        
        if app=='diabetes_prediction':
            st.session_state.consult_for_heart=False
            st.session_state.consult_for_parkinson=False
            diabetes_prediction.app()

            
        if app=='heart_disease_prediction':
            st.session_state_consult_dr_llm=False
            st.session_state.consult_for_parkinson=False
            heart_disease_prediction.app()
        if app=='parkinson_prediction':
            st.session_state_consult_dr_llm=False
            st.session_state.consult_for_heart=False
            
            parkinson_prediction.app()
        if app=='brain_tumor_detection':
            brain_tumor_detection.app()
        
    run()

            
                
        
                                               
    
    
    

                    
        

