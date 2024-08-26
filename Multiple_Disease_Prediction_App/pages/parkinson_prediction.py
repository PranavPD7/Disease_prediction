# -*- coding: utf-8 -*-
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
import brain_tumor_detection,heart_disease_prediction,llm_chatbot,parkinson_prediction
def app():
        st.title("Parkinson prediction using ml")
        col1,col2,col3,col4,col5,col6=st.columns(6)
        sc1=pickle.load(open("C:/Users/HP/Desktop/multiple_disease_prediction/scaler1.pkl",'rb'))
        parkinson_model=pickle.load(open("C:/Users/HP/Desktop/multiple_disease_prediction/parkinson.sav",'rb'))
        st.session_state.chat_history=[]
        with col1:
            name=st.text_input("name")
            MDVPFo=st.text_input("MDVP:Fo(Hz)")
            MDVPFhi=st.text_input("MDVP:Fhi(Hz)")
            MDVPFlo=st.text_input("MDVP:Flo(Hz)")
        with col2:
            MDVPJitter=st.text_input(" MDVP:Jitter(%)")
            MDVPJitter=st.text_input(" MDVP:Jitter(Abs)")
            MDVPRAP=st.text_input(" MDVP:RAP")
            MDVPPPQ=st.text_input(" MDVP:PPQ")
        with col3:
            JitterDDP=st.text_input(" Jitter:DDP")
            MDVPShimmer=st.text_input(" MDVP:Shimmer")
            MDVPShimmer1=st.text_input("MDVP:Shimmer(dB")
            ShimmerAPQ3=st.text_input("Shimmer:APQ3")
        with col4:
            ShimmerAPQ5=st.text_input("Shimmer:APQ5")
            MDVPAPQ=st.text_input(" MDVP:APQ")
            ShimmerDDA=st.text_input(" Shimmer:DDA")
            NHR=st.text_input("   NHR")
        with col5:
            HNR=st.text_input(" HNR")
            RPDE=st.text_input(" RPDE")
            DFA=st.text_input("DFA")
            spread1=st.text_input("spread1")
        with col6: 
            spread2=st.text_input("spread2")
            D2=st.text_input("D2")
            PPE=st.text_input("PPE")
        diag=''
        if 'consult_for_parkinson' not in  st.session_state:
            st.session_state.consult_for_parkinson=False
        if st.button('parkinson disease result'):
            pred=parkinson_model.predict(sc1.transform([[MDVPFo,MDVPFhi,MDVPFlo,MDVPJitter,MDVPJitter,MDVPRAP,MDVPPPQ,JitterDDP,MDVPShimmer,MDVPShimmer1,ShimmerAPQ3,ShimmerAPQ5,MDVPAPQ,ShimmerDDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]]))
            if pred[0]==0:
                diag='no parkinson'
            else:
                diag='parkinson positive'
                st.session_state.consult_for_parkinson=True
            st.success(diag)
            if  st.session_state.consult_for_parkinson:
                
                st.title('Doctor Advice')
                
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []

                for message in st.session_state.chat_history:
                    with st.markdown(message['role']):
                        st.markdown(message['content'])

              
                for message in st.session_state.chat_history:
                   with st.chat_message(message['role']):
                       st.markdown(message['content'])

                user_prompt = st.chat_input("Enter message:")
                if user_prompt:
                    st.chat_message('user').markdown(user_prompt)
                    st.session_state.chat_history.append({'role': 'user', 'content': user_prompt})

                    conversations = ' '.join([msg['content'] for msg in st.session_state.chat_history if msg['role'] == 'user'])

                    client = Client("https://cf9a3f3df62ed72103.gradio.live")

                    result = client.predict(
                        conversations,
                        api_name="/predict"
                    )
                    st.session_state.chat_history.append({'role': 'assistant', 'content': result})

                    with st.chat_message('assistant'):
                        st.markdown(result)
