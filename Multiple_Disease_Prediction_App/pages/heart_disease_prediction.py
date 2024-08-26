import pickle
import streamlit as st
import numpy as np
import os
from gradio_client import Client

def app():
    # Load models and scaler
    sc = pickle.load(open("C:/Users/HP/Desktop/multiple_disease_prediction/scaler.pkl", 'rb'))
    diabetes_model = pickle.load(open("C:/Users/HP/Desktop/multiple_disease_prediction/diabetes_predictor.sav", 'rb'))
    heart_disease_model = pickle.load(open("C:/Users/HP/Desktop/multiple_disease_prediction/heart_disease.sav", 'rb'))
    st.session_state.consult_dr_llm=False
    st.session_state.consult_for_parkinson=False
    st.session_state.chat_history = []
    
    # Heart Disease Prediction
    st.title("Heart Disease prediction using ML")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        age = st.text_input("Age")
        sex = st.text_input("Sex")
        cp = st.text_input("CP")
    with col2:
        trestbps = st.text_input("Trestbps")
        chol = st.text_input("Cholesterol")
        fbs = st.text_input("FBS")
    with col3:
        restecg = st.text_input("Rest ECG")
        thalach = st.text_input("Thalach")
        exang = st.text_input("Exang")
    with col4:
        oldpeak = st.text_input("Old Peak")
        slope = st.text_input("Slope")
        ca = st.text_input("CA")
        thal = st.text_input("Thal")

    if st.button("Predict Heart Disease Result"):
        try:
            input_data = np.array([[float(age), float(sex), float(cp), float(trestbps), float(chol), float(fbs),
                                    float(restecg), float(thalach), float(exang), float(oldpeak), float(slope),
                                    float(ca), float(thal)]])
            prediction = heart_disease_model.predict(input_data)
            output = ''

            if 'consult_for_heart' not in st.session_state:
                st.session_state.consult_for_heart = False

            if prediction[0] == 0:
                output = 'No heart disease'
            else:
                output = 'Heart disease detected'
                
                st.session_state.consult_for_heart = True
            
            st.success(output)

        except ValueError:
            st.error("Please enter valid numeric values")

    if st.session_state.consult_for_heart:
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
