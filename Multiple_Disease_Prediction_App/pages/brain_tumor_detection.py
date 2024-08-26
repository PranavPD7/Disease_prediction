import pickle
import streamlit as st
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
def app():
    st.title('Brain tumor detector')
    tumor_detection_model=tensorflow.keras.models.load_model("C:/Users/HP/Desktop/multiple_disease_prediction/tumor_detection.h5")

    upload_img=st.file_uploader("upload an image",type=['jpg','jpeg','png'])
    if upload_img is not None:
        im=Image.open(upload_img)
        im=np.array(im)
        im=cv2.resize(im,(64,64))
        image=im/255.0
        image=np.expand_dims(image,axis=0)
        col1,col2=st.columns(2)
        with col1:
            st.image(cv2.resize(im,(500,500)))
        with col2:
            res='a'
            if st.button('classify'):
                
                prediction=tumor_detection_model.predict(image)
                if np.argmax(prediction)==0:
                    res='type A Glioma'
                if np.argmax(prediction)==1:
                    res='type B Meningioma'
                if np.argmax(prediction)==2:
                    res='No tumor detected'
                if np.argmax(prediction)==3:
                    res='type C Pituitary'
                st.success(res)

