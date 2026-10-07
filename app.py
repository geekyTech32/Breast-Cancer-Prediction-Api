#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from tensorflow.keras.applications import efficientnet


# In[ ]:


# loading model
model = load_model("Classification_model.keras")
# backend logic
def classification_predict(img):
    # preprocessing img 
    img_array = efficientnet.preprocess_input(img)
    # expanding array 
    img_array = np.expand_dims(img_array, axis=0)
    # predicting array
    result = model.predict(img_array)
    return result 

