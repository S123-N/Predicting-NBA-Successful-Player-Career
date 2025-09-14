#!/usr/bin/env python
# coding: utf-8

# In[38]:


import streamlit as st
import pandas as pd
import joblib


# In[39]:


xgb_model_nba = joblib.load('xgboost_bpm_model.pkl')
scaler = joblib.load('scaler.pkl')


# In[40]:


st.title('Basketball BPM Predictor')


# In[41]:


TOTG = st.number_input('Total Games', min_value=-1000.0, max_value=50000.0, step=0.001, format="%.3f")
TOTMP = st.number_input('Total Minutes Played', min_value=-1000.0, max_value=50000.0, step=0.001, format="%.3f")
TOTPTS = st.number_input('Total Points', min_value=-1000.0, max_value=50000.0, step=0.001, format="%.3f")
TRB = st.number_input('Total Rebounds', min_value=-1000.0, max_value=50000.0, step=0.001, format="%.3f")
TOTAST = st.number_input('Total Assists', min_value=-1000.0, max_value=50000.0, step=0.001, format="%.3f")
FGPER = st.number_input('Field Goal Percentage', min_value=-1000.0, max_value=50000.0, step=0.001, format="%.3f")
TPPER = st.number_input('Three Point Field Goal Percentage', min_value=-1000.0, max_value=50000.0, step=0.001, format="%.3f")
FTPER = st.number_input('Free Throw Percentage', min_value=-1000.0, max_value=50000.0, step=0.001, format="%.3f")
MPPG = st.number_input('Minutes Played Per Game', min_value=-1000.0, max_value=50000.0, step=0.001, format="%.3f")
PTSPG = st.number_input('Points Per Game', min_value=-1000.0, max_value=50000.0, step=0.001, format="%.3f")
TRBPG = st.number_input('Total Rebound Per Game', min_value=-1000.0, max_value=50000.0, step=0.001, format="%.3f")
ASTPG = st.number_input('Assist Per Game', min_value=-1000.0, max_value=50000.0, step=0.001, format="%.3f")


# In[42]:


user_data = pd.DataFrame({
    'TOTG': [TOTG],
    'TOTMP': [TOTMP],
    'TOTPTS': [TOTPTS],
    'TRB': [TRB],
    'TOTAST': [TOTAST],
    'FGPER': [FGPER],
    'TPPER': [TPPER],
    'FTPER': [FTPER],
    'MPPG': [MPPG],
    'PTSPG': [PTSPG],
    'TRBPG': [TRBPG],
    'ASTPG': [ASTPG]
})


# In[43]:


user_data_scaled = scaler.transform(user_data)


# In[44]:


if st.button('Predict BPM'):
    predicted_bpm = xgb_model_nba.predict(user_data_scaled)
    st.success(f'Predicted BPM: {predicted_bpm[0]:.2f}')


# In[ ]:




