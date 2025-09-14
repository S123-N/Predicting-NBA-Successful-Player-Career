#!/usr/bin/env python
# coding: utf-8

# In[108]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib


# In[109]:


NBA = pd.read_csv("C:\\Users\\Sajal Nandi\\Downloads\\SN_30111888_Project\\30111888_NBA-Draft-Data.csv")


# In[110]:


empty_cells = NBA.isna().sum() + (NBA == '').sum()


# In[111]:


NBA.replace('', np.nan, inplace=True)


# In[112]:


NBA.fillna(0, inplace=True)


# In[113]:


X_NBA_Original = NBA.drop(["Rk", "Pk", "Draft Year", "Team", "Player Name", "College", "Year", "WS", "WS/48", "VORP", "BPM"], axis=1)


# In[114]:


X_NBA_ONE = pd.DataFrame(X_NBA_Original)


# In[115]:


X_NBA_ONE = X_NBA_ONE.drop(["Player ID"], axis=1)


# In[116]:


y_nba_o = NBA["BPM"]


# In[117]:


y_nba_o = pd.DataFrame(y_nba_o)


# In[118]:


y_NBA_ONE = pd.DataFrame(y_nba_o)


# In[119]:


X_NBA_ONE_train, X_NBA_ONE_test, y_NBA_ONE_train, y_NBA_ONE_test = train_test_split(X_NBA_ONE, y_NBA_ONE, test_size=0.2, random_state=42)


# In[120]:


scaler = StandardScaler()
X_train_scaled_nba_boost = scaler.fit_transform(X_NBA_ONE_train)
X_test_scaled_nba_boost = scaler.transform(X_NBA_ONE_test)


# In[121]:


xgb_model_nba = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    random_state=42
)


# In[122]:


xgb_model_nba.fit(X_train_scaled_nba_boost, y_NBA_ONE_train)


# In[123]:


joblib.dump(xgb_model_nba, 'xgboost_bpm_model.pkl')


# In[124]:


joblib.dump(scaler, 'scaler.pkl')


# In[125]:


print("Model and scaler saved successfully!")


# In[ ]:




