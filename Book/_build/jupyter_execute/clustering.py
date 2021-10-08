#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans


# In[2]:


choice = pd.read_csv("./IGTdataSteingroever2014/choice_100.csv")
choice


# In[18]:


loss = pd.read_csv("./IGTdataSteingroever2014/lo_100.csv")
win = pd.read_csv("./IGTdataSteingroever2014/wi_100.csv")
loss


# In[28]:


plt.scatter(win, loss)
# plt.xlim(-180,180)
# plt.ylim(-90,90)
plt.show()


# In[29]:


x = data.iloc[:,1:3] # 1t for rows and second for columns
x


# In[31]:


kmeans = KMeans(3)
kmeans.fit(x)


# In[33]:


identified_clusters = kmeans.fit_predict(x)
identified_clusters


# In[35]:


data_with_clusters = data.copy()
data_with_clusters['Clusters'] = identified_clusters 
plt.scatter(data_with_clusters['Choice_2'],data_with_clusters['Choice_3'],c=data_with_clusters['Clusters'],cmap='rainbow')

