#!/usr/bin/env python
# coding: utf-8

# # Cluster Analysis

# In[1]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans


# In[2]:


df = pd.read_csv("./data/agg_all.csv", index_col=0)
df


# 100 has most studies, 95 has 1 and 150 in between.
# if i do totals of each picked and amount won for each dataset and normalise the amount of number of trials and decks picked over total picks.
# 
# could also look at if its really after 50th trial they start to go one way or the other.

# In[3]:


plt.scatter(df['Total'], df['A'])
# plt.xlim(-180,180)
# plt.ylim(-90,90)
plt.show()


# In[4]:


plt.scatter(df['Total'], df['B'])
# plt.xlim(-180,180)
# plt.ylim(-90,90)
plt.show()


# In[5]:


plt.scatter(df['Total'], df['C'])
# plt.xlim(-180,180)
# plt.ylim(-90,90)
plt.show()


# In[6]:


plt.scatter(df['Total'], df['D'])
# plt.xlim(-180,180)
# plt.ylim(-90,90)
plt.show()


# In[7]:


plt.scatter(df['A'], df['B'])
plt.show()


# In[8]:


x = df.iloc[:,:2] # 1t for rows and second for columns
x


# In[9]:


kmeans = KMeans(3)
kmeans.fit(x)


# In[10]:


identified_clusters = kmeans.fit_predict(x)
identified_clusters


# In[11]:


data_with_clusters = df.copy()
data_with_clusters['Clusters'] = identified_clusters 
plt.scatter(data_with_clusters['Total'],data_with_clusters['A'],c=data_with_clusters['Clusters'],cmap='rainbow')


# In[12]:


y = df.iloc[:,[0,2]] # 1t for rows and second for columns
y


# In[13]:


kmeans = KMeans(3)
kmeans.fit(x)
data_with_clusters = df.copy()
data_with_clusters['Clusters'] = identified_clusters 
plt.scatter(data_with_clusters['Total'],data_with_clusters['B'],c=data_with_clusters['Clusters'],cmap='rainbow')


# In[17]:




