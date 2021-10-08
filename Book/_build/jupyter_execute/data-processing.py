#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


choice_100 = pd.read_csv("./IGTdataSteingroever2014/choice_100.csv")
choice_150 = pd.read_csv("./IGTdataSteingroever2014/choice_150.csv")
choice_95 = pd.read_csv("./IGTdataSteingroever2014/choice_95.csv")
index_100 = pd.read_csv("./IGTdataSteingroever2014/index_100.csv")
index_150 = pd.read_csv("./IGTdataSteingroever2014/index_100.csv")
index_95 = pd.read_csv("./IGTdataSteingroever2014/index_95.csv")
lo_100 = pd.read_csv("./IGTdataSteingroever2014/lo_100.csv")
lo_150 = pd.read_csv("./IGTdataSteingroever2014/lo_100.csv")
lo_95 = pd.read_csv("./IGTdataSteingroever2014/lo_100.csv")
wi_100 = pd.read_csv("./IGTdataSteingroever2014/wi_100.csv")
wi_150 = pd.read_csv("./IGTdataSteingroever2014/wi_100.csv")
wi_95 = pd.read_csv("./IGTdataSteingroever2014/wi_100.csv")


# I used .head() to see what each type of dataframe looked like

# In[9]:


print(choice_100.head())
print(index_100.head())
print(lo_100.head())
print(wi_100.head())


# # Part 1
# First I will go through each individual dataframe and see if there are any obvious errors, data type differences and and nulls that should be changed.

# In[22]:


print(choice_95.info())
print(choice_100.info())
print(choice_150.info())


# In[24]:


print(index_95.info())
print(index_100.info())
print(index_150.info())


# In[23]:


print(lo_95.info())
print(lo_100.info())
print(lo_150.info())


# In[25]:


print(wi_95.info())
print(wi_100.info())
print(wi_150.info())


# All columns are non-null and of the same type (int64), so I did not need to change any datatypes.

# # Part 2
# Now I will merge the datasets to include choice, amount won/lost and index.

# In[29]:


df = choice_100.merge(index_100, on )


# In[ ]:




