#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


choice_100 = pd.read_csv("./data/choice_100.csv")
choice_150 = pd.read_csv("./data/choice_150.csv")
choice_95 = pd.read_csv("./data/choice_95.csv")
index_100 = pd.read_csv("./data/index_100.csv")
index_150 = pd.read_csv("./data/index_150.csv")
index_95 = pd.read_csv("./data/index_95.csv")
lo_100 = pd.read_csv("./data/lo_100.csv")
lo_150 = pd.read_csv("./data/lo_150.csv")
lo_95 = pd.read_csv("./data/lo_95.csv")
wi_100 = pd.read_csv("./data/wi_100.csv")
wi_150 = pd.read_csv("./data/wi_150.csv")
wi_95 = pd.read_csv("./data/wi_95.csv")


# I used .head() to see what each type of dataframe looked like

# In[3]:


print(choice_100.head())
print(index_100.head())
print(lo_100.head())
print(wi_100.head())


# # Part 1
# First I will go through each individual dataframe and see if there are any obvious errors, data type differences and and nulls that should be changed.

# In[4]:


print(choice_95.info())
print(choice_100.info())
print(choice_150.info())


# In[5]:


print(index_95.info())
print(index_100.info())
print(index_150.info())


# In[6]:


print(lo_95.info())
print(lo_100.info())
print(lo_150.info())


# In[7]:


print(wi_95.info())
print(wi_100.info())
print(wi_150.info())


# All columns are non-null and of the same type (int64), so I did not need to change any datatypes.

# # Part 2
# Now I will merge the datasets to include choice, amount won/lost and index.

# In[8]:


total_95 = wi_95.sub(lo_95)
total_95
# 500 - lo_95.iloc(1)[0][0]


# In[9]:


l_95 = lo_95.copy()
ld = {}
for i in range(1, 96):
    ld['Losses_' + str(i)] = 'Total_' + str(i)
w_95 = wi_95.copy()
wd = {}
for i in range(1, 96):
    wd['Wins_' + str(i)] = 'Total_' + str(i)
    
l_95.rename(columns=ld, inplace=True)
w_95.rename(columns=wd, inplace=True)

total_95 = w_95.add(l_95)
total_95


# In[10]:


all_95 = total_95.join(choice_95)


# In[11]:


cols = all_95.columns.tolist()
cols = sorted(cols, key = lambda x: int(x.split('_')[-1]))
all_95 = all_95[cols]
all_95


# In[12]:


index_95['Subj'] = index_95['Subj'].apply(lambda x: 'Subj_' + str(x))
index_95.set_index('Subj', inplace=True)
index_95.index.name = None
all_95 = all_95.join(index_95)


# In[13]:


l = []
for i in range(1, 96):
    l.append([i, 'Total_' + str(i)])
    l.append([i, 'Choice_' + str(i)])
l.append(['Name', 'Study'])
all_95.columns = pd.MultiIndex.from_tuples(l)
# >>> cols = pd.MultiIndex.from_tuples([("a", "b"), ("a", "c")])
# all_95.columns = all_95.columns.droplevel(0)


# In[14]:


all_95.to_csv(("./data/all_95.csv"))
print(all_95.head())


# In[ ]:




