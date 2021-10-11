#!/usr/bin/env python
# coding: utf-8

# # Data Processing

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

# I begin by changing the column names in the wins and losses datasets to 'Total_{n}'. This made the merging of the two dataframes easier because I was then able to use 
# ```
# pandas.DataFrame.add()
# ```
# to get the total amount won or loss for each choice.  

# In[8]:


ld, wd = {}, {}
for i in range(1, 96):
    ld['Losses_' + str(i)] = 'Total_' + str(i)
    wd['Wins_' + str(i)] = 'Total_' + str(i)
    
lo_95.rename(columns=ld, inplace=True)
wi_95.rename(columns=wd, inplace=True)

total_95 = wi_95.add(lo_95)
total_95


# I then joined all_95 with choice_95 so that the all_95 now contains the total amount won or lost and the choice that was made for each round.

# In[9]:


all_95 = total_95.join(choice_95)

cols = all_95.columns.tolist()
cols = sorted(cols, key = lambda x: int(x.split('_')[-1]))
all_95 = all_95[cols]
all_95


# I then changed the cells in index_95 to be in the form 'Subj_{}' and set that column as the index and joined it with all_95.

# In[10]:


index_95['Subj'] = index_95['Subj'].apply(lambda x: 'Subj_' + str(x))
index_95.set_index('Subj', inplace=True)
index_95.index.name = None
all_95 = all_95.join(index_95)


# Finally, I changed the order of the columns in the dataframe so that it was more easily readable and I could understand it more. I made it to go in alternating order of total won/lost and choice made, and then the study name is the final column. I also added multilevel columns to make each trial separate. I then exported the file in csv format to the data folder in my book.

# In[11]:


l = []
for i in range(1, 96):
    l.append([i, 'Total_' + str(i)])
    l.append([i, 'Choice_' + str(i)])
l.append(['Name', 'Study'])
all_95.columns = pd.MultiIndex.from_tuples(l)

all_95.to_csv(("./data/all_95.csv"))
print(all_95.head())


# I will now continue to do this for the two other dataframe amounts; 100 and 150.

# In[12]:


ld, wd = {}, {}
for i in range(1, 101):
    ld['Losses_' + str(i)] = 'Total_' + str(i)
    wd['Wins_' + str(i)] = 'Total_' + str(i)
    
lo_100.rename(columns=ld, inplace=True)
wi_100.rename(columns=wd, inplace=True)

total_100 = wi_100.add(lo_100)

all_100 = total_100.join(choice_100)

cols = all_100.columns.tolist()
cols = sorted(cols, key = lambda x: int(x.split('_')[-1]))
all_100 = all_100[cols]

index_100['Subj'] = index_100['Subj'].apply(lambda x: 'Subj_' + str(x))
index_100.set_index('Subj', inplace=True)
index_100.index.name = None
all_100 = all_100.join(index_100)

l = []
for i in range(1, 101):
    l.append([i, 'Total_' + str(i)])
    l.append([i, 'Choice_' + str(i)])
l.append(['Name', 'Study'])
all_100.columns = pd.MultiIndex.from_tuples(l)

all_100.to_csv(("./data/all_100.csv"))
print(all_100.head())


# In[13]:


ld, wd = {}, {}
for i in range(1, 151):
    ld['Losses_' + str(i)] = 'Total_' + str(i)
    wd['Wins_' + str(i)] = 'Total_' + str(i)
    
lo_150.rename(columns=ld, inplace=True)
wi_150.rename(columns=wd, inplace=True)

total_150 = wi_150.add(lo_150)

all_150 = total_150.join(choice_150)

cols = all_150.columns.tolist()
cols = sorted(cols, key = lambda x: int(x.split('_')[-1]))
all_150 = all_150[cols]

index_150['Subj'] = index_150['Subj'].apply(lambda x: 'Subj_' + str(x))
index_150.set_index('Subj', inplace=True)
index_150.index.name = None
all_150 = all_150.join(index_150)

l = []
for i in range(1, 151):
    l.append([i, 'Total_' + str(i)])
    l.append([i, 'Choice_' + str(i)])
l.append(['Name', 'Study'])
all_150.columns = pd.MultiIndex.from_tuples(l)

all_150.to_csv(("./data/all_150.csv"))
print(all_150.head())


# # Part 3
# 
# Now I will aggregate the 'all' dataframes to and normalise them.
# 

# In[14]:


data = []
for row in all_95.iterrows():
    total, a, b, c, d = 0,0,0,0,0
    for j in range(0, len(list(row[1])) - 1):
        
        i = list(row[1])[j]
        if j % 2 != 0:
            if i == 1:
                a += 1
            elif i == 2:
                b += 1
            elif i == 3:
                c += 1
            elif i == 4:
                d += 1
                
        else:
            total += i
    data.append([total, a, b, c, d, list(row[1])[-1]])

agg_95 = pd.DataFrame(data, columns=['Total', 'A', 'B', 'C', 'D', 'Study'])
agg_95


# In[15]:


data = []
for row in all_100.iterrows():
    total, a, b, c, d = 0,0,0,0,0
    for j in range(0, len(list(row[1])) - 1):
        
        i = list(row[1])[j]
        if j % 2 != 0:
            if i == 1:
                a += 1
            elif i == 2:
                b += 1
            elif i == 3:
                c += 1
            elif i == 4:
                d += 1
                
        else:
            total += i
    data.append([total, a, b, c, d, list(row[1])[-1]])

agg_100 = pd.DataFrame(data, columns=['Total', 'A', 'B', 'C', 'D', 'Study'])
agg_100


# In[16]:


data = []
for row in all_150.iterrows():
    total, a, b, c, d = 0,0,0,0,0
    for j in range(0, len(list(row[1])) - 1):
        
        i = list(row[1])[j]
        if j % 2 != 0:
            if i == 1:
                a += 1
            elif i == 2:
                b += 1
            elif i == 3:
                c += 1
            elif i == 4:
                d += 1
                
        else:
            total += i
    data.append([total, a, b, c, d, list(row[1])[-1]])

agg_150 = pd.DataFrame(data, columns=['Total', 'A', 'B', 'C', 'D', 'Study'])
agg_150


# In[17]:


agg_150[['Total', 'A', 'B', 'C', 'D']]/= 150
agg_100[['Total', 'A', 'B', 'C', 'D']]/= 150
agg_95[['Total', 'A', 'B', 'C', 'D']]/= 150


# In[18]:


agg_all = pd.concat([agg_95, agg_100, agg_150])
agg_all.reset_index(drop=True)

agg_all.to_csv(("./data/agg_all.csv"))
agg_all

