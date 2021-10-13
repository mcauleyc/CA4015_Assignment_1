#!/usr/bin/env python
# coding: utf-8

# # Data Processing
# 
# This is where I cleaned and processed the datasets that were given. I did this in three steps:
# 1. Data cleaning
# 2. Merging the Data
# 3. Aggregating the Data

# In[1]:


import pandas as pd
import numpy as np


# Firstly, I read in each dataset and used `pandas.DataFrame.head()` to get an idea of what the datasets looked like.

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


# In[3]:


print(choice_100.head())
print(index_100.head())
print(lo_100.head())
print(wi_100.head())


# ## 1. Data Cleaning
# 
# My first step was data cleaning. I went through each individual dataframe to see if there were any obvious errors, data type differences or any nulls that needed to be changed.

# In[4]:


print(choice_95.info(verbose=False))
print(choice_100.info(verbose=False))
print(choice_150.info(verbose=False))
print(index_95.info(verbose=False))
print(index_100.info(verbose=False))
print(index_150.info(verbose=False))
print(lo_95.info(verbose=False))
print(lo_100.info(verbose=False))
print(lo_150.info(verbose=False))
print(wi_95.info(verbose=False))
print(wi_100.info(verbose=False))
print(wi_150.info(verbose=False))


# In[5]:


print(choice_100.isnull().values.any())
print(choice_150.isnull().values.any())
print(choice_95.isnull().values.any())
print(index_100.isnull().values.any())
print(index_150.isnull().values.any())
print(index_95.isnull().values.any())
print(lo_100.isnull().values.any())
print(lo_150.isnull().values.any())
print(lo_95.isnull().values.any())
print(wi_100.isnull().values.any())
print(wi_150.isnull().values.any())
print(wi_95.isnull().values.any())


# All columns were non-null and of the same type (int64), so I did not need to change any datatypes. The data was already clean.

# ## 2. Merging the Datasets
# 
# In this step, I merged the datasets to include choice, amount won/lost, and index.

# I began by changing the column names in the wins and losses datasets to 'Total_{n}'. This made the merging of the two dataframes easier because I was then able to use `pandas.DataFrame.add()`to get the total amount won or lost for each choice.  

# In[6]:


ld, wd = {}, {}
for i in range(1, 96):
    ld['Losses_' + str(i)] = 'Total_' + str(i)
    wd['Wins_' + str(i)] = 'Total_' + str(i)
    
lo_95.rename(columns=ld, inplace=True)
wi_95.rename(columns=wd, inplace=True)

total_95 = wi_95.add(lo_95)
total_95


# I then joined 'total_95' with 'choice_95' so that the dataframe, 'all_95', now contains the total amount won or lost and the choice that was made for each round. The columns alternate between total and choice because it is easier to understand that way.

# In[7]:


all_95 = total_95.join(choice_95)

cols = all_95.columns.tolist()
cols = sorted(cols, key = lambda x: int(x.split('_')[-1]))
all_95 = all_95[cols]
all_95


# I changed the cells in 'index_95' to be in the form 'Subj_{}' and set that column as the index. I then joined it with 'all_95'.

# In[8]:


index_95['Subj'] = index_95['Subj'].apply(lambda x: 'Subj_' + str(x))
index_95.set_index('Subj', inplace=True)
index_95.index.name = None
all_95 = all_95.join(index_95)


# Finally, I added multilevel columns to make each trial separate. I then exported the file in csv format to the data folder in my book.

# In[9]:


l = []
for i in range(1, 96):
    l.append([i, 'Total_' + str(i)])
    l.append([i, 'Choice_' + str(i)])
l.append(['Name', 'Study'])
all_95.columns = pd.MultiIndex.from_tuples(l)

all_95.to_csv(("./data/all_95.csv"))
print(all_95.head())


# I repeated this process for the other datasets.

# In[10]:


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


# In[11]:


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


# ## 3. Aggregating the Data
# 
# Then, I aggregated the 'all' dataframes into one dataframe and normalised the data.

# I began by adding up each choice option so that I had the total number of times A, B, C and D were picked. For the purposes of this task I assumed that 1 was 'A', 2 was 'B' etc. I then added the totals so that I could see the total amount won or lost by each participant. I then added these columns as well as study to a dataframe named 'agg_95'.

# In[12]:


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
    data.append(['95_' + row[1].name, total, a, b, c, d, list(row[1])[-1]])

agg_95 = pd.DataFrame(data, columns=['Subj', 'Total', 'A', 'B', 'C', 'D', 'Study'])
agg_95


# This was then repeated for 100 and 150.

# In[13]:


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
    data.append(['100_' + row[1].name, total, a, b, c, d, list(row[1])[-1]])

agg_100 = pd.DataFrame(data, columns=['Subj','Total', 'A', 'B', 'C', 'D', 'Study'])
agg_100


# In[14]:


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
    data.append(['150_' + row[1].name, total, a, b, c, d, list(row[1])[-1]])

agg_150 = pd.DataFrame(data, columns=['Subj', 'Total', 'A', 'B', 'C', 'D', 'Study'])
agg_150


# I then normalised them by dividing each dataframe by the amount of trials it contained.

# In[15]:


agg_150[['Total', 'A', 'B', 'C', 'D']]/= 150
agg_100[['Total', 'A', 'B', 'C', 'D']]/= 100
agg_95[['Total', 'A', 'B', 'C', 'D']]/= 95


# I then added in a 'Good' column (the sum of C and D), a 'Bad' column (the sum of A and B), a column for the numeric representation of each study and a column with the payload of the study. I then saved the dataframe as a csv file in the data folder under 'agg_all.csv'.

# In[16]:


agg_all = pd.concat([agg_95, agg_100, agg_150])
agg_all.reset_index(inplace=True, drop=True)
agg_all['Bad'] = agg_all['A'] + agg_all['B']
agg_all['Good'] = agg_all['C'] + agg_all['D']

agg_all['StudyNo'] = ''   
agg_all['Payload'] = ''

stud_d = {'Fridberg': 1, 'Horstmann': 2, 'Kjome': 3, 'Maia': 4, 'Premkumar': 5, 'Steingroever2011': 6, 
    'SteingroverInPrep': 7, 'Wetzels': 8, 'Wood': 9, 'Worthy': 10}
payload_d = {'Fridberg': 1, 'Horstmann': 2, 'Kjome': 3, 'Maia': 1, 'Premkumar': 3, 'Steingroever2011': 2, 
    'SteingroverInPrep': 2, 'Wetzels': 2, 'Wood': 3, 'Worthy': 1}

for i in range(0, len(agg_all)):
    agg_all.loc[i, 'StudyNo'] = stud_d[agg_all['Study'][i]]
    agg_all.loc[i, 'Payload'] = payload_d[agg_all['Study'][i]]


agg_all.to_csv("./data/agg_all.csv")
agg_all


# The data was then ready for me to use in my cluster analysis.
