#!/usr/bin/env python
# coding: utf-8

# Task - 1A: Using abt small.csv and buy small.csv, implement the linkage between the
# two data sets.
# Your code for this question is to be contained in a single Python le called task1a.py and
# produce a single csv le task1a.csv containing the following two column headings:
# idAbt,idBuy
# Each row in the datale must contain a pair of matched products. For example, if your
# algorithm only matched product 10102 from the Abt dataset with product
# 203897877 from the Buy dataset your output task1a.csv would be as follows:
# idAbt, idBuy
# 10102,203897877

# In[1]:


import pandas as pd
import re
import math
import csv
from fuzzywuzzy import fuzz 
import numpy as np


# In[32]:


# Replace data
abt_small = pd.read_csv('abt_small.csv', usecols=['idABT', 'name'],
                      encoding='ISO-8859-1', dtype={
    'idABT': str,
    'name': str,
    })


buy_small = pd.read_csv('buy_small.csv', usecols=['idBuy', 'name'],
                      encoding='ISO-8859-1', dtype={
    'idBuy': str,
    'name': str,
    })

# rename columns
abt_small.rename(columns={
    'idABT': 'idAbt',
    'name': 'ABTName'}, inplace=True)
buy_small.rename(columns={
    'idBuy': 'idBuy',
    'name': 'BuyName'}, inplace=True)

abt_small['tmp'] = buy_small['tmp'] = 1


# ### 1a New Algorithm: match serial number first

# In[33]:


def getCode(str):
    str = str.replace('-', '');
    code = re.findall(r'((?:[a-zA-Z]+[0-9]|[0-9]+[a-zA-Z])[a-zA-Z0-9]*)', str)
    #code = re.findall(r'((?:[a-zA-Z]+[0-9]|[0-9]+)[a-zA-Z0-9]*)', str)
    #code = re.findall(r'\w+', str)
    return code 


# In[34]:


s1 = getCode('OmniMount 3-Shelf Large-Component Tower - G303G')
s2 = getCode("OmniMount G-123 3-Shelf Large-Component Tower - G303G")
print(s1)
print(s2)
print(fuzz.token_set_ratio(s1,s2))


# In[35]:


# Replace data
abt_small = pd.read_csv('abt_small.csv', usecols=['idABT', 'name', 'description'],
                      encoding='ISO-8859-1', dtype={
    'idABT': str,
    'name': str,
    'description': str
    })


buy_small = pd.read_csv('buy_small.csv', usecols=['idBuy', 'name'],
                      encoding='ISO-8859-1', dtype={
    'idBuy': str,
    'name': str,
    })

# rename columns
abt_small.rename(columns={
    'idABT': 'idAbt',
    'name': 'AbtName',
    'description' : 'des'}, inplace=True)
buy_small.rename(columns={
    'idBuy': 'idBuy',
    'name': 'BuyName'}, inplace=True)

# remove '/' in description
abt_small['codeAbt'] = abt_small['des'].apply(lambda x: getCode(x))
buy_small['codeBuy'] = buy_small['BuyName'].apply(lambda x: getCode(x))
# add column tmp as a key for further cross merge

abt_small['tmp'] = buy_small['tmp'] = 1


# In[45]:


with open('task1a.csv', 'w', newline='') as fp:
    writer = csv.writer(fp)
    writer.writerow(['idAbt','idBuy', 'code_match_ratio'])
    abt_nrows = abt_small.shape[0]
    # modified from cross merge algorithm 
    for i in range(abt_nrows):
        merge = pd.merge(abt_small.iloc[[i]], buy_small, how='outer',
                        on=['tmp']).drop(columns='tmp')
        for j in range(merge.shape[0]):
            Str1 = merge.at[j, 'des']
            Str2 = merge.at[j, 'BuyName']
            Str3 = merge.at[j, 'codeAbt']
            Str4 = merge.at[j, 'codeBuy']

            Token_Set_Ratio = fuzz.token_set_ratio(Str1.lower(),Str2.lower())
            code_match_ratio = fuzz.token_set_ratio(Str3,Str4)
            if code_match_ratio >= 80 or Token_Set_Ratio >= 70:
                writer.writerow([merge.at[j, 'idAbt'], merge.at[j, 'idBuy'], code_match_ratio])
                


# In[46]:


df = pd.read_csv('task1a.csv', usecols=['idAbt', 'idBuy', 'code_match_ratio'],
                      encoding='ISO-8859-1', dtype={
    'idAbt': str,
    'ABTName':str,
    'idBuy': str,
    'BuyName':str,
    })

df = df.sort_values(["idAbt", "code_match_ratio"], ascending = (True, False))
df = df.drop_duplicates(subset='idAbt', keep="first")

df = df.sort_values(["idBuy", "code_match_ratio"], ascending = (True, False))
df = df.drop_duplicates(subset='idBuy', keep="first")

df = df.drop(axis = 1, columns='code_match_ratio')
df.to_csv('task1a.csv',index = False)


# ### Evaluate B1 algorithim

# In[47]:


# caculate recall ratio
truth = pd.read_csv('abt_buy_truth_small.csv', encoding='ISO-8859-1')
print('length of truth')
print(truth.shape[0])
result = pd.read_csv('task1a.csv', encoding='ISO-8859-1')
print('length of result')
print(result.shape[0])
tp = pd.merge(truth, result,how='inner', left_on=['idAbt','idBuy'], right_on = ['idAbt','idBuy'])
tp.to_csv()

print('length of true positive')
print(tp.shape[0])


a = pd.read_csv('abt_small.csv', usecols=['idABT', 'description'], encoding='ISO-8859-1')
b = pd.read_csv('buy_small.csv', usecols=['idBuy', 'name'], encoding='ISO-8859-1')



print('recall ratio')
print(tp.shape[0]/truth.shape[0])
fail_recall = pd.concat([tp,truth]).drop_duplicates(keep=False)

#fail_recall.to_csv('fail_recall.csv', index = False)


print('precision ratio')
print(tp.shape[0]/result.shape[0])
fail_precise = pd.concat([tp,result]).drop_duplicates(keep=False)
#fail_precise.to_csv('fail_precise.csv', index = False)


# ### Task - 1B: Implement a blocking method for the linkage of the abt.csv and buy.csv data
# sets.
# Your code is be contained in a single Python le called task1b.py and must produce two
# csv les abt blocks.csv and buy blocks.csv, each containing the following two column
# headings:
# block_key, product_id
# The product id eld corresponds to the idAbt and idBuy of the abt.csv and buy.csv les
# respectively. Each row in the output les matches a product to a block. For example, if your
# algorithm placed product 10102 from the Abt dataset in blocks with block keys x & y, your
# abt blocks.csv would be as follows:
# block_key, product_id
# x,10102
# y,10102

# In[48]:


import pandas as pd
import re
import math
import csv
from fuzzywuzzy import fuzz 
import numpy as np
from more_itertools import locate


# In[49]:


def allocate(allocator, product):
    ''' 
        assign blocks based on manufacturer names
    
    '''
    for i in range(len(allocator)):
        if allocator[i] in str(product):
            return i


# In[50]:


def get_manu(name):
    if type(name) == str:
        return name.split()[0].lower()
    return ''


# In[51]:



# read data
abt = pd.read_csv('abt.csv', usecols=['idABT', 'name'],
                      encoding='ISO-8859-1', dtype={
    'idABT': str,
    'name': str,
    })


buy = pd.read_csv('buy.csv', usecols=['idBuy', 'name', 'manufacturer'],
                      encoding='ISO-8859-1', dtype={
    'idBuy': str,
    'name': str,
    })

# rename columns
abt.rename(columns={
    'idABT': 'idAbt',
    'name': 'ABTName'}, inplace=True)
buy.rename(columns={
    'idBuy': 'idBuy',
    'name': 'BuyName',
    'manufacturer': 'manufacturer'}, inplace=True)


# In[52]:


abt['manuABT'] = abt['ABTName'].apply(lambda x : get_manu(x))

buy['manuBuy'] = buy['manufacturer'].astype(str) + ' ' +  buy['BuyName'].astype(str)

buy['manuBuy'] = buy['manuBuy'].apply(lambda x : get_manu(x))

allocator = abt['manuABT'].append(buy['manuBuy'])
allocator = list(set(allocator))


# In[53]:


with open('abt_blocks.csv', 'w', newline='') as fp:
    writer = csv.writer(fp)
    writer.writerow(['block_key', 'product_id'])
    
    for i in range(abt.shape[0]):
        block = allocate(allocator, abt.at[i, 'ABTName'])
        writer.writerow([block, abt.loc[i, 'idAbt']])


# In[489]:


with open('buy_blocks.csv', 'w', newline='') as fp:
    writer = csv.writer(fp)
    writer.writerow(['block_key', 'product_id'])

    for i in range(buy.shape[0]):
        block = allocate(allocator, buy.at[i, 'BuyName'])
        writer.writerow([block, buy.loc[i, 'idBuy']])


# ### Now evaluate pair completeness and reduction ratio.

# In[443]:


truth = pd.read_csv('abt_buy_truth.csv', encoding='ISO-8859-1')
abt = pd.read_csv('abt_blocks.csv', encoding='ISO-8859-1').rename(columns=
                                            {'product_id': 'idAbt'})
buy = pd.read_csv('buy_blocks.csv', encoding='ISO-8859-1').rename(columns=
                                            {'product_id': 'idBuy'})
pair_completeness = pd.merge(pd.merge(abt, truth), pd.merge(buy, truth),
                             how='inner').shape[0]/truth.shape[0]

n = pd.read_csv('abt.csv', encoding='ISO-8859-1').shape[0] * pd.read_csv('buy_small.csv', encoding='ISO-8859-1').shape[0]

abt_block = abt['block_key'].values.tolist()
buy_block = buy['block_key'].values.tolist()

tp_plus_fp = 0

for i in range(len(allocator)):
    tp_plus_fp += abt_block.count(i) * buy_block.count(i)

reduction_ratio = 1 - tp_plus_fp / n


# In[444]:


print('Pair completeness: {0:.3f}\nReduction ratio: {1:.3f}'.format(pair_completeness, reduction_ratio))


# ### Task2a

# we wish to understand how the information can be used to
# predict average lifespan in dierent countries. To this end, we have provided the world.csv
# le, which contains some of the World Development Indicators for each country and the
# life.csv le containing information about the average lifespan for each country (based on
# data from the World Health Organization) [2]. Each data le also contains a country name,
# country code and year as identiers for each record. These may be used to link the two
# datasets but should not be considered features.

# Compare the performance of the following 3 classication algorithms: k-NN (k=3
# and k=7) and Decision tree (with a maximum depth of 3) on the provided data. You may use
# sklearn's KNeighborsClassier and DecisionTreeClassier functions for this task.

# In[132]:


import pandas as pd
import numpy as np
import csv
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing


# ### Organise and verify your data: Before you begin you should ensure that your dataset is
# sorted in ascending alphabetical order by country and that any countries not present in both
# world.csv and life.csv are discarded.

# In[133]:


world = pd.read_csv('world.csv', dtype={'Time': str}, encoding='ISO-8859-1')
life = pd.read_csv('life.csv', usecols=['Country Code', 'Life expectancy at birth (years)'],
                   dtype={'Year': str}, encoding='ISO-8859-1')


df = pd.merge(world, life, how='inner', on=['Country Code'])
df.columns=df.columns.str.strip()
df = df.sort_values(["Country Code"], ascending = True)
data = df[list(world.columns)[3:]]

classlabel = df['Life expectancy at birth (years)']
x_train, x_test, y_train, y_test = train_test_split(data, classlabel, train_size=0.7, test_size=0.3, random_state=200)


# In[134]:


# preprocessing data
train_median = {col: x_train[col][x_train[col]!='..'].astype(float).median()
                  for col in list(x_train.columns)}
for col in train_median.keys():
    for i in range(df.shape[0]):
        if i in x_train.index and x_train.at[i, col] == '..':
            x_train.at[i, col] = train_median[col]
            
with open('task2a.csv', 'w', newline='') as fp:
    writer = csv.writer(fp)
    writer.writerow(['feature', 'median', 'mean', 'variance'])
    for col in train_median.keys():
        writer.writerow([col, train_median[col], x_train[col].astype(float).mean(), x_train[col].astype(float).var()])
test_median = {col: x_test[col][x_test[col]!='..'].astype(float).median()
                  for col in list(x_test.columns)}
for col in test_median.keys():
    for i in range(df.shape[0]):
        if i in x_test.index and x_test.at[i, col] == '..':
            x_test.at[i, col] = test_median[col]
            
            

scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# In[135]:


# Decision tree (with a maximum depth of 3)
dt = DecisionTreeClassifier(criterion="entropy",random_state=200, max_depth=6)
dt.fit(x_train, y_train)
yd_pred=dt.predict(x_test)
print('Accuracy of decision tree: {0}'.format(np.around(accuracy_score(y_test, yd_pred), decimals=3)))


# In[136]:


# fit a K-nearest neighbor classifier with K=3
knn3 = neighbors.KNeighborsClassifier(n_neighbors=3)
knn3.fit(x_train, y_train)
y3_pred=knn3.predict(x_test)
print('Accuracy of k-nn (k=3): {0}'.format(np.around(accuracy_score(y_test, y3_pred), decimals=3)))


# In[137]:


# fit a K-nearest neighbor classifier with K=7
knn7 = neighbors.KNeighborsClassifier(n_neighbors=7)
knn7.fit(x_train, y_train)
y7_pred=knn7.predict(x_test)
print('Accuracy of k-nn (k=7): {0}'.format(np.around(accuracy_score(y_test, y7_pred), decimals=3)))


# ### Task 2b 

# In[154]:


world = pd.read_csv('world.csv', dtype={'Time': str}, encoding='ISO-8859-1')
life = pd.read_csv('life.csv', usecols=['Country Code', 'Life expectancy at birth (years)'],
                   dtype={'Year': str}, encoding='ISO-8859-1')


df = pd.merge(world, life, how='inner', on=['Country Code'])
df.columns=df.columns.str.strip()
df = df.sort_values(["Country Code"], ascending = True)
data = df[list(world.columns)[3:]]

classlabel = df['Life expectancy at birth (years)']
x_train, x_test, y_train, y_test = train_test_split(data, classlabel, train_size=0.7, test_size=0.3, stratify = classlabel, random_state=4)


# In[155]:


import numpy as np
import math,random
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')



def VAT(R):
    """

    VAT algorithm adapted from matlab version:
    http://www.ece.mtu.edu/~thavens/code/VAT.m

    Args:
        R (n*n double): Dissimilarity data input
        R (n*D double): vector input (R is converted to sq. Euclidean distance)
    Returns:
        RV (n*n double): VAT-reordered dissimilarity data
        C (n int): Connection indexes of MST in [0,n)
        I (n int): Reordered indexes of R, the input data in [0,n)
    """
        
    R = np.array(R)
    N, M = R.shape
    if N != M:
        R = squareform(pdist(R))
        
    J = list(range(0, N))
    
    y = np.max(R, axis=0)
    i = np.argmax(R, axis=0)
    j = np.argmax(y)
    y = np.max(y)


    I = i[j]
    del J[I]

    y = np.min(R[I,J], axis=0)
    j = np.argmin(R[I,J], axis=0)
    
    I = [I, J[j]]
    J = [e for e in J if e != J[j]]
    
    C = [1,1]
    for r in range(2, N-1):   
        y = np.min(R[I,:][:,J], axis=0)
        i = np.argmin(R[I,:][:,J], axis=0)
        j = np.argmin(y)        
        y = np.min(y)      
        I.extend([J[j]])
        J = [e for e in J if e != J[j]]
        C.extend([i[j]])
    
    y = np.min(R[I,:][:,J], axis=0)
    i = np.argmin(R[I,:][:,J], axis=0)
    
    I.extend(J)
    C.extend(i)
    
    RI = list(range(N))
    for idx, val in enumerate(I):
        RI[val] = idx

    RV = R[I,:][:,I]
    
    return RV.tolist(), C, I


# In[156]:


print(type(x_train))
print(type(x_test))


# In[157]:


# preprocessing data
train_median = {col: x_train[col][x_train[col]!='..'].astype(float).median()
                  for col in list(x_train.columns)}

for col in train_median.keys():
    for i in range(df.shape[0]):
        if i in x_train.index and x_train.at[i, col] == '..':
            x_train.at[i, col] = train_median[col]
test_median = {col: x_test[col][x_test[col]!='..'].astype(float).median()
                  for col in list(x_test.columns)}
for col in test_median.keys():
    for i in range(df.shape[0]):
        if i in x_test.index and x_test.at[i, col] == '..':
            x_test.at[i, col] = test_median[col]


# ### Feature Engineering and Selection(6 marks)

# Interaction term pairs. Given a pair of features f1 and f2, create a new feature f12 = f1 x f2. All possible pairs can be considered.
# 

# 
# 
# 
# Clustering labels: apply k-means clustering to the data in world and then use the resulting cluster labels as the values for a new feature fclusterlabel. 
# You will need to decide how many clusters to use. 
# At test time, a label for a testing instance can be created by assigning it to its nearest cluster.

# ### interaction term pairs

# In[158]:


x_train_org = x_train.astype(float).copy()
x_test_org = x_test.astype(float).copy()

ncols = x_train.shape[1]
nrows = x_train.shape[0] 
for i in range(ncols-1):
    for j in range(i+1, ncols):
        lst = x_train.iloc[:, i].astype(float) * x_train.iloc[:, j].astype(float)
        x_train[x_train.columns[i]+ '+' + x_train.columns[j]] = lst
        
ncols = x_test.shape[1]
nrows = x_test.shape[0]

for i in range(ncols-1):
    for j in range(i+1, ncols):
        lst = x_test.iloc[:, i].astype(float) * x_test.iloc[:, j].astype(float)
        x_test[x_test.columns[i]+ '+' + x_test.columns[j]] = lst


# ### Clustering Labels

# In[159]:


import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns


# In[160]:


x_train2 = x_train.astype(float).copy()
RV, C, I = VAT(x_train2)
x=sns.heatmap(RV, cmap='viridis', xticklabels=False, yticklabels=False)
x.set(xlabel='Objects', ylabel='Objects')
plt.savefig('heatmap.png')


# In[161]:


# visualiiation determine number of clusters two small and one large


# In[162]:


n = 3
clusters = KMeans(n_clusters=n).fit(x_train.iloc[:, 3:])
x_train['clusterlabel'] = clusters.labels_
clusters = KMeans(n_clusters=n).fit(x_test.iloc[:, 3:])
x_test['clusterlabel'] = clusters.labels_


# In[163]:


from sklearn.feature_selection import SelectKBest, chi2

selector = SelectKBest(score_func=chi2,k=4)

#fit_transform returns the data after selecting the best features
new_data = selector.fit_transform(x_train, y_train)

#so you are trying to access get_support() on new data, which is not possible
mask = selector.get_support()
lst = list(x_train.columns[mask])
print(lst)


# In[164]:


x_train_ = x_train[[lst[0], lst[1], lst[2], lst[3]]]
x_test_ = x_test[[lst[0], lst[1], lst[2], lst[3]]]


scaler = preprocessing.StandardScaler().fit(x_train_)
x_train_ = scaler.transform(x_train_)
x_test_ = scaler.transform(x_test_)

knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train_, y_train)
y_pred = knn.predict(x_test_)
print('\nAccuracy of feature engineering: {0}'.format(np.around(accuracy_score(y_test, y_pred), decimals=3)))


# ###  Implement feature engineering and selection via PCA by taking the rst four principal
# components. You should use only these four features to perform 3-NN classication.

# In[165]:


from sklearn.decomposition import PCA


# In[166]:



scaler = preprocessing.StandardScaler().fit(x_train_org)
x_train_pca=scaler.transform(x_train_org)
x_test_pca=scaler.transform(x_test_org)

pca = PCA(n_components = 4)
x_train_pca = pca.fit_transform(x_train_pca)
x_test_pca = pca.transform(x_test_pca)

knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train_pca, y_train)
y_pred = knn.predict(x_test_pca)

print('\nAccuracy of PCA: {0}'.format(np.around(accuracy_score(y_test, y_pred), decimals=3)))


# ### Take first four features

# In[167]:


x_train_par = x_train_org.iloc[:, :4]
x_test_par = x_test_org.iloc[:, :4]
scaler = preprocessing.StandardScaler().fit(x_train_par)
x_train_par = scaler.transform(x_train_par)
x_test_par = scaler.transform(x_test_par)


# In[168]:





knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train_par, y_train)
y_pred = knn.predict(x_test_par)
print('Accuracy of first four features: {0}\n'.format(np.around(accuracy_score(y_test, y_pred), decimals=3)))


# In[ ]:





# In[ ]:




