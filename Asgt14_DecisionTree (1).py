#!/usr/bin/env python
# coding: utf-8

# In[16]:


#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
#for encoding
from sklearn.preprocessing import LabelEncoder
#for train test splitting
from sklearn.model_selection import train_test_split
#for decision tree object
from sklearn.tree import DecisionTreeClassifier
#for checking testing results
from sklearn.metrics import classification_report, confusion_matrix
#for visualizing tree 
from sklearn.tree import plot_tree


# ### Data

# In[2]:


df = pd.read_csv('C:/Users/17pol/Downloads/Company_Data.csv')
df.head()


# ### EDA

# In[3]:


df.sample(10)


# In[4]:


df.info()


# In[5]:


df.isna().sum()


# In[6]:


df.shape


# ### Pairplot

# In[7]:


sns.pairplot(data=df, hue = 'ShelveLoc')


# #### get dummies

# In[8]:


#Creating dummy vairables dropping first dummy variable
df=pd.get_dummies(df,columns=['Urban','US'], drop_first=True)
print(df.head())


# In[9]:


df.info()


# In[10]:


from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

df['ShelveLoc']=df['ShelveLoc'].map({'Good':1,'Medium':2,'Bad':3})
print(df.head())


# In[11]:


x=df.iloc[:,0:6]
y=df['ShelveLoc']


# In[12]:


print(df['ShelveLoc'].unique())

print(df.ShelveLoc.value_counts())

colnames = list(df.columns)
colnames


# In[13]:


# Splitting data into training and testing data set
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=40)


# ### Building Decision Tree Classifier using Entropy Criteria

# In[17]:


model = DecisionTreeClassifier(criterion = 'entropy',max_depth=3)
model.fit(x_train,y_train)


# In[21]:


from sklearn import tree

#PLot the decision tree
plt.figure(figsize=(15,6))
tree.plot_tree(model);


# In[22]:


fn=['Sales','CompPrice','Income','Advertising','Population','Price']
cn=['1', '2', '3']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(model,
               feature_names = fn, 
               class_names=cn,
               filled = True);


# In[23]:


#Predicting on test data
preds = model.predict(x_test) # predicting on test data set 
pd.Series(preds).value_counts() # getting the count of each category 


# In[24]:


preds


# In[25]:


pd.crosstab(y_test,preds) # getting the 2 way table to understand the correct and wrong predictions


# In[26]:


# Accuracy 
np.mean(preds==y_test)


# ### Building Decision Tree Classifier (CART) using Gini Criteria

# In[27]:


from sklearn.tree import DecisionTreeClassifier
model_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)
model_gini.fit(x_train, y_train)

#Prediction and computing the accuracy
pred=model.predict(x_test)
np.mean(preds==y_test)


# ### Decision Tree Regression Example

# In[28]:


# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor


# In[29]:


array = df.values
X = array[:,0:3]
y = array[:,3]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

#Find the accuracy
model.score(X_test,y_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




