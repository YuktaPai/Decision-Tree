#!/usr/bin/env python
# coding: utf-8

# In[4]:


#importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets  
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing


# ### Data

# In[5]:


df = pd.read_csv("C:/Users/17pol/Downloads/Fraud_check.csv")
df.head()


# ### EDA & Data Preprocessing

# In[6]:


df.sample(10)


# In[7]:


df.shape


# In[8]:


df.isna().sum()


# In[9]:


df.info()


# In[10]:


#Creating dummy vairables for ['Undergrad','Marital.Status','Urban'] dropping first dummy variable
df=pd.get_dummies(df,columns=['Undergrad','Marital.Status','Urban'], drop_first=True)


# In[11]:


#Creating new cols TaxInc and dividing 'Taxable.Income' cols on the basis of [10002,30000,99620] for Risky and Good
df["TaxInc"] = pd.cut(df["Taxable.Income"], bins = [10002,30000,99620], labels = ["Risky", "Good"])
print(df)


# In[12]:


## Lets assume: taxable_income <= 30000 as “Risky=0” and others are “Good=1”
#After creation of new col. TaxInc also made its dummies var concating right side of df


# In[13]:


df = pd.get_dummies(df,columns = ["TaxInc"],drop_first=True)
#Viewing buttom 10 observations
df.tail(10)


# In[14]:


# let's plot pair plot to visualise the attributes all at once
import seaborn as sns
sns.pairplot(data=df, hue = 'TaxInc_Good')


# In[15]:


# Normalization function 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[16]:


# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(df.iloc[:,1:])
df_norm.tail(10)


# In[17]:


# Declaring features & target
X = df_norm.drop(['TaxInc_Good'], axis=1)
y = df_norm['TaxInc_Good']


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


# Splitting data into train & test
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=0)


# In[20]:


##Droping the Taxable income variable
df.drop(["Taxable.Income"],axis=1,inplace=True)


# In[21]:


df.rename(columns={"Undergrad":"undergrad","Marital.Status":"marital","City.Population":"population","Work.Experience":"experience","Urban":"urban"},inplace=True)
## As we are getting error as "ValueError: could not convert string to float: 'YES'".
## Model.fit doesnt not consider String. So, we encode


# In[22]:


from sklearn import preprocessing
le=preprocessing.LabelEncoder()
for column_name in df.columns:
    if df[column_name].dtype == object:
        df[column_name] = le.fit_transform(df[column_name])
    else:
        pass


# In[23]:


##Splitting the data into featuers and labels
features = df.iloc[:,0:5]
labels = df.iloc[:,5]


# In[24]:


## Collecting the column names
colnames = list(df.columns)
predictors = colnames[0:5]
target = colnames[5]
##Splitting the data into train and test


# In[25]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size = 0.2,stratify = labels)


# In[26]:


##Model building
from sklearn.ensemble import RandomForestClassifier as RF
model = RF(n_jobs = 3,n_estimators = 15, oob_score = True, criterion = "entropy")
model.fit(x_train,y_train)


# In[27]:


model.estimators_
model.classes_
model.n_features_
model.n_classes_


# In[28]:


model.n_outputs_


# In[29]:


model.oob_score_


# In[30]:


##Predictions on train data
prediction = model.predict(x_train)


# In[31]:


##Accuracy
# For accuracy 
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_train,prediction)


# In[32]:


np.mean(prediction == y_train)


# In[34]:


##Confusion matrix
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_train,prediction)
confusion


# In[ ]:





# In[35]:


##Prediction on test data
pred_test = model.predict(x_test)


# In[36]:


##Accuracy
acc_test =accuracy_score(y_test,pred_test)
##78.333%


# In[37]:


pip install pydotplus


# In[38]:


## In random forest we can plot a Decision tree present in Random forest
from sklearn.tree import export_graphviz
import pydotplus
from six import StringIO


# In[39]:


tree = model.estimators_[5]
dot_data = StringIO()
export_graphviz(tree,out_file = dot_data, filled = True,rounded = True, feature_names = predictors ,class_names = target,impurity =False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())


# ###  Building Decision Tree Classifier using Entropy Criteria

# In[40]:


model = DecisionTreeClassifier(criterion = 'entropy',max_depth=3)
model.fit(x_train,y_train)
DecisionTreeClassifier(criterion='entropy', max_depth=3)
from sklearn import tree
#PLot the decision tree
plt.figure(figsize=(15,8))
tree.plot_tree(model);


# In[41]:


colnames = list(df.columns)
colnames


# In[42]:


fn=['population','experience','Undergrad_YES','Marital.Status_Married','Marital.Status_Single','Urban_YES']
cn=['1', '0']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (8,8), dpi=300)
tree.plot_tree(model,
               feature_names = fn, 
               class_names=cn,
               filled = True);


# In[43]:


#Predicting on test data
preds = model.predict(x_test) # predicting on test data set 
pd.Series(preds).value_counts() # getting the count of each category 


# In[44]:


preds


# In[45]:


pd.crosstab(y_test,preds) # getting the 2 way table to understand the correct and wrong predictions


# In[46]:


# Accuracy 
np.mean(preds==y_test)


# In[ ]:





# ### Building Decision Tree Classifier (CART) using Gini Criteria

# In[47]:


from sklearn.tree import DecisionTreeClassifier
model_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)
model_gini.fit(x_train, y_train)


# In[48]:


#Prediction and computing the accuracy
pred=model.predict(x_test)
np.mean(preds==y_test)


# ### Decision Tree Regression Example

# In[49]:


# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
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




