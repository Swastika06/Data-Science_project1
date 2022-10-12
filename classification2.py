#!/usr/bin/env python
# coding: utf-8

# In[1]:


# To work with dataframes
import pandas as pd 

# To perform numerical operations
import numpy as np

# To visualize data
import seaborn as sns

# To partition the data
from sklearn.model_selection import train_test_split

# Importing library for logistic regression
from sklearn.linear_model import LogisticRegression

# Importing performance metrics - accuracy score & confusion matrix
from sklearn.metrics import accuracy_score,confusion_matrix


# In[2]:


income=r"C:\Users\RPH\Downloads\income(1).csv"
data_income=pd.read_csv(income)
data = data_income.copy()

"""
#Exploratory data analysis:

#1.Getting to know the data
#2.Data preprocessing (Missing values)
#3.Cross tables and data visualization
"""
# =============================================================================
# Getting to know the data
# =============================================================================
#**** To check variables' data type
print(data.info())


# In[3]:


data.isnull()          
       
print('Data columns with null values:\n', data.isnull().sum())
#**** No missing values !

#**** Summary of numerical variables
summary_num = data.describe()
print(summary_num)            

#**** Summary of categorical variables
summary_cate = data.describe(include = "O")
print(summary_cate)

#**** Frequency of each categories
data['JobType'].value_counts()
data['occupation'].value_counts()

#**** Checking for unique classes
print(np.unique(data['JobType'])) 
print(np.unique(data['occupation']))


# In[4]:


summary_cate = data.describe(include = "O")
print(summary_cate)


# In[5]:


data = pd.read_csv(income,na_values=[" ?"]) 

# =============================================================================
# Data pre-processing
# =============================================================================
data.isnull().sum()

missing = data[data.isnull().any(axis=1)]
# axis=1 => to consider at least one column value is missing in a row


# In[6]:


print(missing)


# In[7]:


data2 = data.dropna(axis=0)
data3 = data2.copy()
data4 = data3.copy()
# Realtionship between independent variables
correlation = data2.corr()

# =============================================================================
# Cross tables & Data Visualization
# =============================================================================
# Extracting the column names
data2.columns   


# In[8]:


# =============================================================================
# Gender proportion table:
# =============================================================================
gender = pd.crosstab(index = data2["gender"], columns  = 'count', normalize = True)
print(gender)
# =============================================================================
#  Gender vs Salary Status:
# =============================================================================
gender_salstat = pd.crosstab(index = data2["gender"],columns = data2['SalStat'], margins = True, normalize =  'index') 
                 # Include row and column totals
print(gender_salstat)


# In[9]:


SalStat = sns.countplot(data2['SalStat'])

"""  75 % of people's salary status is <=50,000 
     & 25% of people's salary status is > 50,000
"""

##############  Histogram of Age  #############################
sns.distplot(data2['age'], bins=10, kde=False)


# In[10]:


sns.distplot(data2['age'], bins=10, kde=False)


# In[11]:


sns.boxplot('SalStat', 'age', data=data2)
data2.groupby('SalStat')['age'].median()


# In[12]:


JobType     = sns.countplot(y=data2['JobType'],hue = 'SalStat', data=data2)
job_salstat =pd.crosstab(index = data2["JobType"],columns = data2['SalStat'], margins = True, normalize =  'index')  
round(job_salstat*100,1)


# In[13]:


Education   = sns.countplot(y=data2['EdType'],hue = 'SalStat', data=data2)
EdType_salstat = pd.crosstab(index = data2["EdType"], columns = data2['SalStat'],margins = True,normalize ='index')  
round(EdType_salstat*100,1)


# In[14]:


Occupation  = sns.countplot(y=data2['occupation'],hue = 'SalStat', data=data2)
occ_salstat = pd.crosstab(index = data2["occupation"], columns =data2['SalStat'],margins = True,normalize = 'index')  
round(occ_salstat*100,1)


# In[15]:


sns.distplot(data2['capitalgain'], bins = 10, kde = False)

sns.distplot(data2['capitalloss'], bins = 10, kde = False)


# In[16]:


data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])


# In[17]:


new_data=pd.get_dummies(data2, drop_first=True)


# In[18]:


# Storing the column names 
columns_list=list(new_data.columns)
print(columns_list)


# In[19]:


# Separating the input names from data
features=list(set(columns_list)-set(['SalStat']))
print(features)


# In[20]:


# Storing the output values in y
y=new_data['SalStat'].values
print(y)


# In[21]:


# Storing the values from input features
x = new_data[features].values
print(x)


# In[22]:


# Splitting the data into train and test
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3, random_state=0)


# In[23]:


# Make an instance of the Model
logistic = LogisticRegression()


# In[24]:


# Fitting the values for x and y
logistic.fit(train_x,train_y)
logistic.coef_
logistic.intercept_


# In[25]:


# Prediction from test data
prediction = logistic.predict(test_x)
print(prediction)


# In[26]:


# Confusion matrix
confusion_matrix = confusion_matrix(test_y, prediction)
print(confusion_matrix)


# In[27]:


# Calculating the accuracy
accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)



# In[28]:


# Printing the misclassified values from prediction

print('Misclassified samples: %d' % (test_y != prediction).sum())


# In[29]:


# LOGISTIC REGRESSION - REMOVING INSIGNIFICANT VARIABLES
# =============================================================================

# Reindexing the salary status names to 0,1
data3['SalStat']=data3['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data3['SalStat'])


# In[30]:


cols = ['gender','nativecountry','race','JobType']
new_data = data3.drop(cols,axis = 1)

new_data=pd.get_dummies(new_data, drop_first=True)


# In[31]:


# Storing the column names 
columns_list2=list(new_data.columns)
print(columns_list2)


# In[32]:


# Separating the input names from data
features2=list(set(columns_list2)-set(['SalStat']))
print(features2)


# In[33]:


y2=new_data['SalStat'].values
print(y2)


# In[34]:


# Storing the values from input features
x2 = new_data[features2].values
print(x2)


# In[35]:


# Splitting the data into train and test
train_x2,test_x2,train_y2,test_y2 = train_test_split(x2,y2,test_size=0.3, random_state=0)


# In[36]:


# Make an instance of the Model
logistic2 = LogisticRegression()


# In[37]:


# Fitting the values for x and y
logistic2.fit(train_x2,train_y2)


# In[38]:


# Prediction from test data
prediction2 = logistic2.predict(test_x2)


# In[39]:


print('Misclassified samples: %d' % (test_y2 != prediction2).sum())


# In[40]:


from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


# In[41]:


# Storing the K nearest neighbors classifier
KNN_classifier = KNeighborsClassifier(n_neighbors = 5)  


# In[42]:


# Fitting the values for X and Y
KNN_classifier.fit(train_x, train_y) 


# In[43]:


# Predicting the test values with model
prediction = KNN_classifier.predict(test_x)


# In[44]:


confusion_matrix = confusion_matrix[test_y,prediction]
print(confusion_matrix)


# In[45]:


print('Misclassified samples: %d' % (test_y != prediction).sum())


# In[ ]:


print('Misclassified samples: %d' % (test_y != prediction).sum())
Misclassified_sample = []
# Calculating error for K values between 1 and 20
for i in range(1, 20):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x, train_y)
    pred_i = knn.predict(test_x)
    Misclassified_sample.append((test_y != pred_i).sum())

print(Misclassified_sample)


# In[ ]:


Misclassified_sample = []
# Calculating error for K values between 1 and 20
for i in range(1, 20):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x, train_y)
    pred_i = knn.predict(test_x)
    Misclassified_sample.append((test_y != pred_i).sum())

print(Misclassified_sample)


# In[ ]:


print(Misclassified_sample)


# In[ ]:




