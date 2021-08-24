#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


customers = pd.read_csv("Ecommerce Customers")


# **Check the head of customers.**

# In[3]:


customers.head()


# In[278]:


customers.describe()


# In[279]:


customers.info()


# ## Training and Testing Data
# 
# let's go ahead and split the data into training and testing sets.
#  Set a variable X equal to the numerical features of the customers dataset and a variable y equal to the "Yearly Amount Spent" column(label). 

# In[4]:


y = customers['Yearly Amount Spent']


# In[5]:


X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]


# Import and Use model_selection.train_test_split from sklearn to split the data into training and testing sets.
# we are going to Set test_size = 0.3 (30%) and random_state=101 (optional) but if you are going to code along its best to set it at the same number so our ouputs can be same.
# 
# TRAINING SETS
# 
# X_train = [features] 70% of datasets,
# 
# y_train = [features] 70% of datasets,
# 
# TEST SETS
# 
# X_test =  [label] 30% of datasets,
# 
# y_test =  [label] 30% of datasets
# 

# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# ## Time to Train the Model
# 
# we are going to Import LinearRegression from sklearn.linear_model 

# In[8]:


from sklearn.linear_model import LinearRegression


# **we are creating an instance of a LinearRegression() model named lm.**

# In[9]:


lm = LinearRegression()


#  Train or fit lm on the training data splits.

# In[10]:


lm.fit(X_train,y_train)


# lets go ahead and print out the coefficients of the model

# In[11]:


# The coefficients
print('The Coefficients:', lm.coef_)


# ## Predicting Test Data
# Now that we have fit our model, we are going to evaluate its predictions using our test data split.
# 
# We are going to Use lm.predict() method to predict off the X_test split set of the data.

# In[12]:


predictions = lm.predict( X_test)


#  We would Create a scatterplot of the real test values versus the predicted values.

# In[14]:


sns.set_style('whitegrid')
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# ## Evaluating the Model
# 
# 
# Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error.

# In[15]:


# calculate these metrics by hand!
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# ## Residuals
# 
# Exploring the residuals after getting a good fit 
# 
# **Plot a histogram of the residuals and make sure it looks normally distributed.**
# 
# We can use either seaborn distplot, or matplotlib plt.hist().

# In[317]:


sns.distplot((y_test-predictions),bins=50);


# ## Conclusion
# we are going to answer the question to our problem statement here.
# 
# The company is trying to decide whether to focus their efforts on their mobile app experience or their website.
# 
# creating a dataframe of the coefficients

# In[16]:


coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients


# In[ ]:




