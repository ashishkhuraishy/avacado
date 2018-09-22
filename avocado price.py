#!/usr/bin/env python
# coding: utf-8

# In[53]:


import pandas as pd
from time import time


# In[54]:


path = 'C:/Users/Ashish Khuraishy/Desktop/New folder/Avacado/avocado.csv'
avocado = pd.read_csv(path)
avocado.describe()


# In[55]:


avocado.columns


# In[56]:


avocado.head()


# In[57]:


y = avocado.AveragePrice
features = ['Total Volume', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags', '4046', '4225', '4770']

x = avocado[features]


# In[58]:


from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(x, y, random_state = 0)


# In[59]:


from sklearn.tree import DecisionTreeRegressor

clf = DecisionTreeRegressor()
t1 = time()

clf.fit(train_x, train_y)
tr_time = round(time()-t1, 3)
print("Training Time : %ss" % tr_time)


# In[60]:


t2 = time()
pred = clf.predict(test_x)
pr_time = round(time()-t2, 2)
print("Predicting time : %ss" % pr_time)


# In[61]:


from sklearn.metrics import mean_absolute_error

men = mean_absolute_error(pred, test_y)
print(men)


# In[62]:


from sklearn.metrics import accuracy_score

score = clf.score(test_x, test_y)
print("Accuracy : %s" % score)

