#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


df=pd.read_csv("Credit Card Default II (balance).csv")


# In[5]:


#see if there are rows with negative values for 'age', assumption is that these rows are typos

df[df.age<0]


# In[6]:


#return positive values for values that are negative for age

df[df.age<0]=df[df.age<0].abs()


# In[7]:


#drop NA values 

df=df.dropna()


# In[9]:


from sklearn import preprocessing


# In[10]:


import numpy as np


# In[11]:


scaler = preprocessing.MinMaxScaler()
names = df.columns
df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(df, columns=names)
scaled_df.head()


# In[12]:


df = pd.DataFrame(df,columns = ['income','age','loan','default'])


# In[13]:


print(df)
print(type(df))


# In[14]:


#convert default column back to integer, dummy variables
df['default'] = df['default'].astype('int')


# In[16]:


#convert default column to categorical value
df['default'] = df['default'].astype('category')


# In[17]:


df.dtypes


# In[19]:


X = df.iloc[:, 0:3]
Y= df.loc[:,["default"]]


# In[20]:


#train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3)
print(X_train, X_test, Y_train, Y_test )


# In[21]:


#logistic regression model
from sklearn import linear_model
model = linear_model.LogisticRegression()

#train model
model.fit(X_train, Y_train)

#predict on trainset
pred = model.predict(X_train)


# In[22]:


#confusion matrix for trainset
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_train, pred)
print(cm)
accuracy = (cm[0,0]+cm[1,1])/sum(sum(cm))
print(accuracy)

#predict on testset
pred = model.predict(X_test)

#confusion matrix for testset
cm = confusion_matrix(Y_test, pred)
print(cm)
accuracy = (cm[0,0]+cm[1,1])/sum(sum(cm))
print(accuracy)



# In[23]:


#Decision tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, Y_train)
pred = model.predict(X_train)



# In[24]:


#confusion matrix for trainset

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_train, pred)
print(cm)

accuracy = (cm[0,0] + cm[1,1])/sum(sum(cm))
print(accuracy)


# In[25]:


#confusion matrix for testset

pred = model.predict(X_test)
cm = confusion_matrix(Y_test, pred)
print(cm)
accuracy = (cm[0,0] + cm[1,1])/sum(sum(cm))
print(accuracy)


# In[26]:


#Random forest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(max_depth=3)
model.fit(X_train, Y_train)
pred = model.predict(X_train)



# In[27]:


#confusion matrix for trainset

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_train, pred)
print(cm)
accuracy = (cm[0,0] + cm[1,1])/sum(sum(cm))
print(accuracy)


# In[28]:


#confusion matrix for testset

pred = model.predict(X_test)
cm = confusion_matrix(Y_test, pred)
print(cm)
accuracy = (cm[0,0] + cm[1,1])/sum(sum(cm))
print(accuracy)


# In[29]:


#XGBoost

from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(max_depth=3)
model.fit(X_train, Y_train)
pred = model.predict(X_train)


# In[30]:


#confusion matrix for trainset

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_train, pred)
print(cm)
accuracy = (cm[0,0] + cm[1,1])/sum(sum(cm))
print(accuracy)



# In[31]:


#confusion matrix for testset

pred = model.predict(X_test)
cm = confusion_matrix(Y_test, pred)
print(cm)
accuracy = (cm[0,0] + cm[1,1])/sum(sum(cm))
print(accuracy)



# In[32]:


#keras sequential

from keras.models import Sequential
from keras.layers import Dense, Dropout

model=Sequential()


# In[33]:


#network design

model.add(Dense(5, input_dim=3, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(4, activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(1, activation='sigmoid'))

model.summary()


# In[34]:


#compile

model.compile(loss = 'binary_crossentropy', optimizer = 'Adamax', metrics = ['accuracy'])


# In[35]:


#fit

model.fit(X_train, Y_train, batch_size = 10, epochs=10)


# In[36]:


#evaluate

model.evaluate(X_train, Y_train)
model.evaluate(X_test, Y_test)


# In[37]:


#prediction

import numpy as np
from sklearn.metrics import confusion_matrix

pred=model.predict(X_train)
pred=np.where(pred>0.5,1,0)
cm=confusion_matrix(Y_train, pred)
print(cm)
accuracy=(cm[0,0]+cm[1,1])/sum(sum(cm))
print(accuracy)

pred=model.predict(X_test)
pred=np.where(pred>0.5,1,0)
cm=confusion_matrix(Y_test, pred)
print(cm)
accuracy=(cm[0,0]+cm[1,1])/sum(sum(cm))
print(accuracy)


# In[38]:


model.save("Default")


# In[37]:





# In[38]:





# In[39]:





# In[ ]:




