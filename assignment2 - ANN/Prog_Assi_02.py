#%%
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('./Bank_Predictions.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values
#%%
print(repr(X))

#%%
# ------ Part-1: Data preprocessing ----------

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer 
#%%
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

#%%
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])


#onehotencoder = OneHotEncoder('auto' )
onehotencoder = OneHotEncoder()
ct = ColumnTransformer([("locaction", onehotencoder,[1])], remainder="passthrough") 
X = ct.fit_transform(X)

X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#%%
# ------- Part-2: Build the ANN --------

# import keras library and packages

import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
classifier = Sequential()
input_dim = len(X_train[0])

# You might use the following parameters: activation: 'relu' and/or 'sigmoid', optimization function is 'adam', loss function is 'binary_crossentropy', number of epochs is 100, samples per epoch are 10) 

# Adding the input layer and the first hidden layer

classifier.add( Dense(units=10, activation='relu', input_dim=len(X[0])) )

# Adding second hidden layer
classifier.add(Dense(units=6, activation='relu'))

# Adding output layer
classifier.add(Dense(units=1, activation='sigmoid'))

# Compiling the ANN
classifier.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

# Fitting the ANN to the training set
classifier.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0,
          validation_data=(X_test, y_test))

#%%
# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(y_pred)
y_pred = (y_pred > .5)
# Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# %%
print(cm)
# %%
