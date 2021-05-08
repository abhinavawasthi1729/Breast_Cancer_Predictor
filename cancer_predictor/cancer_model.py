#import libraries 
import numpy as np 
import pandas as pd 

df = pd.read_csv('data.csv') 

df = df.dropna(axis=1)

df=df[['id','diagnosis','radius_mean','texture_mean','smoothness_mean','compactness_mean','concavity_mean','symmetry_mean','fractal_dimension_mean']]

from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
df.iloc[:,1]= labelencoder_Y.fit_transform(df.iloc[:,1].values)

#splitting the data set
X = df.iloc[:, 2:9].values 
Y = df.iloc[:, 1].values 

#Split the data again, but this time into 75% training and 25% testing data sets.
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

#Feature Scaling(Scale the data to bring all features to the same level of magnitude)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Create a function to hold many different models
def models(X_train,Y_train):
     #Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm
  from sklearn.ensemble import RandomForestClassifier
  forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
  forest.fit(X_train, Y_train)
  
  #print model accuracy on the training data.
  
  print('Random Forest Classifier Training Accuracy:', forest.score(X_train, Y_train))
  
  return forest

#Create the model that contains all of the models
model = models(X_train,Y_train)

import pickle

pickle.dump(model,open('model1.pkl', 'wb'))

