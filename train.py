import os
import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#Load data
#iris = load_iris()
#X, y = iris.data, iris.target
file_path = os.path.join('data', 'Iris.csv')
print(file_path)
df = pd.read_csv(file_path)
df = df.drop(columns=['Id'])
print(df.head())
X = df.drop('Species', axis=1)
y = df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#Train model
rf = RandomForestClassifier()    
rf.fit(X, y)  
print(rf.score(X, y))
#save model
joblib.dump(rf, 'models/rf_iris.pkl')
print('Model saved successfully')