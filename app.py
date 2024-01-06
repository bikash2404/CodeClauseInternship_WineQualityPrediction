
# IMPORTING NECESSARY LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st

# READING DATA
wine = pd.read_csv('WineQT.csv')

#Create the predictor'X' and target 'Y' variables 
X= wine.drop('quality',axis=1)
Y= wine['quality'].apply(lambda yval:1 if yval>=7 else 0)

#Split the data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

#using random forest
model= RandomForestClassifier()
model.fit(X_train,Y_train)

#accuracy
X_test_prediction = model.predict(X_test)
print(accuracy_score(X_test_prediction, Y_test))

#web app
st.title("Wine Quality Prediction Model")
input_text = st.text_input('Enter the values of the following parameters in the same order as given below separated by commas: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol :')
input_text_list = input_text.split(',')
features = np.asarray(input_text_list)
prediction = model.predict(features.reshape(1,-1))
if prediction[0] == 1:
  st.write("The wine is of good quality")
else:
    st.write("The wine is of bad quality")