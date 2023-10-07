from tensorflow import keras # for buidling neural networks
from keras.models import Sequential # for creating a linear stack of layers for the neural network
from keras import Input # for instantiating a keras tensor
from keras.layers import Dense # for creating regular densely-connected neural network

import pandas as pd
import numpy as np
import sklearn # for model evaluation
from sklearn.model_selection import train_test_split # for splitting data into training and test samples
from sklearn.metrics import classification_report # for model evaluation metrics

import plotly
import plotly.express as px
import plotly.graph_objects as go


pd.options.display.max_columns=50
df = pd.read_csv('weatherAUS.csv', encoding='utf-8')

# drop records where target RainTomorrow=NaN
df = df.dropna(subset=['RainTomorrow'])

# Fill missing values for numeric columns
numeric_columns = df.select_dtypes(include=['number']).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
# create a flag for RainToday and RainTomorrow, note: RainTomorrowFlag will be our target variable
df['RainTodayFlag'] = df['RainToday'].apply(lambda x: 1 if x == 'Yes' else 0)
df['RainTomorrowFlag'] = df['RainTomorrow'].apply(lambda x: 1 if x == 'Yes' else 0)

# using one input, [humidity3pm]

X = df[['Humidity3pm']] # double brackets are for obtaining a new dataframe and not just a series from the column Humidity3pm. To give the column more structure we want a dataframe
y = df['RainTomorrowFlag'].values  # .values converts the series to a numpy array

# Create training and testing samples. Randomly shuffle the data and divide it into training and testing sets based on test_size (20% for testing) Seed for the random number generator is 0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Specify the structure of the Neural Network (1-2-1 Structure)
model = Sequential(name='Model-with-One-Input')
model.add(Input(shape=(1,), name='Input-Layer'))
model.add(Dense(2, activation='softplus', name="Hidden-Layer")) # softplus(x) = log(exp(x) + 1) using the softplus function as a model curve to pass biased and weighted inputs to construct a model curve that best fits the labelled/target data
model.add(Dense(1, activation='sigmoid', name='Output-Layer')) # sigmoid(x) = 1 / (1 + exp(-x)) (sigmoid function is for binary categorization)

# Compile keras model
model.compile(optimizer='adam', # default=RMSProp, algorithm to be used in backpropogation
              loss='binary_crossentropy', # loss function to be optimized. A string(name of loss function) or a tf.keras.losses.Loss instance
              metrics=['Accuracy', 'Precision', 'Recall'], # List of metrics to be evaluated by the model during training and testing. Each of these can be a string(name of a built in function)
              loss_weights=None, # default=None, Optional list or dictionary specifying scalar coefficients(Python floats) to weigh the loss contributions of different model outputs
              run_eagerly=None, # List of metrics to be evaluated and weighted by sample_weight or class_weight during training and testing
              steps_per_execution =None # Defaults to 1. The number of batches to run during each tf.function call. Running multiple batches inside a single tf.function call can greatly improve performance on TPU's or small models with a large Python overhead             
              )

# fit keras model on the dataset
model.fit(X_train, # input data
          y_train,  # target data
          batch_size=10, 
          
          
          )






