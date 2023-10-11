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

# Fill missing values for numeric columns only, using mean of column
numeric_columns = df.select_dtypes(include=['number']).columns #.columns accesses the columns names attribute of the resulting dataframe
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# create a flag for RainToday and RainTomorrow, note: RainTomorrowFlag will be our target variable
df['RainTodayFlag'] = df['RainToday'].apply(lambda x: 1 if x == 'Yes' else 0)
df['RainTomorrowFlag'] = df['RainTomorrow'].apply(lambda x: 1 if x == 'Yes' else 0)



# using one input, [humidity3pm], for the FFNN

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
model.compile(optimizer='adam', # default=RMSProp, algorithm to be used in backpropogation. Adam is based on stochastic gradient descent (Adaptive movement estimation algorithm)
              loss='binary_crossentropy', # loss function to be optimized. A string(name of loss function) or a tf.keras.losses.Loss instance
              metrics=['Accuracy', 'Precision', 'Recall'], # List of metrics to be evaluated by the model during training and testing. Each of these can be a string(name of a built in function)
              loss_weights=None, # default=None, Optional list or dictionary specifying scalar coefficients(Python floats) to weigh the loss contributions of different model outputs
              run_eagerly=None, # List of metrics to be evaluated and weighted by sample_weight or class_weight during training and testing
              steps_per_execution =None # Defaults to 1. The number of batches to run during each tf.function call. Running multiple batches inside a single tf.function call can greatly improve performance on TPU's or small models with a large Python overhead             
              )

# fit keras model on the dataset
model.fit(X_train, # input data
          y_train,  # target data
          batch_size=10, # Number of samples per gradient update. If unspecified, batch_size defaults to 32
          epochs= 3, # default=1, an epoch is an iteration over the entire X and y data provided
          verbose='auto', # default='auto' auto defaults to one(progress bar)
          callbacks= None, # default none. see tf.keras.callbacks
          validation_split=0.2, # fraction of the training data to be used as validation data. The model sets apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this
          shuffle=True, # default=True Boolean, whether to shuffle the training data before each epoch
          class_weight={0 : 0.3, 1 : 0.7}, # default=None Optional dictionary mapping class indices to a weight. Used for weighting the loss function(during training only). This can be useful to tell the model to pay more attention to samples from an under-represented class
          sample_weight=None, # default=None Optional numpy array of weight for the training samples, used for weighting the loss function(during training only)
          initial_epoch=0, # Integer, default=0, Epoch at which to start training (useful for resuming a previous training run).
          steps_per_epoch=None, # Integer or None, default=None, Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. When training with input tensors such as TensorFlow data tensors, the default None is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be determined. 
          validation_steps=None, # Only relevant if validation_data is provided and is a tf.data dataset. Total number of steps (batches of samples) to draw before stopping when performing validation at the end of every epoch.
          validation_batch_size=None, # Integer or None, default=None, Number of samples per validation batch. If unspecified, will default to batch_size.
          validation_freq=3, # default=1, Only relevant if validation data is provided. If an integer, specifies how many training epochs to run before a new validation run is performed, e.g. validation_freq=2 runs validation every 2 epochs.
          max_queue_size=10, # default=10, Used for generator or keras.utils.Sequence input only. Maximum size for the generator queue. If unspecified, max_queue_size will default to 10.
          workers=1, # default=1, Used for generator or keras.utils.Sequence input only. Maximum number of processes to spin up when using process-based threading. If unspecified, workers will default to 1.
          use_multiprocessing=False, # default=False, Used for generator or keras.utils.Sequence input only. If True, use process-based threading. If unspecified, use_multiprocessing will default to False. 
         )


# Use model to make predictions
# Predict class labels on training data
pred_labels_tr = (model.predict(X_train) > 0.5).astype(int)
# Predict class labels on a test data
pred_labels_te = (model.predict(X_test) > 0.5).astype(int)


# Model Performance Summary
print("")
print('-------------------- Model Summary --------------------')
model.summary() # print model summary
print("")
print('-------------------- Weights and Biases --------------------')
for layer in model.layers:
    print("Layer: ", layer.name) # print layer name
    print("  --Kernels (Weights): ", layer.get_weights()[0]) # weights
    print("  --Biases: ", layer.get_weights()[1]) # biases
    
print("")
print('---------- Evaluation on Training Data ----------')
print(classification_report(y_train, pred_labels_tr))
print("")



# Create 100 evenly spaced points from smallest X to largest X
X_range = np.linspace(X.min(), X.max(), 100)
# Predict probabilities for rain tomorrow
y_predicted = model.predict(X_range.reshape(-1, 1))

# Create a scatter plot
fig = px.scatter(x=X_range.ravel(), y=y_predicted.ravel(), 
                 opacity=0.8, color_discrete_sequence=['black'],
                 labels=dict(x="Value of Humidity3pm", y="Predicted Probability of Rain Tomorrow",))

# Change chart background color
fig.update_layout(dict(plot_bgcolor = 'white'))

# Update axes lines
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                 zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                 showline=True, linewidth=1, linecolor='black')

fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                 zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                 showline=True, linewidth=1, linecolor='black')

# Set figure title
fig.update_layout(title=dict(text="Feed Forward Neural Network (1 Input) Model Results", 
                             font=dict(color='black')))
# Update marker size
fig.update_traces(marker=dict(size=7))

fig.show()


