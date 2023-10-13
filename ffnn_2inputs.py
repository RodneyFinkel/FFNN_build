from tensorflow import keras 
from keras.models import Sequential 
from keras import Input 
from keras.layers import Dense 

import pandas as pd
import numpy as np
import sklearn 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report 

import plotly
import plotly.express as px
import plotly.graph_objects as go


pd.options.display.max_columns=50
df = pd.read_csv('weatherAUS.csv', encoding='utf-8')

df = df.dropna(subset=['RainTomorrow'])

# Fill missing values for numeric columns only, using mean of column
numeric_columns = df.select_dtypes(include=['number']).columns 
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# create a flag for RainToday and RainTomorrow, note: RainTomorrowFlag will be our target variable
df['RainTodayFlag'] = df['RainToday'].apply(lambda x: 1 if x == 'Yes' else 0)
df['RainTomorrowFlag'] = df['RainTomorrow'].apply(lambda x: 1 if x == 'Yes' else 0)

# Select data for multiviariate modeling, two inputs
X=df[['WindGustSpeed', 'Humidity3pm']]
y=df['RainTomorrowFlag'].values

# Create training and testing samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 2-2-1 Neural Network
model2 = Sequential(name="Model-with-Two-Inputs") 
model2.add(Input(shape=(2,), name='Input-Layer')) 
model2.add(Dense(2, activation='softplus', name='Hidden-Layer')) 
model2.add(Dense(1, activation='sigmoid', name='Output-Layer')) 

# Compile the keras model
model2.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['Accuracy', 'Precision', 'Recall'],  
              loss_weights=None, 
              weighted_metrics=None, 
              run_eagerly=None, 
              steps_per_execution=None 
             )

# Fit keras model on the dataset
model2.fit(X_train, 
          y_train, 
          batch_size=10, 
          epochs=3, 
          verbose='auto', 
          callbacks=None, 
          validation_split=0.2, 
          shuffle=True, 
          class_weight={0 : 0.3, 1 : 0.7}, 
          sample_weight=None, 
          initial_epoch=0, 
          steps_per_epoch=None, 
          validation_steps=None, 
          validation_batch_size=None, 
          validation_freq=3, 
          max_queue_size=10, 
          workers=1, 
          use_multiprocessing=False, 
         )

# Predict class labels on training data
pred_labels_tr = (model2.predict(X_train) > 0.5).astype(int)
# Predict class labels on a test data
pred_labels_te = (model2.predict(X_test) > 0.5).astype(int)

# Model Performance Summary
print("")
print('-------------------- Model Summary --------------------')
model2.summary() # print model summary
print("")
print('-------------------- Weights and Biases --------------------')
for layer in model2.layers:
    print("Layer: ", layer.name) # print layer name
    print("  --Kernels (Weights): ", layer.get_weights()[0]) # kernels (weights)
    print("  --Biases: ", layer.get_weights()[1]) # biases
    
print("")
print('---------- Evaluation on Training Data ----------')
print(classification_report(y_train, pred_labels_tr))
print("")

print('---------- Evaluation on Test Data ----------')
print(classification_report(y_test, pred_labels_te))
print("")


def Plot_3D(X, X_test, y_test, clf, x1, x2, mesh_size, margin):
    
    # clf refers to the name of the model constructed in the neural network. In this case model2       
    # Specify a size of the mesh to be used and assign to local variables
    mesh_size=mesh_size # granularity/step size of the mesh grid
    margin=margin

    # Create a mesh grid on which we will run our model
    x_min, x_max = X.iloc[:, 0].min() - margin, X.iloc[:, 0].max() + margin
    y_min, y_max = X.iloc[:, 1].min() - margin, X.iloc[:, 1].max() + margin
    xrange = np.arange(x_min, x_max, mesh_size)
    yrange = np.arange(y_min, y_max, mesh_size)
    # np.meshgrid(x, y, z) possible to provide three or more input arrays, 
    # which will create the values for a higher-dimensional grid space.
    xx, yy = np.meshgrid(xrange, yrange, sparse=False) # meshgrid generates coordinate matrices by taking 2 1-D arrays and returns 2 2-D arrays with all pairs of x,y in the input arrays
            
    # Calculate Neural Network predictions on the grid
    # np.c_() is np.concatenate(axis)
    Z = model2.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Create a 3D scatter plot using the test data for true z values for comparison with the surface plot
    # of predicted values (not working correctly, possible that test size split is too small)
    fig = px.scatter_3d(x=X_test[x1], y=X_test[x2], z=y_test,
                     opacity=0.8, color_discrete_sequence=['black'], height=900, width=1000)
    
    # Create a 3D scatter plot using the X data from meshgrid for predicted z values
    # fig = px.scatter_3d(x=xx.ravel(), y=yy.ravel(), z=Z.ravel(),
    #                  opacity=0.8, color_discrete_sequence=['black'], height=900, width=1000)

    # Set figure title and colors
    fig.update_layout(#title_text="Scatter 3D Plot with FF Neural Network Prediction Surface",
                      paper_bgcolor = 'white',
                      scene_camera=dict(up=dict(x=0, y=0, z=1), 
                                        center=dict(x=0, y=0, z=-0.1),
                                        eye=dict(x=0.75, y=-1.75, z=1)),
                                        margin=dict(l=0, r=0, b=0, t=0),
                      scene = dict(xaxis=dict(title=x1,
                                              backgroundcolor='white',
                                              color='black',
                                              gridcolor='#f0f0f0'),
                                   yaxis=dict(title=x2,
                                              backgroundcolor='white',
                                              color='black',
                                              gridcolor='#f0f0f0'
                                              ),
                                   zaxis=dict(title='Probability of Rain Tomorrow',
                                              backgroundcolor='lightgrey',
                                              color='black', 
                                              gridcolor='#f0f0f0', 
                                              )))
    
    # Update marker size
    fig.update_traces(marker=dict(size=1))

    # Add prediction plane/surface plot to fig (add_traces ie traces equivalent to plot)
    fig.add_traces(go.Surface(x=xrange, y=yrange, z=Z, name='FF NN Prediction Plane',
                              colorscale='Bluered',
                              reversescale=True,
                              showscale=False, 
                              contours = {"z": {"show": True, "start": 0.5, "end": 0.9, "size": 0.5}}))
    fig.show()
    return fig


# Call the above function
fig = Plot_3D(X, X_test, y_test, model2, x1='WindGustSpeed', x2='Humidity3pm', mesh_size=1, margin=0)



