from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Define input layers for each input
input1 = Input(shape=(input1_shape,), name='input1')  # Input 1
input2 = Input(shape=(input2_shape,), name='input2')  # Input 2

# Concatenate the inputs
concatenated_inputs = concatenate([input1, input2], name='concatenated_inputs')

# Hidden layer with 3 nodes and ReLU activation
hidden_layer = Dense(3, activation='relu', name='hidden_layer')(concatenated_inputs)

# Output layer with 2 nodes and softmax activation (assuming a classification task)
output_layer = Dense(2, activation='softmax', name='output_layer')(hidden_layer)

# Define the model
model = Model(inputs=[input1, input2], outputs=output_layer)

# Compile the model with appropriate loss and optimizer
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()
