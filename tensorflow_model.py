import tensorflow as tf

# Define the model architecture
model_tf = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(2,)), # Hidden layer
    tf.keras.layers.Dense(1) ])

model_tf.compile(optimizer='adam', loss='mean_squared_error')

print("TensorFlow model summary:")
model_tf.summary()