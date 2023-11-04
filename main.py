import numpy as np
import tensorflow as tf

# training data
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

# training variables
learning_rate = 0.05
epochs = 10000

# define network model
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(units=3, activation=tf.keras.activations.tanh),
    tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid)
])

# setup and compile the network
model.compile(optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.metrics.mean_squared_error, tf.keras.metrics.binary_accuracy])
model.summary()

# train the model
history = model.fit(x_train, y_train, batch_size=1, epochs=epochs)

predictions = model.predict_on_batch(x_train)
print(predictions)
