import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

datasetX = np.zeros(1000)
datasetY = np.zeros(1000)

#generate synthetic training data. See full text for explanation
for i in range(1000):
	datasetY[i] = np.exp(i/500.0) / 10.0

for j in range(1000):
	t = 1.0/1000.0
	datasetX[j] = t*j

slicesX = np.zeros((760,240))
slicesY = np.zeros(760)
for i in range(760):
	working = datasetX[i:i+240]
	slicesX[i,:] = np.copy(working)
	print(slicesX[i,:10])
	slicesY[i] = datasetY[i+240]

#Define the model using Keras
model = keras.Sequential([
	layers.Dense(units=32, input_shape=[240,], activation='sigmoid'), 
	layers.Dense(units=32, activation='sigmoid'), 
	layers.Dense(units=32, activation='sigmoid'), 
	layers.Dense(units=32, activation='sigmoid'), 
	layers.Dense(units=32, activation='sigmoid'), 
	layers.Dense(units=1, activation='linear'),
	])
model.summary()
model.compile(optimizer='adam',loss='mean_squared_error')

#train the network and save the trained version
history = model.fit(slicesX, slicesY, epochs=200, batch_size=1, verbose=1)
model.save("appleTestTrained.h5")

#check the predictions
plt.plot(model.predict(datasetX))
plt.show()
plt.plot(datasetY)
plt.show()




