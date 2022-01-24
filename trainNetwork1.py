import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

datasetX = np.zeros((14,200))
datasetY = np.zeros(200)

#create training dataset (see full text for explanation)
#build a synthetic set of sensor curves
for i in range(200):
	datasetY[i] = np.exp(i/100.0) / 10.0

for i in range(14):
	for j in range(200):
		t = (-1.0+(i/7.0)) / 200.0
		datasetX[i, j] = t*j

#reorganise the data into windowed time slices as per the research paper
slicesX = np.zeros((176,336))
slicesY = np.zeros(176)
for i in range(176):
	working = np.zeros(336)
	for j in range(14):
		working[(j*24):(j+1)*24] = datasetX[j, i:i+24]
	slicesX[i,:] = np.copy(working[:])
	print(slicesX[i,:10])
	slicesY[i] = datasetY[i+24]

#Define the neural network using Keras
model = keras.Sequential([
	layers.Dense(units=20, input_shape=[336,], activation='relu'), 
	layers.Dense(units=20, activation='relu'), 
	layers.Dense(units=1, activation='linear'),
	])
model.summary()

model.compile(optimizer='adam',loss='mean_squared_error')
datasetX = np.copy(slicesX)
datasetY = np.copy(slicesY)

#Train the model on the dataset and save the trained version
history = model.fit(datasetX, datasetY, epochs=300, batch_size=1, verbose=1)
model.save("evolutionaryTestTrained.h5")

#check the models behaviour
plt.plot(model.predict(datasetX))
plt.show()
plt.plot(datasetY)
plt.show()




