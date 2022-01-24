import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

datasetX = np.zeros((14,200))
datasetY = np.zeros(200)

#generate synthetic training data. See full text for explanation
for i in range(200):
	datasetY[i] = np.exp(i/100.0) / 10.0

for i in range(14):
	for j in range(200):
		t = 1.0/200.0
		datasetX[i, j] = t*j

#reorganise sensor data into windowed time slices as per netwroks requirements
slicesX = np.zeros((170,14,30))
slicesY = np.zeros(170)
for i in range(170):
	working = np.zeros(30)
	for j in range(14):
		slicesX[i,j, :] = datasetX[j, i:i+30]
	print(slicesX[i,:10])
	slicesY[i] = datasetY[i+30]


#Define the network using Keras - mostly layers of 14 parallel 1D convolutions...
fn=1
initializer = tf.keras.initializers.GlorotNormal()
in1 = layers.Input(shape=(30,1))
in2 = layers.Input(shape=(30,1))
in3= layers.Input(shape=(30,1))
in4 = layers.Input(shape=(30,1))
in5 = layers.Input(shape=(30,1))
in6 = layers.Input(shape=(30,1))
in7 = layers.Input(shape=(30,1))
in8 = layers.Input(shape=(30,1))
in9 = layers.Input(shape=(30,1))
in10 = layers.Input(shape=(30,1))
in11 = layers.Input(shape=(30,1))
in12 = layers.Input(shape=(30,1))
in13 = layers.Input(shape=(30,1))
in14 = layers.Input(shape=(30,1))

#14
conv1_1 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv1_1')(in1)
conv2_1 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv2_1')(in2)
conv3_1 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv3_1')(in3)
conv4_1 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv4_1')(in4)
conv5_1 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv5_1')(in5)
conv6_1 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv6_1')(in6)
conv7_1 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv7_1')(in7)
conv8_1 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv8_1')(in8)
conv9_1 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv9_1')(in9)
conv10_1 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv10_1')(in10)
conv11_1 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv11_1')(in11)
conv12_1 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv12_1')(in12)
conv13_1 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv13_1')(in13)
conv14_1 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv14_1')(in14)

#28
conv1_2 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv1_2')(conv1_1)
conv2_2 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv2_2')(conv2_1)
conv3_2 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv3_2')(conv3_1)
conv4_2 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv4_2')(conv4_1)
conv5_2 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv5_2')(conv5_1)
conv6_2 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv6_2')(conv6_1)
conv7_2 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv7_2')(conv7_1)
conv8_2 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv8_2')(conv8_1)
conv9_2 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv9_2')(conv9_1)
conv10_2 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv10_2')(conv10_1)
conv11_2 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv11_2')(conv11_1)
conv12_2 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv12_2')(conv12_1)
conv13_2 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv13_2')(conv13_1)
conv14_2 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv14_2')(conv14_1)

#42
conv1_3 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv1_3')(conv1_2)
conv2_3 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv2_3')(conv2_2)
conv3_3 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv3_3')(conv3_2)
conv4_3 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv4_3')(conv4_2)
conv5_3 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv5_3')(conv5_2)
conv6_3 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv6_3')(conv6_2)
conv7_3 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv7_3')(conv7_2)
conv8_3 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv8_3')(conv8_2)
conv9_3 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv9_3')(conv9_2)
conv10_3 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv10_3')(conv10_2)
conv11_3 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv11_3')(conv11_2)
conv12_3 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv12_3')(conv12_2)
conv13_3 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv13_3')(conv13_2)
conv14_3 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv14_3')(conv14_2)

#56
conv1_4 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv1_4')(conv1_3)
conv2_4 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv2_4')(conv2_3)
conv3_4 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv3_4')(conv3_3)
conv4_4 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv4_4')(conv4_3)
conv5_4 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv5_4')(conv5_3)
conv6_4 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv6_4')(conv6_3)
conv7_4 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv7_4')(conv7_3)
conv8_4 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv8_4')(conv8_3)
conv9_4 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv9_4')(conv9_3)
conv10_4 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv10_4')(conv10_3)
conv11_4 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv11_4')(conv11_3)
conv12_4 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv12_4')(conv12_3)
conv13_4 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv13_4')(conv13_3)
conv14_4 = layers.Conv1D(filters=fn, kernel_size=10, padding='same', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv14_4')(conv14_3)

#70
conv1_5 = layers.Conv1D(filters=fn, kernel_size=3, padding='valid', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv1_5')(conv1_4)
conv2_5 = layers.Conv1D(filters=fn, kernel_size=3, padding='valid', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv2_5')(conv2_4)
conv3_5 = layers.Conv1D(filters=fn, kernel_size=3, padding='valid', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv3_5')(conv3_4)
conv4_5 = layers.Conv1D(filters=fn, kernel_size=3, padding='valid', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv4_5')(conv4_4)
conv5_5 = layers.Conv1D(filters=fn, kernel_size=3, padding='valid', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv5_5')(conv5_4)
conv6_5 = layers.Conv1D(filters=fn, kernel_size=3, padding='valid', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv6_5')(conv6_4)
conv7_5 = layers.Conv1D(filters=fn, kernel_size=3, padding='valid', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv7_5')(conv7_4)
conv8_5 = layers.Conv1D(filters=fn, kernel_size=3, padding='valid', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv8_5')(conv8_4)
conv9_5 = layers.Conv1D(filters=fn, kernel_size=3, padding='valid', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv9_5')(conv9_4)
conv10_5 = layers.Conv1D(filters=fn, kernel_size=3, padding='valid', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv10_5')(conv10_4)
conv11_5 = layers.Conv1D(filters=fn, kernel_size=3, padding='valid', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv11_5')(conv11_4)
conv12_5 = layers.Conv1D(filters=fn, kernel_size=3, padding='valid', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv12_5')(conv12_4)
conv13_5 = layers.Conv1D(filters=fn, kernel_size=3, padding='valid', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv13_5')(conv13_4)
conv14_5 = layers.Conv1D(filters=fn, kernel_size=3, padding='valid', kernel_initializer=initializer, use_bias=True, activation='tanh', name='conv14_5')(conv14_4)

#84
flt1 = layers.Flatten()(conv1_5)
flt2 = layers.Flatten()(conv2_5)
flt3 = layers.Flatten()(conv3_5)
flt4 = layers.Flatten()(conv4_5)
flt5 = layers.Flatten()(conv5_5)
flt6 = layers.Flatten()(conv6_5)
flt7 = layers.Flatten()(conv7_5)
flt8 = layers.Flatten()(conv8_5)
flt9 = layers.Flatten()(conv9_5)
flt10 = layers.Flatten()(conv10_5)
flt11 = layers.Flatten()(conv11_5)
flt12 = layers.Flatten()(conv12_5)
flt13 = layers.Flatten()(conv13_5)
flt14 = layers.Flatten()(conv14_5)

#98
mrg = layers.concatenate([flt1, flt2, flt3, flt4, flt5, flt6, flt7, flt8, flt9, flt10, flt11, flt12, flt13, flt14], axis=1)

#99
denseWide = layers.Dense(100, activation='tanh')(mrg)
op = layers.Dense(1, activation='linear')(denseWide)

model = keras.Model(inputs=[in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12, in13, in14], outputs=op)
model.compile(optimizer="Adam", metrics=['accuracy'], loss=tf.keras.losses.MeanSquaredError())
print(model.summary())


#Train the network and save the trained version
history = model.fit([slicesX[:,0,:], slicesX[:,1,:], slicesX[:,2,:], slicesX[:,3,:], slicesX[:,4,:], slicesX[:,5,:], slicesX[:,6,:], slicesX[:,7,:], slicesX[:,8,:], slicesX[:,9,:], slicesX[:,10,:], slicesX[:,11,:], slicesX[:,12,:], slicesX[:,13,:]], slicesY, batch_size=100, epochs=2000)
model.save("convolutionalTestTrained.h5")

#check predictions
plt.plot(model.predict([slicesX[:,0,:], slicesX[:,1,:], slicesX[:,2,:], slicesX[:,3,:], slicesX[:,4,:], slicesX[:,5,:], slicesX[:,6,:], slicesX[:,7,:], slicesX[:,8,:], slicesX[:,9,:], slicesX[:,10,:], slicesX[:,11,:], slicesX[:,12,:], slicesX[:,13,:]]))
plt.show()
plt.plot(datasetY)
plt.show()




