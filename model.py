import csv 
import cv2
import numpy as np 

# load csv file
lines = []
with open('../data3/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

# load images data and steering angle data
# using multiple cameras and correction parameter
images = []
measurements = []
lines.remove(lines[0])
for line in lines:
	for i in range(3):
		source_path = line[i]
		filename = source_path.split('/')[-1]
		current_path = '../data3/IMG/' + filename
		image = cv2.imread(current_path)
		images.append(image)
		measurement = float(line[3])
		if i == 1:
			measurements.append(measurement + 0.18)
		elif i == 2:
			measurements.append(measurement - 0.18)
		else:
			measurements.append(measurement)

# To get training data and validation data 
X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D

# Construcet a Convolutional Neural Networks Model 
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3))) # Normalize the data
model.add(Cropping2D(cropping=((70,25),(0,0)))) #Crop the picture

# Five convolutional layers
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu")) 
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))

# Four fully connected layers
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Use an adam optimizer
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

# save model.h5 to run in the simulator
model.save('model.h5')