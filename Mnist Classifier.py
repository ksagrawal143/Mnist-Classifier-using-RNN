import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

#The type of RNN cell that we're going to use is the LSTM cell. Layers will have dropout, and we'll have a dense layer at the end, before the output layer.

mnist = tf.keras.datasets.mnist  # mnist is a dataset of 28x28 images of handwritten digits and their labels
(x_train, y_train),(x_test, y_test) = mnist.load_data()  # unpacks images to x_train/x_test and labels to y_train/y_test

x_train = x_train/255.0## Performing normalisation so as to get value between 0 and 1
x_test = x_test/255.0

print(x_train.shape)
print(x_train[0].shape)

model = Sequential()
#Defining the LSTM cells
model.add(LSTM(128, input_shape=(x_train.shape[1:]), activation='relu', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(128, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

# Compiling the RNN model
model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['accuracy'],)
model.fit(x_train,y_train,epochs=3,validation_data=(x_test, y_test))


#Softmax produces the output that represents the probability distribution function of the input matrix.Sum of all the elements is 1.
