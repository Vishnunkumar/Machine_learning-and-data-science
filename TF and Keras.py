import tensorflow as tf
import keras

mnist = keras.datasets.mnist

#splitting datasets into testing and training
(x_train,y_train), (x_test,y_test) = mnist.load_data()
x_train = keras.utils.normalize(x_train,axis=1)
x_test = keras.utils.normalize(x_test,axis=1)

#model description
model = keras.models.Sequential()

model.add(keras.layers.Convolution1D(128,kernel_size=3,activation=tf.nn.relu))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128,activation=tf.nn.relu))
model.add(keras.layers.Dense(128,activation=tf.nn.relu))
model.add(keras.layers.Dense(10,activation=tf.nn.softmax))

#Model compilation
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#modelfitting
model.fit(x_train,y_train,epochs=3)
val_loss, val_score = model.evaluate(x_test, y_test)
print(val_loss)
print(val_score)

#checking accuracy manually
prediction = model.predict_classes(x_test)
prediction.resize(10000,1)

Error = []
for i in range(len(prediction)):
    if prediction[i] == y_test[i]:
        Error.append(0)
    else:
        Error.append(1)
        
import matplotlib.pyplot as plt
import numpy as np
plt.scatter(np.arange(1,100),Error[1:100])

model.save('my_model')