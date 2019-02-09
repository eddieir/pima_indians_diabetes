import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
#from keras_LRFinder import LRFinder
import matplotlib.pyplot as plt
from keras.utils import plot_model


from keras.callbacks import Callback
import keras.backend as K

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix,classification_report



# --- reading the dataset
df= pd.read_csv('pima-indians-diabetes.csv')
data=df.values
X = data[:,:8]
Y = data[:,8]

batch_size = 16
epochs=200
rate=0.5

encoded_Y = to_categorical(Y)

X_train, X_test, y_train, y_test = train_test_split(X,encoded_Y,test_size=0.3,random_state=7,shuffle=False)




# as you can see, you've the same number of rows 891
# but now you've so many more columns due to how we changed all the categorical data into numerical data
check=keras.callbacks.ModelCheckpoint('output/{val_acc:.4f}_N.hdf5', monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

scaler = MinMaxScaler()
scaled_x_train = scaler.fit_transform(X_train)
scaled_x_test = scaler.transform(X_test)

adam = keras.optimizers.Adam(lr=0.001)

model = keras.models.Sequential()
model.add(keras.layers.Dense(8, input_dim=8, kernel_initializer='uniform'))
model.add(keras.layers.LeakyReLU(alpha=0.2))
model.add(keras.layers.Dense(64, kernel_initializer='uniform'))
model.add(keras.layers.LeakyReLU(alpha=0.2))
keras.layers.Dropout(rate, noise_shape=None, seed=None)
model.add(keras.layers.Dense(128, kernel_initializer='uniform'))
model.add(keras.layers.LeakyReLU(alpha=0.2))
keras.layers.Dropout(rate, noise_shape=None, seed=None)
model.add(keras.layers.Dense(64, kernel_initializer='uniform'))
model.add(keras.layers.LeakyReLU(alpha=0.2))
keras.layers.Dropout(rate, noise_shape=None, seed=None)
model.add(keras.layers.Dense(2,  kernel_initializer='uniform', activation='softmax'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Find learning rate
#lr_finder = LRFinder(min_lr=1e-8, max_lr=1e-2, step_size=np.ceil(scaled_x_train.shape[0]/batch_size))

# Fit the model
history=model.fit(scaled_x_train, y_train,
                  validation_data=(scaled_x_test, y_test),
                  epochs=epochs,
                  batch_size=batch_size,
                  shuffle=False,
                  callbacks=[check])
# evaluate the model
scores = model.evaluate(scaled_x_test, y_test)
print()
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
plot_model(model, to_file='model.png')
#lr_finder.plot_avg_loss()

# summarize history for accuracy
plt.figure(1)
plt.title('accuracy')
plt.plot(history.history['acc'], label='training_accuracy')
plt.plot(history.history['val_acc'], label = 'testing_accuracy')
plt.legend()
plt.show()

# summarize history for loss
plt.figure(2)
plt.title('loss')
plt.plot(history.history['loss'], label='testing_loss')
plt.plot(history.history['val_loss'],label='training_loss')
plt.legend()
plt.show()

y_pred = model.predict(scaled_x_test)
print(classification_report(np.argmax(y_pred,axis=1),np.argmax(y_test,axis=1)))
print(confusion_matrix(np.argmax(y_pred,axis=1),np.argmax(y_test,axis=1)))