

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
# from keras_LRFinder import LRFinder
import matplotlib.pyplot as plt

from keras.callbacks import Callback, LearningRateScheduler
import keras.backend as K

import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import load_model
from keras.utils import plot_model

def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * np.power(drop, np.floor((1 + epoch) / epochs_drop))
    return lrate


# --- reading the dataset
df = pd.read_csv('pima-indians-diabetes.csv')
data = df.values
X = data[:, :8]
Y = data[:, 8]

batch_size = 8
epochs = 400
rate = 0.5
lr = 0.0001

encoded_Y = to_categorical(Y)


X_train, X_test, y_train, y_test = train_test_split(X, encoded_Y, test_size=0.3, random_state=7, shuffle=False)

# as you can see, you've the same number of rows 891
# but now you've so many more columns due to how we changed all the categorical data into numerical data
check = keras.callbacks.ModelCheckpoint('output/{val_acc:.4f}_P.hdf5', monitor='val_acc', verbose=0,
                                        save_best_only=True, save_weights_only=False, mode='auto', period=1)

scaler = MinMaxScaler()
scaled_x_train = scaler.fit_transform(X_train)
scaled_x_test = scaler.transform(X_test)

decay = lr / epochs

#adam=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.000001, amsgrad=False)
adam=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.9, epsilon=None, decay=0.000001, amsgrad=False)
#opt = keras.optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=False)
# opt = keras.optimizers.Adadelta(lr=0.0001, rho=0.95, epsilon=None, decay=1e-6)


model = load_model('output/0.8052_N.hdf5')
for layer in model.layers:
    layer.trainable = True

for layer in model.layers[:-1]:
    layer.trainable = False
for layer in model.layers:
    print(layer, layer.trainable)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

lrate = LearningRateScheduler(step_decay)
# Fit the model
history = model.fit(scaled_x_train, y_train,
                    validation_data=(scaled_x_test, y_test),
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=False,
                    callbacks=[check])


# --------- evaluate the model ----------
scores = model.evaluate(scaled_x_test, y_test)
print()
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

# ------- summarize history for accuracy -------
plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.show()

# ----------summarize history for loss ---------
plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

# ------ plotting the model definition -----
plot_model(model, to_file='Final_model.png')