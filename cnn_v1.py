import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as pyplot
import pickle
import os

seed = 7
np.random.seed(seed)
batch_size = 128
epochs = 2000
model_path = 'cb_mse.h5'

# The data, split between train and test sets:
xin = np.load('dataset/idata.npy')
y = np.load('dataset/odata.npy')

x = np.reshape(xin, (5200, 48, 640, 1))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=seed)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

model = Sequential()
model.add(Conv2D(4, (3, 2), strides=(1, 1), input_shape=(48, 640, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(8, (3, 4), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(16, (3, 8), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 16), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(2))
model.add(Activation('linear'))

# initiate optimizer
opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(loss='mse',
              optimizer=opt,
              metrics=['mse', 'mae', 'mape', 'cosine'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
mc = ModelCheckpoint('best_model.h5', monitor='val_mse', mode='min', verbose=1, save_best_only=True)

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test),
                    shuffle=True, verbose=1, callbacks=[es, mc])

model.save(model_path)
print('Saved trained model at %s ' % model_path)

with open('/trainHistoryDict', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test metrics:', scores)

# plot metrics
# pyplot.plot(history.history['mean_squared_error'])
# pyplot.plot(history.history['mean_absolute_error'])
# pyplot.plot(history.history['mean_absolute_percentage_error'])
# pyplot.plot(history.history['cosine_proximity'])
# pyplot.show()

#os.system("sudo shutdown now -h")
