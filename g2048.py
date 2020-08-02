import numpy as np
from game2048.game_logic import *
import json
#import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Conv2D, Flatten, BatchNormalization
import os
import matplotlib.pyplot as plt


def create_model():
    model = Sequential()
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=(2, 1), activation='relu', input_shape=(4, 4, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=(1, 2), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu'))
    model.add(Flatten())
    # model.add(Dense(64, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model


def estimator_nn(game):
    mod = tf.keras.models.load_model('nn_model')
    xt = np.zeros((1, 4, 4, 1))
    xt[0, :, :, 0] = game.row.copy()
    return mod.predict(xt)


def random_estimator(game):
    return np.random.random()

a = Game.trial_run(estimator_lf())
a.replay()


'''
a = Game.trial_run(estimator_nn, verbose=True)
print(a)

mod = create_model()
mod.build((None, 4, 4, 1))
mod.summary()

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)



mod = tf.keras.models.load_model('nn_model')


a = Game.trial_run(estimator_lf)
X, y = Game.batch_from_history(a)
print(a)
mod.fit(X, y, batch_size=1024, verbose=2, epochs=2000)
mod.evaluate(X, y)
y_pred = mod.predict(X)
for i in range(len(y)):
    print(y[i], ' = ', y_pred[i])



losses = []
for i in range(500):
    a = Game.trial_run(estimator_lf)
    X, y = Game.batch_from_history(a)
    result = mod.fit(X, y, batch_size=1024, verbose=0, epochs=100)
    mod.evaluate(X, y)
    losses.append(result.history['loss'][-1])
    print(i, ' ', len(y), ' ', result.history['loss'][-1])

plt.plot(losses)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('run')
plt.show()

y_pred = mod.predict(X)
for i in range(len(y)):
    print(y[i], ' = ', y_pred[i])

mod.save('nn_model')
'''


'''
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
'''