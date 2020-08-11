import numpy as np
from random import random, randrange
from game2048.game_logic import *
import json
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Concatenate, Lambda, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model


def prepare_input(game):
    X = game.row.copy()[..., np.newaxis]
    X_av = np.array([0, 0, 0, 0])
    for direction in range(4):
        test = game.copy()
        X_av[direction] = int(test.move(direction))
    return X, X_av


def prepare_input_batch(states):
    m = len(states)
    batch_X = np.zeros((m, 4, 4, 1))
    batch_av = np.zeros((m, 4, ))
    for i in (range(m)):
        batch_X[i], batch_av[i] = prepare_input(states[i])
    return batch_X, batch_av


def create_model(input_shapes=((4, 4, 1), (4, ))):
    sh1, sh2 = input_shapes
    row, available_moves = Input(sh1), Input(sh2)

    V1 = Conv2D(32, kernel_size=(1, 2), activation='relu')(row)
    V2 = Conv2D(32, kernel_size=(2, 1), activation='relu')(row)

    V11 = Conv2D(128, kernel_size=(1, 2), activation='relu')(V1)
    V12 = Conv2D(128, kernel_size=(2, 1), activation='relu')(V1)

    V21 = Conv2D(128, kernel_size=(1, 2), activation='relu')(V2)
    V22 = Conv2D(128, kernel_size=(2, 1), activation='relu')(V2)

    F1 = Concatenate()([Flatten()(V11), Flatten()(V12), Flatten()(V21), Flatten()(V22)])
    F2 = Dense(256, activation='relu')(F1)
    F3 = Dense(256, activation='relu')(F2)
    F4 = Dense(4, activation='relu')(F3)
    out = Multiply()([F4, available_moves])
    model = Model(inputs=[row, available_moves], outputs=out)
    return model


def forward_propagate():
    game = Game()
    states = []
    labels = []
    while not game.game_over():
        X = prepare_input_batch([game])
        values = mod.predict(X)[0]
        list_of_moves = [0, 1, 2, 3]
        action = np.argmax(values)
        next_state = game.copy()
        change = next_state.move(action)
        if not change:
            values[action] = 0
            list_of_moves.remove(action)
        if random() < epsilon or not change:
            while True:
                action = list_of_moves[randrange(len(list_of_moves))]
                next_state = game.copy()
                change = next_state.move(action)
                if change:
                    break
                else:
                    values[action] = 0
                    list_of_moves.remove(action)
        next_state.new_tile()
        next_X = prepare_input_batch([next_state])
        next_values = mod.predict(next_X)
        max_value = np.max(next_values)
        reward = np.log(next_state.score + 1) - np.log(game.score + 1)
        values[action] = reward + gamma * max_value
        states.append(game)
        labels.append(values)
        game.move(action)
        game.new_tile()
    game.history.append(game)
    states.append(game)
    labels.append([0, 0, 0, 0])
    return game, states, labels


gamma = 1
epsilon = 0.1
lr = 0.01
epochs = 10
episodes = 100
now = time.time()
num_moves = []

mod = create_model()
opt = tf.keras.optimizers.SGD(learning_rate=lr)
mod.compile(optimizer=opt, loss='mean_squared_error')
mod.summary()

for episode in range(episodes):
    game, states, labels = forward_propagate()
    X = prepare_input_batch(states)
    labels = np.array(labels)
    mod.fit(X, labels, epochs=epochs, verbose=0)
    if episode > 50:
        del num_moves[0]
    num_moves.append(game.score)
    average = np.mean(num_moves)
    print(f'ep {episode}: time {time.time() - now}, score {game.score}, moves {game.odometer}, 50-ma = {average}')
    if episode % 1000 == 0 and episode:
        q = input()
        if q == 'q':
            break

mod.save('nn_model')

test_state = [[1, 2, 3, 1],
         [5, 1, 2, 3],
         [1, 2, 1, 2],
         [2, 3, 4, 1]]
a = Game(row=test_state)
X = prepare_input_batch([a])
values = mod.predict(X)[0]
print(values)