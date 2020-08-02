import numpy as np
from random import random, randrange
from game2048.game_logic import *
import json
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Concatenate, Lambda, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
# from rl.agents.dqn import DQNAgent
# from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
# from rl.memory import SequentialMemory
# from rl.core import Processor
# from rl.callbacks import FileLogger, ModelIntervalCheckpoint


def get_epsilon(n):
    return 1 / (np.log((n >> 7) + 1) + 1)


def prepare_input(game):
    X = np.array(game.row)[..., np.newaxis]
    X = (X - np.mean(X)) / 10
    X_vert = X[:3, :] - X[1:, :]
    X_hor = X[:, :3] - X[:, 1:]
    X_av = np.array([0, 0, 0, 0])
    X = X.reshape((16,))
    for direction in range(4):
        test = game.copy()
        X_av[direction] = int(test.move(direction))
    return X, X_vert, X_hor, X_av


def prepare_input_from_game(states):
    m = len(states)
    batch_X = np.zeros((m, 16, ))
    batch_vert = np.zeros((m, 3, 4, 1))
    batch_hor = np.zeros((m, 4, 3, 1))
    batch_av = np.zeros((m, 4, ))
    for i in (range(m)):
        batch_X[i], batch_vert[i], batch_hor[i], batch_av[i] = prepare_input(states[i])
    return batch_X, batch_vert, batch_hor, batch_av


def create_model(input_shapes=((16, ), (3, 4, 1), (4, 3, 1), (4, ))):
    sh1, sh2, sh3, sh4 = input_shapes
    row, vertical_grad, horisontal_grad, available_moves = Input(sh1), Input(sh2), Input(sh3), Input(sh4)
    R10 = Dense(256, activation='relu')(row)
    R4 = Flatten()(R10)
    V1 = Conv2D(32, kernel_size=(2, 1), activation='relu', name='vertical_1')(vertical_grad)
    V2 = Conv2D(32, kernel_size=(1, 2), activation='relu', name='vertical_2')(vertical_grad)
    V3 = Flatten()(V1)
    V4 = Flatten()(V2)
    H1 = Conv2D(32, kernel_size=(2, 1), activation='relu', name='horisontal_1')(horisontal_grad)
    H2 = Conv2D(32, kernel_size=(1, 2), activation='relu', name='horisontal_2')(horisontal_grad)
    H3 = Flatten()(H1)
    H4 = Flatten()(H2)
    F1 = Concatenate()([R4, Flatten()(vertical_grad), Flatten()(horisontal_grad), V3, V4, H3, H4])
    F2 = Dense(256, activation='relu')(F1)
    F3 = Dense(64, activation='relu')(F2)
    F4 = Dense(4, activation='linear')(F3)
    out = Multiply()([F4, available_moves])
    model = Model(inputs=[row, vertical_grad, horisontal_grad, available_moves], outputs=out, name='my2048')
    return model


def forward_propagate(episode):
    #epsilon = get_epsilon(episode)
    game = Game()
    states = []
    labels = []
    while not game.game_over():
        X = prepare_input_from_game([game])
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
        next_X = prepare_input_from_game([next_state])
        next_values = mod.predict(next_X)
        max_value = np.max(next_values)
        reward = np.log2(next_state.score - game.score + 1) + next_state.pair_count() - game.pair_count()
        values[action] = reward + gamma * max_value
        states.append(game)
        labels.append(values)
        game.move(action)
        game.new_tile()
    states.append(game)
    labels.append([0, 0, 0, 0])
    return game, states, labels


gamma = 0.95
epsilon = 0.05
lr = 0.01
epochs = 1
episodes = 10000
now = time.time()
num_moves = []

mod = create_model()
opt = tf.keras.optimizers.SGD(learning_rate=lr)
mod.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
mod.summary()

for episode in range(episodes):
    game, states, labels = forward_propagate(episode)
    X = prepare_input_from_game(states)
    labels = np.array(labels)
    mod.fit(X, labels, epochs=epochs, verbose=0)
    if episode > 50:
        del num_moves[0]
    num_moves.append(game.odometer)
    average = np.mean(num_moves)
    print(f'ep {episode}: time {time.time() - now}, score {game.score}, moves {game.odometer}, 50-ma = {average}')
    if episode % 1000 == 0 and episode:
        q = input()
        if q == 'q':
            break

mod.save('nn_model')

state = [[1, 2, 3, 1],
         [5, 1, 2, 3],
         [1, 2, 1, 2],
         [2, 3, 4, 1]]
a = Game(row=state)
X = prepare_input_from_game([a])
values = mod.predict(X)[0]
print(values)
