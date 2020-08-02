import numpy as np
import time
from functools import partial


def create_table():
    table = {}
    for a in range(17):
        for b in range(17):
            for c in range(17):
                for d in range(17):
                    score = 0
                    change = False
                    line = (a, b, c, d)
                    if (len(set(line)) == 4 and min(line)) or (not max(line)):
                        table[line] = (line, score, change)
                        continue
                    line_1 = [v for v in line if v]
                    for i in range(len(line_1) - 1):
                        x = line_1[i]
                        if x == line_1[i + 1]:
                            score += 1 << (x + 1)
                            change = True
                            line_1[i], line_1[i + 1] = x + 1, 0
                    line_2 = [v for v in line_1 if v]
                    line_2.extend([0] * (4 - len(line_2)))
                    if tuple(line_2) == line:
                        table[line] = (line, score, change)
                    else:
                        change = True
                        table[line] = (line_2, score, change)
    print('table of moves created')
    return table

#   member of the class is the state of the 4*4 board,
#   score = current score in the game
#   odometer = number of moves from the start
#   row = numpy array of shape (4, 4)
#   numbers stored in the Game are 0 for 0 and log2(n) for 2,4,8 ..


class Game:

    moves = {0: 'left', 1: 'up', 2: 'right', 3: 'down'}
    table = create_table()
    counter = 0

    def __init__(self, score=0, odometer=0, row=None):
        self.score = score
        self.odometer = odometer
        if row is None:
            self.history = []
            self.row = np.zeros((4, 4), dtype=np.int8)
            self.new_tile()
            self.new_tile()
        else:
            self.row = np.array(row, dtype=np.int8)
        self.history = []

    def copy(self):
        return Game(self.score, self.odometer, self.row)

    def __eq__(self, other):
        return np.array_equal(self.row, other.row)

    def __str__(self):
        return '\n'.join(['\t\t\t\t'.join([str(1 << val if val else 0) for val in j]) for j in self.row]
                         ) + '\n score = ' + str(self.score) + ' odometer = ' + str(self.odometer)

    def empty(self):
        return [(i, j) for j in range(4) for i in range(4) if self.row[i, j] == 0]

    def empty_count(self):
        return 16 - np.count_nonzero(self.row)

    def pair_count(self):
        state = self.row
        zero = np.count_nonzero(state[:, :3] - state[:, 1:]) + np.count_nonzero(state[:3, :] - state[1:, :])
        return 24 - zero

    def game_over(self):
        return not self.empty_count() and not self.pair_count()

    def new_tile(self, tile_position=None, inplace=True):
        if not tile_position:
            em = self.empty()
            if len(em) == 0:
                return False, (0, (0, 0))
            tile = 1 if np.random.randint(10) else 2
            position = em[np.random.randint(0, len(em))]
        else:
            tile, position = tile_position
        if inplace:
            self.row[position] = tile
            self.history.append((tile << 1, position))
        return tile, position

    # This could be a 8-10 line function. Compress zeros, merge, compress again,
    # and check if the move was valid by comparing arrays etc.
    # But that is significantly slower compared to the below brute-force enumeration of options.
    # The function gets executed gazillions of times during NN training, so speed is important.
    def _left(self):
        change = False
        for i in range(4):
            line, score, change_line = Game.table[tuple(self.row[i])]
            if change_line:
                change = True
                self.score += score
                self.row[i] = line
        return change

    def move(self, direction):
        Game.counter += 1
        self.history.append(self.copy())
        if direction:
            self.row = np.rot90(self.row, direction)
        change = self._left()
        if direction:
            self.row = np.rot90(self.row, 4 - direction)
        if change:
            self.odometer += 1
            self.history.append(direction)
        return change

    @staticmethod
    def trial_run(estimator, game_init=None, step_limit=100000, verbose=False):
        game = game_init or Game()
        while game.odometer < step_limit:
            if game.game_over():
                game.history.append(game)
                return game
            best_dir, best_value = 0, - np.inf
            for direction in range(4):
                variant = game.copy()
                change = variant.move(direction)
                if change:
                    value = estimator(variant)
                    if value > best_value:
                        best_dir, best_value = direction, value
            if verbose:
                print(game.odometer, ' ', Game.moves[best_dir])
                print(game)
            game.move(best_dir)
            game.new_tile()
        game.history.append(game)
        return game

    @staticmethod
    def trial(estimator, num=20, game=None, verbose=False):
        start = time.time()
        results = []
        for i in range(num):
            a = Game.trial_run(estimator, game_init=game, verbose=verbose)
            results.append(a)
            print(f'game {i}, result {a.score}')
        average = np.average([a.score for a in results])
        results.sort(key=lambda b: b.score, reverse=True)
        for a in results[:3]:
            print(a)
        elapsed = time.time() - start
        print(f'average score of {num} runs = {average}')
        print(f'total time = {elapsed}')
        print(f'total number of moves = {Game.counter}')
        print(f'time per move = {elapsed / Game.counter * 1000} ms')
        return results

    def replay(self):
        for step in range(0, len(self.history) - 1, 3):
            state = self.history[step]
            move = Game.moves[self.history[step + 1]]
            new_tile, position = self.history[step + 2]
            print(state)
            print(f'next move = {move}')
            print(f'new tile = {new_tile} at position = {position}')
        print('no more moves possible, final position =')
        print(self.history[-1])

    def batch_from_history(self):
        m = len(self.history) // 3
        top = self.odometer - 1
        X = np.zeros((m, 4, 4, 1))
        y = np.zeros((m, 1))
        for i in range(m):
            game = self.history[3 * i]
            X[i, :, :, 0] = np.array(game.row, dtype=np.int8)
            y[i] = top - game.odometer

        return X, y


def look_forward(game_init, depth=0, width=1):
    empty = game_init.empty_count()
    num_tiles = min(width, empty)
    average = 0
    for i in range(num_tiles):
        game_tile = game_init.copy()
        game_tile.new_tile()
        best_score = game_init.score
        for direction in range(4):
            game = game_tile.copy()
            change = game.move(direction)
            if change:
                if depth:
                    score = look_forward(game, depth - 1, width)
                else:
                    score = game.score
                best_score = max(score, best_score)
        average += best_score
    average /= num_tiles
    return average


def estimator_lf(depth=0, width=1):
    return partial(look_forward, depth=depth, width=width)
