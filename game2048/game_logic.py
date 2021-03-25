import numpy as np
import time
from functools import partial


# Looking up a result of moving one row left in the pre-calculated dictionary is
# about 20% faster than calculating it every time.
# If we optimistically take 65536 tile as a maximum of what we expect to encounter on the board,
# the table has 83521 entries. Not much memory for a 20% speedup.

def create_table():
    table = {}
    for a in range(16):
        for b in range(16):
            for c in range(16):
                for d in range(16):
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
    save_file = 'saved_game.npy'

    def __init__(self, score=0, odometer=0, row=None, file=None):
        self.score = score
        self.odometer = odometer
        if row is None:
            self.history = []
            self.row = np.zeros((4, 4), dtype=np.int32)
            self.new_tile()
            self.new_tile()
        else:
            self.row = np.array(row, dtype=np.int32)
        self.history = []
        self.file = file or Game.save_file

    def copy(self):
        return Game(self.score, self.odometer, self.row)

    # a numpy.save method works fine not only for numpy arrays but also for ordinary lists

    def save_game(self, file=None):
        file = file or self.file
        arr = np.array(self.history)
        np.save(file, arr)

    @staticmethod
    def load_game(file=save_file):
        history = np.load(file, allow_pickle=True)
        game = history[-1]
        game.history = history
        return game

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

    def _left(self):
        change = False
        for i in range(4):
            line, score, change_line = Game.table[tuple(self.row[i])]
            if change_line:
                change = True
                self.score += score
                self.row[i] = line
        return change

    # The numpy library has very nice functions of transpose, rot90, ravel etc.
    # No actual number relocation happens, just the "view" is changed. So it's very fast.

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

    # Mostly the same as above but this is a generator for Show.watch method
    # I discovered that as soon as there is "yield" instruction inside the function,
    # python considers the function as a generator ONLY, and you can't return anything else from it.
    # I've read up a couple of discussions on stackoverflow. Seems this is it, no fix.

    @staticmethod
    def generate_trial_run(estimator, game_init=None):
        game = game_init or Game()
        while True:
            if game.game_over():
                return
            best_dir, best_value = 0, - np.inf
            for direction in range(4):
                variant = game.copy()
                change = variant.move(direction)
                if change:
                    value = estimator(variant)
                    if value > best_value:
                        best_dir, best_value = direction, value
            yield game, Game.moves[best_dir]
            game.move(best_dir)
            game.new_tile()

    @staticmethod
    def trial(estimator, num=20, game=None, verbose=False):
        start = time.time()
        results = []
        for i in range(num):
            now = time.time()
            a = Game.trial_run(estimator, game_init=game, verbose=verbose)
            results.append(a)
            fig = 1 << np.max(a.row)
            print(f'game {i}, result {a.score}, moves {a.odometer}, achieved {fig}, time = {(time.time() - now):.2f}')
        average = np.average([a.score for a in results])
        figures = [(1 << np.max(a.row)) for a in results]
        total_odo = sum([a.odometer for a in results])
        results.sort(key=lambda b: b.score, reverse=True)

        def share(limit):
            return len([0 for i in figures if i >= limit]) / len(figures) * 100

        for a in results[:3]:
            print(a)
        elapsed = time.time() - start
        print(f'average score of {num} runs = {average}')
        print(f'8192 reached in {share(8192)}%')
        print(f'4096 reached in {share(4096)}%')
        print(f'2048 reached in {share(2048)}%')
        print(f'1024 reached in {share(1024)}%')
        print(f'total time = {elapsed}')
        print(f'average time per move = {elapsed / total_odo * 1000} ms')
        print(f'total number of shuffles = {Game.counter}')
        print(f'time per shuffle = {elapsed / Game.counter * 1000} ms')
        return results

    # replay game in text mode, for debugging purposes

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

# This is a plain vanilla "let's try to look several moves ahead taking several random tiles at each step".
# Kind of Expectimax algorithm, except we start only when there are few empty cells left and limit the number of random tiles.
# If you choose (depth, width) = (5,3) , the statistics will be roughly the following:
# 1024 = 100%
# 2048 = 62%
# 4096 = 4%
# The game to 2048 takes about 25-30 minutes. Compare this with 1 second (!) for my Q-agent ...
# The average score and most other statistics is way better for Q-agent with one curious exception:
# You can see that look_forward(depth=5, width=3) reaches 1024 basically always.
# Whereas Q_agent reaches 1024 in 94-95% of games, and occasionally stops as low as 256 tile.
# As you can see below, we can put Q_agent valuation (or any other estimator) as an evaluator
# of the last leave in the tree. The results are pretty good, as described in the Readme file.

def look_forward(game_init, depth=1, width=1, empty_limit=7, evaluator=None):
    if depth == 1:
        return evaluator(game_init) if evaluator else game_init.score
    empty = game_init.empty_count()
    num_tiles = min(width, empty)
    if empty > empty_limit:
        num_tiles = 1
        depth = 2
    average = 0
    empty_cells = game_init.empty()
    tile_positions = random.sample(empty_cells, num_tiles)
    for position in tile_positions:
        game_tile = game_init.copy()
        tile = 1 if np.random.randint(10) else 2
        game_tile.new_tile(tile_position=(tile, position))
        best_score = 0
        for direction in range(4):
            game = game_tile.copy()
            change = game.move(direction)
            if change:
                score = look_forward(game, depth - 1, width, empty_limit, evaluator)
                best_score = max(score, best_score)
        average += best_score
    average /= num_tiles
    return average


def estimator_lf(depth=1, width=1, empty_limit=5, evaluator=None):
    return partial(look_forward, depth=depth, width=width, empty_limit=empty_limit, evaluator=evaluator)


def random_eval(game):
    return np.random.random()


if __name__ == "__main__":

    # Just in case you enjoy watching the paint dry
    est = estimator_lf(depth=5, width=3)
    Game.trial(estimator=est, num=100)
