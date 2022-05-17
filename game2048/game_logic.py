from .start import *


# basic evaluation methods - take best score, and move randomly
def random_eval(row, score):
    return np.random.random()


def score_eval(row, score):
    return score


# Looking up a result of moving one row left in the pre-calculated dictionary is
# about 20% faster than calculating it every time.
# If we optimistically take 65536 tile as a maximum of what we expect to encounter on the board,
# the table has the same 65536 entries. Not much memory for a 20% speedup.

def create_table():
    table = {}
    for a in range(16):
        for b in range(16):
            for c in range(16):
                for d in range(16):
                    score = 0
                    line = (a, b, c, d)
                    if (len(set(line)) == 4 and min(line)) or (not max(line)):
                        table[line] = (line, score, False)
                        continue
                    line_1 = [v for v in line if v]
                    for i in range(len(line_1) - 1):
                        x = line_1[i]
                        if x == line_1[i + 1]:
                            score += 1 << (x + 1)
                            line_1[i], line_1[i + 1] = x + 1, 0
                    line_2 = [v for v in line_1 if v]
                    line_2 = tuple(line_2 + [0] * (4 - len(line_2)))
                    table[line] = (line_2, score, line != line_2)
    print('table of moves created')
    return table

#   member of the class is the state of the 4*4 board,
#   score = current score in the game
#   odometer = number of moves from the start
#   row = numpy array of shape (4, 4)
#   numbers stored in the Game are 0 for 0 and log2(n) for 2,4,8 ..


class Game:

    actions = {0: 'left', 1: 'up', 2: 'right', 3: 'down'}
    table = create_table()
    counter = 0
    save_file = 'saved_game.pkl'

    def __init__(self, score=0, row=None, file=None):
        self.score = score
        self.odometer = 0
        self.moves = []
        self.tiles = []
        if row is None:
            self.row = np.zeros((4, 4), dtype=np.int32)
            self.new_tile()
            self.new_tile()
            self.tiles = []
            self.starting_position = self.row.copy()
        else:
            self.row = np.array(row, dtype=np.int32)
            self.starting_position = row
        self.file = file or Game.save_file

    def copy(self):
        return Game(self.score, self.row)

    # a numpy.save method works fine not only for numpy arrays but also for ordinary lists

    def save_game(self, file=None):
        file = file or self.file
        with open(file, 'wb') as f:
            pickle.dump(self, f, -1)

    @staticmethod
    def load_game(file=save_file):
        with open(file, 'rb') as f:
            game = pickle.load(f)
        return game

    def __eq__(self, other):
        return np.array_equal(self.row, other.row)

    def __str__(self):
        return '\n'.join([''.join([str(1 << val if val else 0) + '\t' * (4 if (1 << val) < 1000 else 3)
                                   for val in j]) for j in self.row]) \
               + f'\n score = {str(self.score)} moves = {str(self.odometer)} reached {1 << np.max(self.row)}'

    @staticmethod
    def empty(row):
        zeros = np.where(row == 0)
        return list(zip(zeros[0], zeros[1]))

    @staticmethod
    def empty_count(row):
        return 16 - np.count_nonzero(row)

    @staticmethod
    def adjacent_pair_count(row):
        return 24 - np.count_nonzero(row[:, :3] - row[:, 1:]) - np.count_nonzero(row[:3, :] - row[1:, :])

    def game_over(self, row):
        return not self.empty_count(row) and not self.adjacent_pair_count(row)

    def create_new_tile(self, row):
        em = self.empty(row)
        tile = 1 if random.randrange(10) else 2
        position = random.choice(em)
        return tile, position

    def new_tile(self):
        tile, position = self.create_new_tile(self.row)
        self.row[position] = tile
        self.tiles.append((tile, position))

    @staticmethod
    def _left(row, score):
        change = False
        new_row = row.copy()
        new_score = score
        for i in range(4):
            line, score, change_line = Game.table[tuple(row[i])]
            if change_line:
                change = True
                new_score += score
                new_row[i] = line
        return new_row, new_score, change

    def pre_move(self, row, score, direction):
        Game.counter += 1
        new_row = np.rot90(row, direction) if direction else row
        new_row, new_score, change = self._left(new_row, score)
        if direction:
            new_row = np.rot90(new_row, 4 - direction)
        return new_row, new_score, change

    def make_move(self, direction):
        self.row, self.score, change = self.pre_move(self.row, self.score, direction)
        self.odometer += 1
        self.moves.append(direction)
        return change

    def trial_run(self, estimator, limit_tile=0, step_limit=100000, depth=0, width=1, since_empty=0, verbose=False):
        if verbose:
            print('Starting position:')
            print(self)
        while self.odometer < step_limit:
            if self.game_over(self.row):
                return
            if limit_tile and np.max(self.row) >= limit_tile:
                break
            best_dir, best_value = 0, - np.inf
            best_row, best_score = None, None
            for direction in range(4):
                new_row, new_score, change = self.pre_move(self.row, self.score, direction)
                if change:
                    value = self.look_forward(estimator, new_row, new_score, depth=depth, width=width, ample=since_empty)
                    if value > best_value:
                        best_dir, best_value = direction, value
                        best_row, best_score = new_row, new_score
            self.moves.append(best_dir)
            self.odometer += 1
            self.row, self.score = best_row, best_score
            self.new_tile()
            if verbose:
                print(f'On {self.odometer} we move {Game.actions[best_dir]}')
                print(self)

    def generate_run(self, estimator, limit_tile=0, depth=0, width=1, ample=16):
        while True:
            if self.game_over(self.row):
                return
            if limit_tile and np.max(self.row) >= limit_tile:
                break
            best_dir, best_value = 0, - np.inf
            best_row, best_score = None, None
            for direction in range(4):
                new_row, new_score, change = self.pre_move(self.row, self.score, direction)
                if change:
                    value = self.look_forward(estimator, new_row, new_score, depth=depth, width=width, ample=ample)
                    if value > best_value:
                        best_dir, best_value = direction, value
                        best_row, best_score = new_row, new_score
            yield self, best_dir
            self.moves.append(best_dir)
            self.odometer += 1
            self.row, self.score = best_row, best_score
            self.new_tile()

    def look_forward(self, estimator, row, score, depth, width, ample):
        if depth == 0:
            return estimator(row, score)
        empty = self.empty_count(row)
        if empty >= ample:
            return estimator(row, score)
        num_tiles = min(width, empty)
        empty_cells = self.empty(row)
        tile_positions = random.sample(empty_cells, num_tiles)
        worst = np.inf
        average = 0
        for position in tile_positions:
            new_tile = 1 if random.randrange(10) else 2
            new_row = row.copy()
            new_row[position] = new_tile
            if self.game_over(new_row):
                best_value = estimator(row, score)
            else:
                best_value = - np.inf
                for direction in range(4):
                    test_row, test_score, change = self.pre_move(new_row, score, direction)
                    if change:
                        value = self.look_forward(estimator, test_row, test_score,
                                                  depth=depth - 1, width=width, ample=ample)
                        best_value = max(best_value, value)
            worst = min(worst, best_value)
            average += max(best_value, 0)
        average = average / num_tiles
        return average

    @staticmethod
    def trial(estimator, limit_tile=0, num=20, game_init=None, depth=0, width=1, ample=6, verbose=False):
        start = time.time()
        results = []
        for i in range(num):
            now = time.time()
            game = Game() if game_init is None else game_init.copy()
            game.trial_run(estimator, limit_tile=limit_tile, depth=depth, width=width, since_empty=ample, verbose=verbose)
            print(f'game {i}, result {game.score}, moves {game.odometer}, achieved {1 << np.max(game.row)}, '
                  f'time = {(time.time() - now):.2f}')
            results.append(game)
        average = np.average([v.score for v in results])
        figures = [(1 << np.max(v.row)) for v in results]
        total_odo = sum([v.odometer for v in results])
        results.sort(key=lambda v: v.score, reverse=True)

        def share(limit):
            return len([0 for v in figures if v >= limit]) / len(figures) * 100

        print('\nBest games:\n')
        for v in results[:3]:
            print(v, '\n')
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

    def replay(self, verbose=True):
        chain = {}
        state = self.starting_position
        replay_game = Game(row=state)
        if verbose:
            print('Starting position:')
            print(replay_game)
        for i in range(self.odometer):
            move = self.moves[i]
            chain[i] = (replay_game.row.copy(), replay_game.score, move)
            new_tile, position = self.tiles[i]
            if verbose:
                print(i, new_tile, position)
            replay_game.make_move(move)
            replay_game.row[position] = new_tile
            if verbose:
                print(f'On {replay_game.odometer} we move = {Game.actions[move]}, '
                      f'new tile = {new_tile} at position = {position}')
                print(replay_game)
        if verbose:
            print('no more moves possible, final position')
        chain[self.odometer] = (self.row.copy(), self.score, -1)
        return chain


if __name__ == "__main__":

    # Just in case you enjoy watching the paint dry

    est = random_eval
    g = Game()
    g.trial_run(est, verbose=False)

