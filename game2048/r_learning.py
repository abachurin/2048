import os

from .game_logic import *


def check_thread(parent, benchmark):
    now = time.time()
    if (now - benchmark) > 2 * dash_intervals['check_run']:
        if RUNNING[parent] == 0:
            return 0
        RUNNING[parent] = 0
        return now
    return benchmark


# features = all adjacent pairs
def f_2(x):
    x_vert = ((x[:3, :] << 4) + x[1:, :]).ravel()
    x_hor = ((x[:, :3] << 4) + x[:, 1:]).ravel()
    return np.concatenate([x_vert, x_hor])


# features = all adjacent triples, i.e. 3 in a row + 3 in a any square missing one corner
def f_3(x):
    x_vert = ((x[:2, :] << 8) + (x[1:3, :] << 4) + x[2:, :]).ravel()
    x_hor = ((x[:, :2] << 8) + (x[:, 1:3] << 4) + x[:, 2:]).ravel()
    x_ex_00 = ((x[1:, :3] << 8) + (x[1:, 1:] << 4) + x[:3, 1:]).ravel()
    x_ex_01 = ((x[:3, :3] << 8) + (x[1:, :3] << 4) + x[1:, 1:]).ravel()
    x_ex_10 = ((x[:3, :3] << 8) + (x[:3, 1:] << 4) + x[1:, 1:]).ravel()
    x_ex_11 = ((x[:3, :3] << 8) + (x[1:, :3] << 4) + x[:3, 1:]).ravel()
    return np.concatenate([x_vert, x_hor, x_ex_00, x_ex_01, x_ex_10, x_ex_11])


# Initially i also made all adjacent quartets of different shape, but the learning was not happening.
# My theory is that: 1) we want our features to intersect and correlate (otherwise we will only learn
# some several separate pieces of the board, and that obviously can not lead to anything.
# but 2) we don't want them to intersect too much (like 3 cells common to two quartets), as they start
# to kinda suppress and contradict each other.
# So i left just columns, rows and squares. 17 features all in all. And it works just fine.
def f_4(x):
    x_vert = ((x[0, :] << 12) + (x[1, :] << 8) + (x[2, :] << 4) + x[3, :]).ravel()
    x_hor = ((x[:, 0] << 12) + (x[:, 1] << 8) + (x[:, 2] << 4) + x[:, 3]).ravel()
    x_sq = ((x[:3, :3] << 12) + (x[1:, :3] << 8) + (x[:3, 1:] << 4) + x[1:, 1:]).ravel()
    return np.concatenate([x_vert, x_hor, x_sq])


# Finally, we try adding 4 "cross" 5-features for middle cells
def f_5(x):
    x_vert = ((x[0, :] << 12) + (x[1, :] << 8) + (x[2, :] << 4) + x[3, :]).ravel()
    x_hor = ((x[:, 0] << 12) + (x[:, 1] << 8) + (x[:, 2] << 4) + x[:, 3]).ravel()
    x_sq = ((x[:3, :3] << 12) + (x[1:, :3] << 8) + (x[:3, 1:] << 4) + x[1:, 1:]).ravel()
    x_middle = ((x[1: 3, 1: 3] << 16) + (x[:2, 1: 3] << 12) + (x[1: 3, :2] << 8) + (x[2:, 1: 3] << 4) + x[1: 3, 2:]
                ).ravel()
    return np.concatenate([x_vert, x_hor, x_sq, x_middle])


# Let's try to add some limited 6-features, up to < 2 ** (cutoff - 1) > tile
def f_6(x):
    x_vert = ((x[0, :] << 12) + (x[1, :] << 8) + (x[2, :] << 4) + x[3, :]).ravel()
    x_hor = ((x[:, 0] << 12) + (x[:, 1] << 8) + (x[:, 2] << 4) + x[:, 3]).ravel()
    x_sq = ((x[:3, :3] << 12) + (x[1:, :3] << 8) + (x[:3, 1:] << 4) + x[1:, 1:]).ravel()
    x_middle = ((x[1: 3, 1: 3] << 16) + (x[:2, 1: 3] << 12) + (x[1: 3, :2] << 8) + (x[2:, 1: 3] << 4) + x[1: 3, 2:]
                ).ravel()
    y = np.minimum(x, 13)
    x_vert_6 = (537824 * y[0: 2, 0: 3] + 38416 * y[1: 3, 0: 3] + 2744 * y[2:, 0: 3] + 196 * y[0: 2, 1:] +
                14 * y[1: 3, 1:] + y[2:, 1:]).ravel()
    x_hor_6 = (537824 * y[0: 3, 0: 2] + 38416 * y[0: 3, 1: 3] + 2744 * y[0: 3, 2:] + 196 * y[1:, 0: 2] +
               14 * y[1:, 1: 3] + y[1:, 2:]).ravel()
    return np.concatenate([x_vert, x_hor, x_sq, x_middle, x_vert_6, x_hor_6])


# The RL agent. It is not actually Q, as it tries to learn values of the states (V), rather than actions (Q).
# Not sure what is the correct terminology here, this is definitely a TD(0), basically a modified Q-learning.
# The important details:
# 1) The discount parameter gamma = 1. Don't see why discount rewards in this episodic task.
# 2) Greedy policy, epsilon = 0, no exploration. The game is pretty stochastic as it is, no need.
# 3) The valuation function is basically just a linear operator. I takes a vector of the values of
#    1114112 (=65536 * 17) features and dot-product it by the vector of 1114122 weights.
#    Sounds like a lot of computation but! and this is the beauty - all except 17 of the features
#    are exactly zero, and those 17 are exactly 1. So the whole dot product is just a sum of 17 weights,
#    corresponding to the 1-features.
# 4) The same goes for back-propagation. We only need to update 17 numbers of 1m+ on every step.
# 5) But in fact we update 17 * 8 weights using an obvious D4 symmetry group acting on the board

class QAgent:

    feature_functions = {2: f_2, 3: f_3, 4: f_4, 5: f_5, 6: f_6}
    parameter_shape = {2: (24, 16 ** 2), 3: (52, 16 ** 3), 4: (17, 16 ** 4), 5: (21, 16 ** 5), 6: (33, 0)}

    def __init__(self, name='agent', config_file=None, storage='s3', console='web', log_file=None, n=4, alpha=0.25,
                 decay=0.75, decay_step=10000, low_alpha_limit=0.01, with_weights=True):

        # basic params
        self.name = name
        self.file = name + '.pkl'
        self.game_file = 'best_of_' + self.file
        self.s3 = (storage == 's3')
        self.log_file = log_file
        self.print = print if (console == 'local' or log_file is None) else Logger(log_file=log_file).add

        # params from config file or init/defaults
        if config_file:
            config = load_s3(config_file) or {}
        else:
            config = {}
        self.n = config.get('n', n)
        self.alpha = config.get('alpha', alpha)
        self.decay = config.get('decay', decay)
        self.decay_step = config.get('decay_step', decay_step)
        self.low_alpha_limit = config.get('low_alpha_limit', low_alpha_limit)

        # derived params
        self.num_feat, self.size_feat = QAgent.parameter_shape[self.n]
        self.features = QAgent.feature_functions[self.n]

        # operational params
        self.step = 0
        self.top_game = None
        self.top_score = 0
        self.train_history = []
        self.next_decay = self.decay_step
        self.top_tile = 10

        # The weights can be safely initialized to just zero, but that gives the 0 move (="left")
        # an initial preference. Most probably this is irrelevant, but i wanted an option to avoid it.
        # Besides, this can lead to blow-up, when some weights promptly go to infinity.
        if with_weights:
            self.init_weights()
        else:
            self.weights = None
            self.weight_signature = None

    def __str__(self):
        return f'Agent {self.name}, n={self.n}\ntrained for {self.step} episodes, top score = {self.top_score}'

    def init_weights(self):
        if self.n == 6:
            cutoff_for_6_f = 14  # hard coding this for faster performance of f_6 functions
            self.weights = (np.random.random((17, 16 ** 4)) / 100).tolist() + \
                           (np.random.random((4, 16 ** 5)) / 100).tolist() + \
                           (np.random.random((12, cutoff_for_6_f ** 6)) / 100).tolist()
            self.weight_signature = (17, 4, 12)
        elif self.n == 5:
            self.weights = (np.random.random((17, 16 ** 4)) / 100).tolist() + \
                           (np.random.random((4, 16 ** 5)) / 100).tolist()
            self.weight_signature = (17, 4)
        else:
            self.weights = (np.random.random((self.num_feat, self.size_feat)) / 100).tolist()
            self.weight_signature = (self.num_feat,)

    def list_to_np(self):
        start = 0
        nps = []
        for d in self.weight_signature:
            y = self.weights[start: start + d]
            nps.append(np.array(y, dtype=np.float32))
            start += d
        return nps

    def np_to_list(self):
        real = []
        for weight_component in self.weights:
            real += weight_component.tolist()
        self.weights = real

    def save_agent(self):
        if self.s3:
            nps = self.list_to_np()
            agent_params = QAgent(name=self.name, with_weights=False)
            for key in self.__dict__:
                if key != 'weights':
                    setattr(agent_params, key, getattr(self, key))
            save_s3(agent_params, 'a/' + self.file)
            save_s3(nps, 'weights/' + self.file)
            del nps
        else:
            self.weights = self.list_to_np()
            with open(self.file, 'wb') as f:
                pickle.dump(self, f, -1)
            self.np_to_list()

    def save_game(self, game):
        if self.s3:
            save_s3(game, 'g/' + self.game_file)
        else:
            game.save_game(self.game_file)

    @staticmethod
    def load_agent_local(file):
        with open(file, 'r') as f:
            agent = pickle.load(f)
        agent.np_to_list()
        return agent

    @staticmethod
    def load_agent(file):
        agent = load_s3(file)
        agent.weights = load_s3(f'weights/{file[2:]}')
        agent.np_to_list()
        return agent

    def evaluate(self, row, score=None):
        return sum([self.weights[i][f] for i, f in enumerate(self.features(row))])

    # The numpy library has very nice functions of transpose, rot90, ravel etc.
    # No actual data relocation happens, just the "view" is changed. So it's very fast.
    def update(self, row, dw):
        for _ in range(4):
            for i, f in enumerate(self.features(row)):
                self.weights[i][f] += dw
            row = np.transpose(row)
            for i, f in enumerate(self.features(row)):
                self.weights[i][f] += dw
            row = np.rot90(np.transpose(row))

    # The game 2048 has two kinds of states. After we make a move - this is the one we try to evaluate,
    # and after the random 2-4 tile is placed afterwards.
    # On each step we check which of the available moves leads to a state, which has the highest value
    # according to the current weights of our evaluator. Now we use that best value, our learning rate
    # and the usual Bellman Equation to make a back-propagation update for the previous such state.
    # In this case - we adjust several weights by the same small delta.
    # A very fast and efficient procedure.
    # Then we move in that best direction, add random tile and proceed to the next cycle.
    def episode(self):
        game = Game()
        state, old_label = None, 0

        while not game.game_over(game.row):
            action, best_value = 0, -np.inf
            best_row, best_score = None, None
            for direction in range(4):
                new_row, new_score, change = game.pre_move(game.row, game.score, direction)
                if change:
                    value = self.evaluate(new_row)
                    if value > best_value:
                        action, best_value = direction, value
                        best_row, best_score = new_row, new_score
            if state is not None:
                # reward = self.R(game.row, game.score, best_row, best_score)
                dw = (best_score - game.score + best_value - old_label) * self.alpha / self.num_feat
                self.update(state, dw)
            game.row, game.score = best_row, best_score
            game.odometer += 1
            game.moves.append(action)
            state, old_label = game.row.copy(), best_value
            game.new_tile()
        game.moves.append(-1)
        dw = - old_label * self.alpha / self.num_feat
        self.update(state, dw)

        self.step += 1
        return game

    def _display_lr(self):
        self.print(f'episode = {self.step + 1}, current learning rate = {round(self.alpha, 4)}:')

    def decay_alpha(self):
        self.alpha = round(max(self.alpha * self.decay, self.low_alpha_limit), 4)
        self.next_decay = self.step + self.decay_step
        self.print('------')
        self._display_lr()
        self.print('------')

    # We save the agent every 100 steps, and best game so far - when we beat the previous record.
    # So if you train it and have to make a break at some point - no problem, by loading the agent back
    # you only lose last <100 episodes. Also, after reloading the agent one can adjust the learning rate,
    # decay of this rate etc. Helps with the experimentation.

    def train_run(self, num_eps=100000, add_weights='already', saving=True, stopper=None):
        if add_weights == 'add':
            self.init_weights()
        elif add_weights != 'already':
            self.print('loading weights ...')
            self.weights = load_s3(add_weights)
            self.np_to_list()
        if stopper:
            parent, this_thread = stopper['parent'], stopper['a']
        av1000, ma100 = [], deque(maxlen=100)
        reached = [0] * 7
        best_of_1000 = Game()
        global_start = start = benchmark_time = time.time()
        self.print(f'Agent {self.name} training session started, current step = {self.step}')
        self.print(f'Agent will be saved every 1000 episodes and on STOP command')
        for i in range(self.step + 1, self.step + num_eps + 2):
            if stopper:
                if AGENT_PANE[parent]['id'] != this_thread:
                    break
                benchmark_time = check_thread(parent, benchmark_time)
                if not benchmark_time:
                    return

            # check if it's time to decay learning rate
            if self.step > self.next_decay and self.alpha > self.low_alpha_limit:
                self.decay_alpha()

            game = self.episode()
            ma100.append(game.score)
            av1000.append(game.score)
            if game.score > best_of_1000.score:
                best_of_1000 = game
                if game.score > self.top_score:
                    self.top_game, self.top_score = game, game.score
                    self.print(f'\nnew best game at episode {i}!\n{game.__str__()}\n')
                    if saving:
                        self.save_game(game)
                        self.print(f'game saved at {self.game_file}')
            max_tile = np.max(game.row)
            if max_tile >= 10:
                reached[max_tile - 10] += 1
            # decay learning rate of new maximum tile is achieved
            if max_tile > self.top_tile:
                self.top_tile = max_tile
                self.decay_alpha()

            if i % 100 == 0:
                ma = int(np.mean(ma100))
                self.train_history.append(ma)
                self.print(f'episode {i}: score {game.score} reached {1 << max_tile} ma_100 = {ma}')
            if i % 1000 == 0:
                average = np.mean(av1000)
                self.print('\n------')
                self.print(f'{round((time.time() - start) / 60, 2)} min')
                start = time.time()
                self.print(f'episode = {i}')
                self.print(f'average over last 1000 episodes = {average}')
                av1000 = []
                for j in range(7):
                    r = sum(reached[j:]) / 10
                    if r:
                        self.print(f'{1 << (j + 10)} reached in {r} %')
                reached = [0] * 7
                self.print(f'best of last 1000:')
                self.print(best_of_1000.__str__())
                self.print(f'best of this Agent:')
                self.print(self.top_game.__str__())
                self._display_lr()
                self.print('------\n')
                if saving:
                    self.save_agent()
                    self.print(f'agent saved in {self.file}')
                best_of_1000 = Game()
        total_time = int(time.time() - global_start)
        self.print(f'Total time = {total_time // 60} min {total_time % 60} sec')
        if saving:
            self.save_agent()
            self.print(f'{self.name} saved at step {self.step} in {self.file}\n------------------------\n')

    @staticmethod
    def trial(estimator=None, agent_file=None, limit_tile=0, num=20, game_init=None, depth=0, width=1, since_empty=6,
              storage='s3', console='local', log_file=None, game_file=None, verbose=False, stopper=None):
        display = print if console == 'local' else Logger(log_file=log_file).add
        if stopper:
            parent, this_thread = stopper['parent'], stopper['a']
        if agent_file:
            display(f'Loading Agent from {agent_file} ...')
            agent = QAgent.load_agent(agent_file)
            estimator = agent.evaluate
            display(f'Trial run for {num} games, Agent = {agent.name}\n'
                    f'Looking forward: depth={depth}, width={width}, since_empty={since_empty}')
        start = benchmark_time = time.time()
        results = []
        for i in range(num):
            if stopper:
                if AGENT_PANE[parent]['id'] != this_thread:
                    break
                benchmark_time = check_thread(parent, benchmark_time)
                if not benchmark_time:
                    return

            now = time.time()
            game = Game() if game_init is None else game_init.copy()
            game.trial_run(estimator, limit_tile=limit_tile, depth=depth, width=width, since_empty=since_empty,
                           verbose=verbose)
            display(f'game {i}, result {game.score}, moves {game.odometer}, achieved {1 << np.max(game.row)}, '
                    f'time = {(time.time() - now):.2f}')
            results.append(game)

        if not results:
            return
        average = np.average([v.score for v in results])
        figures = [(1 << np.max(v.row)) for v in results]
        total_odo = sum([v.odometer for v in results])
        results.sort(key=lambda v: v.score, reverse=True)

        def share(limit):
            return len([0 for v in figures if v >= limit]) / len(figures) * 100

        message = '\nBest games:\n'
        for v in results[:3]:
            message += v.__str__() + '\n' + '\n'
        elapsed = time.time() - start
        message += f'average score of {len(results)} runs = {average}\n' + \
                   f'16384 reached in {share(16384)}%\n' + f'8192 reached in {share(8192)}%\n' + \
                   f'4096 reached in {share(4096)}%\n' + f'2048 reached in {share(2048)}%\n' + \
                   f'1024 reached in {share(1024)}%\n' + f'total time = {round(elapsed, 2)}\n' + \
                   f'average time per move = {round(elapsed / total_odo * 1000, 2)} ms\n' + \
                   f'total number of shuffles = {Game.counter}\n' + \
                   f'time per shuffle = {round(elapsed / Game.counter * 1000, 2)} ms'
        display(message)
        if game_file:
            if storage == 's3':
                save_s3(results[0], game_file)
            else:
                results[0].save_game(file=game_file)
            display(f'Best game saved at {game_file}\n------------------------\n')
        return results
