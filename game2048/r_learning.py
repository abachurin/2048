from .game_logic import *


def basic_reward(row, score, new_row, new_score):
    return new_score - score


# Intuitively (at least my initial intuition said so :) log-score should work better than the score itself.
# And indeed it starts learning much faster compared to the basic reward. But then it slows down significantly.
# I am not sure how to explain it, may be it's just an issue of learning rate tuning ..

def log_reward(row, score, new_row, new_score):
    return np.log(new_score + 1) - np.log(score + 1)


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
    y = np.minimum(x, 11)
    x_vert_6 = (248832 * y[0: 2, 0: 3] + 20736 * y[1: 3, 0: 3] + 1728 * y[2:, 0: 3] + 144 * y[0: 2, 1:] +
                12 * y[1: 3, 1:] + y[2:, 1:]).ravel()
    x_hor_6 = (248832 * y[0: 3, 0: 2] + 20736 * y[0: 3, 1: 3] + 1728 * y[0: 3, 2:] + 144 * y[1:, 0: 2] +
               12 * y[1:, 1: 3] + y[1:, 2:]).ravel()
    return np.concatenate([x_vert, x_hor, x_sq, x_middle, x_vert_6, x_hor_6])


def max_tile_in_feature(n):
    result = {}
    for i in range(1 << (4 * n)):
        start, top = i, 0
        while start:
            top = max(top, start % 16)
            start //= 16
        result[i] = top
    return result


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

class Q_agent:

    feature_functions = {2: f_2, 3: f_3, 4: f_4, 5: f_5, 6: f_6}
    parameter_shape = {2: (24, 16 ** 2), 3: (52, 16 ** 3), 4: (17, 16 ** 4), 5: (21, 16 ** 5), 6: (33, 0)}

    def __init__(self, name='agent', config_file=None, storage='s3', console='local', log_file=None, reward='basic',
                 decay_model='simple', n=4, alpha=0.25, decay=0.75, decay_step=10000, low_alpha_limit=0.01):

        # basic params
        self.name = name
        self.file = name + '.pkl'
        self.game_file = 'best_of_' + self.file
        self.save_agent = self.save_agent_s3 if storage == 's3' else self.save_agent_local
        self.save_game = self.save_game_s3 if storage == 's3' else self.save_game_local
        self.print = print if (console == 'local' or log_file is None) else Logger(log_file=log_file).add

        # params from config file or init/defaults
        if config_file:
            config = load_s3(config_file) or {}
        else:
            config = {}
        self.reward = config.get('reward', reward)
        self.decay_model = config.get('decay_model', decay_model)
        self.n = config.get('n', n)
        self.alpha = config.get('alpha', alpha)
        self.decay = config.get('decay', decay)
        self.decay_step = config.get('decay_step', decay_step)
        self.low_alpha_limit = config.get('low_alpha_limit', low_alpha_limit)

        # derived params
        self.R = basic_reward if reward == 'basic' else log_reward
        self._upd = self._upd_simple if decay_model == 'simple' else self._upd_scaled
        self.num_feat, self.size_feat = Q_agent.parameter_shape[self.n]
        self.features = Q_agent.feature_functions[self.n]
        if self.decay_model == 'scaled':
            self.max_in_f = max_tile_in_feature(self.n)
            self.lr = {v: self.alpha for v in range(16)}
            self.lr_from_f = {i: self.lr[self.max_in_f[i]] for i in range(self.size_feat)}

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
        if self.n == 6:
            self.cutoff_for_6_f = 12       # hard coding this for faster performance of f_6 functions
            self.weights = (np.random.random((17, 16 ** 4)) / 100).tolist() + \
                           (np.random.random((4, 16 ** 5)) / 100).tolist() + \
                           (np.random.random((12, self.cutoff_for_6_f ** 6)) / 100).tolist()
            self.weight_signature = (17, 4, 12)
        elif self.n == 5:
            self.weights = (np.random.random((17, 16 ** 4)) / 100).tolist() + \
                           (np.random.random((4, 16 ** 5)) / 100).tolist()
            self.weight_signature = (17, 4)
        else:
            self.weights = (np.random.random((self.num_feat, self.size_feat)) / 100).tolist()
            self.weight_signature = (self.num_feat, )

    def __str__(self):
        return f'Agent {self.name}, n={self.n}, reward={self.reward}, decay_model={self.decay_model}\n' \
               f'trained for {self.step} episodes, top score = {self.top_score}'

    def list_to_np(self):
        start = 0
        nps = []
        for d in self.weight_signature:
            y = self.weights[start: start + d]
            nps.append(np.array(y, dtype=np.float32))
            start += d
        self.weights = nps

    def np_to_list(self):
        real = []
        for weight_component in self.weights:
            real += weight_component.tolist()
        self.weights = real

    def save_agent_local(self):
        self.list_to_np()
        with open(self.file, 'wb') as f:
            pickle.dump(self, f, -1)
        self.np_to_list()

    def save_agent_s3(self):
        self.list_to_np()
        save_s3(self, 'a/' + self.file)
        self.np_to_list()

    def save_game_local(self, game):
        game.save_game(self.game_file)

    def save_game_s3(self, game):
        save_s3(game, 'g/' + self.game_file)

    @staticmethod
    def load_agent(file):
        with open(file, 'rb') as f:
            agent = pickle.load(f)
        agent.np_to_list()
        return agent

    # numpy arrays have a nice "advanced slicing" trick, used in this function
    def evaluate(self, row, score=None):
        return sum([self.weights[i][f] for i, f in enumerate(self.features(row))])

    def _upd_scaled(self, x, dw):
        for i, f in enumerate(self.features(x)):
            self.weights[i][f] += dw * self.lr_from_f[f]

    def _upd_simple(self, x, dw):
        dw *= self.alpha
        for i, f in enumerate(self.features(x)):
            self.weights[i][f] += dw

    # The numpy library has very nice functions of transpose, rot90, ravel etc.
    # No actual number relocation happens, just the "view" is changed. So it's very fast.
    def update(self, row, dw):
        for _ in range(4):
            self._upd(row, dw)
            row = np.transpose(row)
            self._upd(row, dw)
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
                reward = self.R(game.row, game.score, best_row, best_score)
                dw = (reward + best_value - old_label) / self.num_feat
                self.update(state, dw)
            game.row, game.score = best_row, best_score
            game.odometer += 1
            game.moves.append(action)
            state, old_label = game.row.copy(), best_value
            game.new_tile()
        game.moves.append(-1)
        dw = - old_label / self.num_feat
        self.update(state, dw)

        self.step += 1
        return game

    def _display_lr(self):
        if self.decay_model == 'scaled':
            self.print(f'episode = {self.step + 1}, current learning rate as function of max tile in the element:')
            self.print({1 << v if v else 0: round(self.lr[v], 4) for v in self.lr if v >= 9})
            self.print(f'next learning rate decay scheduled at step {self.next_decay + 1}')
        elif self.decay_model == 'simple':
            self.print(f'episode = {self.step + 1}, current learning rate = {round(self.alpha, 4)}:')

    def decay_alpha(self):
        if self.decay_model == 'scaled':
            for i in range(16):
                if i < self.top_tile - 1:
                    self.lr[i] = max(self.lr[i] * self.decay, self.low_alpha_limit)
            self.lr_from_f = {i: self.lr[self.max_in_f[i]] for i in range(self.size_feat)}
        self.alpha = round(max(self.alpha * self.decay, self.low_alpha_limit), 4)
        self.next_decay = self.step + self.decay_step
        self.print('------')
        self._display_lr()
        self.print('------')

    # We save the agent every 100 steps, and best game so far - when we beat the previous record.
    # So if you train it and have to make a break at some point - no problem, by loading the agent back
    # you only lose last <100 episodes. Also, after reloading the agent one can adjust the learning rate,
    # decay of this rate etc. Helps with the experimentation.

    def train_run(self, num_eps=100000, saving=True):
        av1000, ma100 = [], deque(maxlen=100)
        reached = [0] * 7
        save_steps = 250 if self.n <=5 else 500
        global_start = start = time.time()
        self.print(f'Agent {self.name} training session started, current step = {self.step}')
        self.print(f'Agent will be saved every {save_steps} episodes')
        for i in range(self.step + 1, self.step + num_eps + 2):

            # check if it's time to decay learning rate
            if self.step > self.next_decay and self.alpha > self.low_alpha_limit:
                self.decay_alpha()

            game = self.episode()
            ma100.append(game.score)
            av1000.append(game.score)
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
            if i % save_steps == 0:
                t = time.time()
                self.save_agent()
                self.print(f'agent saved in {self.file}')
            if i % 1000 == 0:
                average = np.mean(av1000)
                self.print('\n------')
                self.print(f'{(time.time() - start) / 60} min')
                start = time.time()
                self.print(f'episode = {i}')
                self.print(f'average over last 1000 episodes = {average}')
                av1000 = []
                for j in range(7):
                    r = sum(reached[j:]) / 10
                    self.print(f'{1 << (j + 10)} reached in {r} %')
                reached = [0] * 7
                self.print(f'best score so far = {self.top_score}')
                self.print(self.top_game.__str__())
                self._display_lr()
                self.print('------\n')
                if saving:
                    self.save_agent()
                    self.print(f'agent saved in {self.file}')
        self.print(f'Total time for {num_eps} = {time.time() - global_start}')
        if saving:
            self.save_agent()
            self.print(f'agent saved in {self.file}')
