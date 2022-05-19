from .game_logic import *


def basic_reward(row, score, new_row, new_score):
    return new_score - score


# Intuitively (at least my initial intuition said so :) log-score should work better than the score itself.
# And indeed it starts learning much faster compared to the basic reward. But then it slows down significantly.
# I am not sure how to explain it, may be it's just an issue of learning rate tuning ..

def log_reward(row, score, new_row, new_score):
    return np.log(new_score + 1) - np.log(score + 1)


# features = all adjacent pairs

def f_2(X):
    X_vert = (16 * X[:3, :] + X[1:, :]).ravel()
    X_hor = (16 * X[:, :3] + X[:, 1:]).ravel()
    return np.concatenate([X_vert, X_hor])


# features = all adjacent triples, i.e. 3 in a row + 3 in a any square missing one corner

def f_3(X):
    X_vert = (256 * X[:2, :] + 16 * X[1:3, :] + X[2:, :]).ravel()
    X_hor = (256 * X[:, :2] + 16 * X[:, 1:3] + X[:, 2:]).ravel()
    X_ex_00 = (256 * X[1:, :3] + 16 * X[1:, 1:] + X[:3, 1:]).ravel()
    X_ex_01 = (256 * X[:3, :3] + 16 * X[1:, :3] + X[1:, 1:]).ravel()
    X_ex_10 = (256 * X[:3, :3] + 16 * X[:3, 1:] + X[1:, 1:]).ravel()
    X_ex_11 = (256 * X[:3, :3] + 16 * X[1:, :3] + X[:3, 1:]).ravel()
    return np.concatenate([X_vert, X_hor, X_ex_00, X_ex_01, X_ex_10, X_ex_11])


# Initially i also made all adjacent quartets of different shape, but the learning was not happening.
# My theory is that: 1) we want our features to intersect and correlate (otherwise we will only learn
# some several separate pieces of the board, and that obviously can not lead to anything.
# but 2) we don't want them to intersect too much (like 3 cells common to two quartets), as they start
# to kinda suppress and contradict each other.
# So i left just columns, rows and squares. 17 features all in all. And it works just fine.

def f_4(X):
    X_vert = (4096 * X[0, :] + 256 * X[1, :] + 16 * X[2, :] + X[3, :]).ravel()
    X_hor = (4096 * X[:, 0] + 256 * X[:, 1] + 16 * X[:, 2] + X[:, 3]).ravel()
    X_sq = (4096 * X[:3, :3] + 256 * X[1:, :3] + 16 * X[:3, 1:] + X[1:, 1:]).ravel()
    return np.concatenate([X_vert, X_hor, X_sq])


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

    feature_functions = {2: f_2, 3: f_3, 4: f_4}
    parameter_shape = {2: (24, 256), 3: (52, 4096), 4: (17, 65536)}

    def __init__(self, name='agent', config_file=None, storage='s3', console='local', weights_type='random',
                 reward='basic', decay_model='simple', n=4, alpha=0.25, decay=0.75,
                 decay_step=10000, low_alpha_limit=0.01):
        self.name = name
        self.file = name + '.pkl'
        self.game_file = 'best_of_' + self.file
        self.save_agent = self.save_agent_s3 if storage == 's3' else self.save_agent_local
        self.save_game = self.save_game_s3 if storage == 's3' else self.save_game_local
        self.print = LOGS.add if console == 'web' else print

        if config_file:
            config = load_s3(config_file) or {}
        else:
            config = {}
        self.weights_type = config.get('weights', weights_type)
        self.reward = config.get('reward', reward)
        self.decay_model = config.get('decay_model', decay_model)
        self.n = config.get('n', n)
        self.alpha = config.get('alpha', alpha)
        self.decay = config.get('decay', decay)
        self.decay_step = config.get('decay_step', decay_step)
        self.low_alpha_limit = config.get('low_alpha_limit', low_alpha_limit)

        self.R = basic_reward if reward == 'basic' else log_reward
        self.decay_model = decay_model
        self._upd = self._upd_simple if decay_model == 'simple' else self._upd_scaled
        self.logs = ''
        self.step = 0
        self.top_game = None
        self.top_score = 0
        self.train_history = []
        self.num_feat, self.size_feat = Q_agent.parameter_shape[n]
        self.features = Q_agent.feature_functions[self.n]
        self.top_tile = 10
        self.max_in_f = max_tile_in_feature(n)
        self.lr = {v: self.alpha for v in range(16)}
        self.lr_from_f = {i: self.lr[self.max_in_f[i]] for i in range(self.size_feat)}
        self.next_decay = self.decay_step

        # The weights can be safely initialized to just zero, but that gives the 0 move (="left")
        # an initial preference. Most probably this is irrelevant, but i wanted an option to avoid it.
        if self.weights_type == 'random':
            self.weights = (np.random.random((self.num_feat, self.size_feat)) / 100).tolist()
        else:
            self.weights = [[0] * self.size_feat] * self.num_feat

    def save_agent_local(self):
        with open(self.file, 'wb') as f:
            pickle.dump(self, f, -1)

    def save_agent_s3(self):
        save_s3(self, 'a/' + self.file)

    def save_game_local(self, game):
        game.save_game(self.game_file)

    def save_game_s3(self, game):
        save_s3(game, 'g/' + self.game_file)

    @staticmethod
    def load_agent(file):
        with open(file, 'rb') as f:
            agent = pickle.load(f)
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
            self.print(f'episode = {self.step + 1}, current learning rate = {self.alpha}:')

    def decay_alpha(self):
        for i in range(16):
            if i < self.top_tile - 1:
                self.lr[i] = max(self.lr[i] * self.decay, self.low_alpha_limit)
        self.lr_from_f = {i: self.lr[self.max_in_f[i]] for i in range(self.size_feat)}
        self.alpha = max(self.alpha * self.decay, self.low_alpha_limit)
        self.next_decay = self.step + self.decay_step
        self.print('------')
        self._display_lr()
        self.print('------')

    # We save the agent every 100 steps, and best game so far - when we beat the previous record.
    # So if you train it and have to make a break at some point - no problem, by loading the agent back
    # you only lose last <100 episodes. Also, after reloading the agent one can adjust the learning rate,
    # decay of this rate etc. Helps with the experimentation.

    def train_run(self, num_eps=100000, saving=True, chart=False):
        av1000, ma100 = [], deque(maxlen=100)
        reached = [0] * 7
        global_start = start = time.time()
        self.print(f'Agent {self.name} training session started')
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
                self.save_agent()
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
        if chart:
            self.chart_ma_100(self.train_history)

    @staticmethod
    def chart_ma_100(ma100):
        plt.figure(figsize=(8, 6))
        plt.plot(ma100)
        plt.show()


if __name__ == "__main__":

    # Run the below line to see the magic. How it starts with random moves and immediately
    # starts climbing the ladder
    a = load_s3('a/Loki.pkl')
    print(a.alpha, a.decay, a.step)
    sys.exit()
    a_4 = Q_agent(name='agent')
    a_4.train_run(chart=True)
