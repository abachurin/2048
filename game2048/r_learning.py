from game2048.game_logic import *


def basic_reward(game, action):
    next_game = game.copy()
    next_game.move(action)
    return next_game.score - game.score


# Intuitively (at least my initial intuition said so :) log-score should work better than the score itself.
# And indeed it starts learning much faster compared to the basic reward. But then it slows down significantly.
# I am not sure how to explain it, may be it's just an issue of learning rate tuning ..

def log_reward(game, action):
    next_game = game.copy()
    next_game.move(action)
    return np.log(next_game.score + 1) - np.log(game.score + 1)


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

    save_file = "agent.npy"     # saves the weights, training step, current alpha and type of features
    feature_functions = {2: f_2, 3: f_3, 4: f_4}
    parameter_shape = {2: (24, 256), 3: (52, 4096), 4: (17, 65536)}

    def __init__(self, weights=None, reward=basic_reward, step=0, alpha=0.2, decay=0.999,
                 file=None, n=4):
        self.R = reward
        self.step = step
        self.alpha = alpha
        self.decay = decay
        self.file = file or Q_agent.save_file
        self.n = n
        self.num_feat, self.size_feat = Q_agent.parameter_shape[n]

        # The weights can be safely initialized to just zero, but that gives the 0 move (="left")
        # an initial preference. Most probably this is irrelevant, but i wanted to avoid it.

        if weights is None:
            self.weights = weights or np.random.random((self.num_feat, self.size_feat)) / 100
        else:
            self.weights = weights

    # a numpy.save method works fine not only for numpy arrays but also for ordinary lists
    def save_agent(self, file=None):
        file = file or self.file
        arr = np.array([self.weights, self.step, self.alpha, self.n])
        np.save(file, arr)
        pass

    @staticmethod
    def load_agent(file=save_file):
        arr = np.load(file, allow_pickle=True)
        agent = Q_agent(weights=arr[0], step=arr[1], alpha=arr[2], n=arr[3])
        return agent

    def features(self, X):
        return Q_agent.feature_functions[self.n](X)

    # numpy arrays have a nice "advanced slicing" trick, used in this function
    def evaluate(self, state):
        features = self.features(state.row)
        return np.sum(self.weights[range(self.num_feat), features])

    def update(self, state, dw):
        self.step += 1
        if self.step % 200000 == 0 and self.alpha > 0.02:
            self.alpha *= self.decay
            print('------')
            print(f'step = {self.step}, learning rate = {self.alpha}')
            print('------')

        # Didn't use advanced slicing here because i experimented with the idea of updating
        # a feature only once, even if it happens several times in D4 images of the board
        # Not in the final version, but maybe makes sense, left it as it is.

        def _upd(X):
            features = self.features(X)
            for i, f in enumerate(features):
                self.weights[i, f] += dw

        # The numpy library has very nice functions of transpose, rot90, ravel etc.
        # No actual number relocation happens, just the "view" is changed. So it's very fast.

        X = state.row
        for _ in range(4):
            _upd(X)
            X = np.transpose(X)
            _upd(X)
            X = np.rot90(np.transpose(X))

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
        while not game.game_over():
            action, best_value = 0, -np.inf
            for direction in range(4):
                test = game.copy()
                change = test.move(direction)
                if change:
                    value = self.evaluate(test)
                    if value > best_value:
                        action, best_value = direction, value
            if state:
                reward = self.R(game, action)
                dw = self.alpha * (reward + best_value - old_label) / self.num_feat
                self.update(state, dw)
            game.move(action)
            state, old_label = game.copy(), best_value
            game.new_tile()
        dw = - self.alpha * old_label / self.num_feat
        self.update(state, dw)
        game.history.append(game)
        return game

    # We save the agent every 100 steps, and best game so far - when we beat the previous record.
    # So if you train it and have to make a break at some point - no problem, by loading the agent back
    # you only lose last <100 episodes. Also, after reloading the agent one can adjust the learning rate,
    # decay of this rate etc. Helps with the experimentation.

    @staticmethod
    def train_run(num_eps, agent=None, file=None, start_ep=0, saving=True):
        if agent is None:
            agent = Q_agent()
        if file:
            agent.file = file
        av1000 = []
        ma100 = []
        reached = [0] * 7
        best_game, best_score = None, 0
        start = time.time()
        for i in range(start_ep + 1, num_eps + 1):
            game = agent.episode()
            ma100.append(game.score)
            av1000.append(game.score)
            if game.score > best_score:
                best_game, best_score = game, game.score
                print('new best game!')
                print(game)
                if saving:
                    game.save_game(file='best_game.npy')
                    print('game saved at best_game.npy')
            max_tile = np.max(game.row)
            if max_tile >= 10:
                reached[max_tile - 10] += 1
            if i - start_ep > 100:
                ma100 = ma100[1:]
            print(i, game.odometer, game.score, 'reached', 1 << np.max(game.row), '100-ma=', int(np.mean(ma100)))
            if saving and i % 100 == 0:
                agent.save_agent()
                print(f'agent saved in {agent.file}')
            if i % 1000 == 0:
                print('------')
                print((time.time() - start) / 60, "min")
                start = time.time()
                print(f'episode = {i}')
                print(f'average over last 1000 episodes = {np.mean(av1000)}')
                av1000 = []
                for j in range(7):
                    r = sum(reached[j:]) / 10
                    print(f'{1 << (j + 10)} reached in {r} %')
                reached = [0] * 7
                print(f'best score so far = {best_score}')
                print(best_game)
                print(f'current learning rate = {agent.alpha}')
                print('------')


if __name__ == "__main__":

    num_eps = 100000

    # Run the below line to see the magic. How it starts with random moves and immediately
    # starts climbing the ladder

    agent = Q_agent(n=4, reward=basic_reward, alpha=0.2, file="new_agent.npy")

    # Uncomment/comment the above line with the below if you continue training the same agent,
    # update agent.alpha and agent.decay if needed.

    # agent = Q_agent.load_agent(file="best_agent.npy")

    Q_agent.train_run(num_eps, agent=agent, file="new_best_agent.npy", start_ep=0)
