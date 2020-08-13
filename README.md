# A Very Simple and Fast Reinforcement Learning Agent for the 2048 Game

## 2048 Game
2048 is a single-player sliding block puzzle game designed by Italian web developer Gabriele Cirulli. The game's objective is to slide numbered tiles on a grid to combine them to create a tile with the number 2048. Of course, one can keep playing and achieve bigger tiles, with theoretical (and probably unachiavable) limit
of 131072 (2 to the power of 17). When I used to play the game as a time-killer and stress-releiver some years ago, the best i've achieved was 8192 tile and the score around 150,000.

For those who never played it but are nevertheless interested, here is a brief description:
2048 is played on a 4×4 grid, with tiles numbered by powers of 2: 2, 4, 8 etc. The board starts with two random 2 or 4 tiles. At each step the Player can try to shake the board in one of the four directions: left, up, right or down. Tiles slide as far as possible in the chosen direction until they are stopped by either another tile or the edge of the grid. If two tiles with the same number collide while moving, they merge into a tile with twice the value. This new number is added to the score. The resulting tile cannot merge with another tile again in the same move. If nothing on the board changed as a result of the Player's action, i.e. the move did not happen, the Player has to choose another move. If there are no valid moves - the game is over. Now, every turn after the Player's move, a new tile randomly appears in an empty spot on the board with a value of either 2 or 4, with 0.9 and 0.1 probabilities respectively.

You can play the game by choosing option 0 while running:
`python3 show.py`

A scoreboard on the upper-right keeps track of the user's score. The user's score starts at zero, and is incremented whenever two tiles combine, by the value of the new tile. As with many arcade games, the user's best score is shown alongside the current score.

The game is won when a tile with a value of 2048 appears on the board, hence the name of the game. After reaching the 2048 tile, players can continue to play (beyond the 2048 tile) to reach higher scores. When the player has no legal moves (there are no empty spaces and no adjacent tiles with the same value), the game ends.

## Reinforcement Learning
Reinforcement learning (RL) is an area of machine learning inspired by behaviourist psychology, concerned with how software agents ought to take actions in an environment so as to maximize some notion of cumulative reward. The problem, due to its generality, is studied in many other disciplines, such as game theory, control theory, operations research, information theory, simulation-based optimization, multi-agent systems, swarm intelligence, statistics and genetic algorithms. In the operations research and control literature, reinforcement learning is called approximate dynamic programming, or neuro-dynamic programming. The problems of interest in reinforcement learning have also been studied in the theory of optimal control, which is concerned mostly with the existence and characterization of optimal solutions, and algorithms for their exact computation, and less with learning or approximation, particularly in the absence of a mathematical model of the environment. In economics and game theory, reinforcement learning may be used to explain how equilibrium may arise under bounded rationality.
