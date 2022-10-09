from game2048.show import *
from config import config

def main():
    # The agent actually plays a game to 2048 in about 1 second. I set the speed of replays at 5 moves/sec,
    # change the speed parameter in ms below if you like

    print('option 0 = play yourself. Not sure why anybody would want it on a PC, but there os an option :)')
    print('option 1 = replay the best game in the best_game.npy file')
    print('option 2 = load the trained agent from best_agent.npy file. Play 100 games, replay the best')
    print('any other input - load the trained agent from best_agent.npy file and see it play online')

    option = int(input())
    viewer = Show()
    if option == 0:
        viewer.play()
    elif option == 1:
        game = Game.load_game(config["SAVE_DIR"] + "best_game.npy")
        viewer.replay(game, speed=25)
    elif option == 2:
        agent = Q_agent.load_agent(config["SAVE_DIR"] + "best_agent.npy")
        est = agent.evaluate
        results = Game.trial(estimator=est, num=100)
        viewer.replay(results[0], speed=200)
    else:
        agent = Q_agent.load_agent(config["SAVE_DIR"] + "best_agent.npy")
        est = agent.evaluate
        viewer.watch(estimator=est, speed=20)

if __name__ == "__main__":
  main()