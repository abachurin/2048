from game2048.r_learning import *
from game2048.dqn_learning import *
from game2048.show import *
from config import config

def main():

    num_eps = config["EPISODES_TO_TRAIN"] 

    # Run the below line to see the magic. How it starts with random moves and immediately
    # starts climbing the ladder

    agent = Q_agent(n=4, reward=basic_reward, alpha=config["ALPHA"], file="new_agent.npy")

    # Uncomment/comment the above line with the below if you continue training the same agent,
    # update agent.alpha and agent.decay if needed.

    # agent = Q_agent.load_agent(file="best_agent.npy")

    Q_agent.train_run(num_eps, agent=agent, file="new_best_agent.npy", start_ep=0)


if __name__ == "__main__":
  main()