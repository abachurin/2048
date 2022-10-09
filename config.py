

config = {}
config["EPISODES_TO_TRAIN"] = 100000
config["ALPHA"] = 0.1
config["USE_DQN"] = False
config["SAVE_DIR"] = "./saved_data/abachurin/"
if config["USE_DQN"] :
  config["SAVE_DIR"] = "./saved_data/farkas93/"