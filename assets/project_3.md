* Model with some combinations of 4 adjacent tiles + several 5-tiles, trained over 100,000 episodes.
/ I wrote in the comments in `rl_learning.py` how i tried all 4-combinations at the start but it didn't work. /
```
average over last 1000 episodes = 44422.796
1024 reached in 94.4 %
2048 reached in 83.4 %
4096 reached in 45.9 %
8192 reached in 0.4 %
best score = 130664
2				4				256				4
4				16				2048				8192
8				32				256				1024
2				16				128				32
```