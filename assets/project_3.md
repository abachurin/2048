* Model with some combinations of 4 adjacent tiles + several 5-tiles, trained over 100,000 episodes.
/ I wrote in the comments in `rl_learning.py` how i tried all 4-combinations at the start but it didn't work. /
```
average over last 1000 episodes = 54718.676
1024 reached in 96.3 %
2048 reached in 90.4 %
4096 reached in 63.9 %
8192 reached in 1.0 %
best score so far = 168276
2				16				2				4				
64				8				16				2				
16				256				512				128				
4096			8192			2048			4				
```