* ###### General comments
    * The UI is made with Dash Python and utilises Bootstrap components, so it dynamically survives different screen sizes relatively well. E.g. it even works on my Iphone 13 Pro all right.
    * The board has two panes, left and right (which become top/bottom on small screen). On the left side we can *Train Agent* or *Collect Agent Statistics*. On the right side you can simultaneously *Watch Agent Play*, *Replay Game* or *Play Yourself*. Any new process immediately terminates the previous one in its part of the screen but doesn't affect the other.
* ###### Train Agent
    We can train new Agent, or keep training existing one. The most important parameter is N, it determines which feature function is employed. With higher N - more weights and better results.
* ###### Collect Agent Statistics
    Collect statistics for an existing Agent. The Agent is choosing next move by trying to make a move in all directions and taking the one with the highest internal evaluation. We can try to improve performance by looking `depth` moves ahead and trying `width` random tiles at each after-move stage. And start doing this when number of empty cells goes below `empty`. Needless to say, this slows down the agent significantly.
* ###### Watch Agent Play
    All the same parameters as above, but now we can watch the Agent play one game in real time. Actually, it plays very fast, ~3-4 seconds for a full game reaching 4096. But Dash and your Browser can't render it so fast, besides it won't be fun. So maximum speed is limited.
* ###### Replay Game
    Replays a chosen Game from Storage. Any Agent saves a currently best game during training, also the best game of the latest Statistics Collection is saved.
* ###### Play Yourself
    Of course, it is more convenient to use a standard mobile 2048 App to play the game. I made this option to practice using Keyboard component + writing some custom javascript ( `Game Instructions` window is draggable thanks to that).