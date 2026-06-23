# TODO
- Add player that prefers unexplored nodes.
- Add player that moves towards Mr. X's last known location.
- Add visuals.

---

# Notes & Ideas
- All detective players (5 separate objects) will train one singular GNN model. 
  Giving it the perspective of 5 players that are pretty much identical. It will learn that cornering is also good, 
  not just following. The players all feed the net the state from their pov, their position and the positions of the 
  other 4 detectives, the last known pos of x and all the transports since then i guess.

---

# Latest Changes
Implemented game logic.
- Fully implemented `Game`.
- Fully implemented `State`.
- Fully implemented `BasePlayer` and `RandomPlayer`.
- Updated TODO.