# TODO
- Add visuals.
- Add player that moves towards Mr. X's last known location.
- Implement GNN Players and Trainers.

---

# Notes & Ideas
- All detective players (5 separate objects) will train one singular GNN model. 
  Giving it the perspective of 5 players that are pretty much identical. It will learn that cornering is also good, 
  not just following. The players all feed the net the state from their pov, their position and the positions of the 
  other 4 detectives, the last known pos of x and all the transports since then i guess.

---

# Latest Changes
Fully implemented game and database logic /wo visuals.
- State now handles double ticket rules and round/step count instead of Game.
- Implemented `Database` class that supports both Neo4j Aura and local instance access.
- `Graph` now relies on `Database` and handles GNN input preparation.
- Replaced the graphs folder with a single graph_data file located in helpers.
- Updated Players to support the `State` changes.
- Updated `requirements.txt`.
- Updated `TODO.md`.