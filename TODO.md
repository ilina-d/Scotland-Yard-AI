# TODO
- Visual-related changes:
  - When Mr. X is revealed, place a permanent indicator on his location.
  - If a detective controlled by a user has no valid moves, their turn is skipped without notice.
  - Add game over screen.
  - Change the Mr. X travel log to clearly indicate at which moves he appears.
  - Add control hints (SPACE for un-pausing wait after turn).
- Add player that moves towards Mr. X's last known location.
- Implement GNN Trainers.
- For Mr. X training, try updating weights per-game and per-segment (from given round to next reveal).

---

# Notes & Ideas
- All detective players (5 separate objects) will train one singular GNN model. 
  Giving it the perspective of 5 players that are pretty much identical. It will learn that cornering is also good, 
  not just following. The players all feed the net the state from their pov, their position and the positions of the 
  other 4 detectives, the last known pos of x and all the transports since then i guess.

---

# Latest Changes
Implemented training for the Detectives.
- Added an option to run the `Graph` class without neo4j, in-memory.
- Implemented a new `TrainingDetectiveNetPlayer` to aid in training of the Models wo\ repeating game logic.
- Implemented `DetectiveTrainer`.
- Split `game.play()` logic wo/ visuals into separate helper functions for training convenience.
- Updated TODO.