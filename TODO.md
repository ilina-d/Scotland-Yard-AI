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
Optimizations and fixes.
- `Game` changes:
  - Now handles updating Mr. X's potential positions through `Graph`.
  - Code refactoring.
- `Graph` changes:
  - Fixed Mr. X's potential positions being updated when they should not be.
  - Static tensors are now sent to the correct device only once.
  - Optimized building of features (no longer relies on `deepcopy`).
- `BaseDetectiveNet` and `BaseMrXNet` changes:
  - Replaced `get_distribution_policy()` and `get_state_evaluation()` with `get_processed_output()` which
    returns both the action policy and state evaluation.
  - Slightly optimization in building self embeddings and `get_action()`.
- Fixed `DetectiveTrainer` not processing Mr. X's first move when running `per_round` training.
- `Database` now imports Neo4j only if necessary.