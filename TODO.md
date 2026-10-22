# TODO
- Implement scripted players that occasionally make bad or random moves for training purposes.
- Try training `per_game` first and `per_show` after the critic gets good, but give +1 or -1 rewards at the end of the segment.
- Visual-related changes:
  - When Mr. X is revealed, place a permanent indicator on his location.
  - If a detective controlled by a user has no valid moves, their turn is skipped without notice.
  - Add game over screen.
  - Change the Mr. X travel log to clearly indicate at which moves he appears.
  - Add control hints (SPACE for un-pausing wait after turn).
  - Hovering over a player token makes it transparent so you can see the node they're on.

---

# Notes & Ideas
- All detective players (5 separate objects) will train one singular GNN model. 
  Giving it the perspective of 5 players that are pretty much identical. It will learn that cornering is also good, 
  not just following. The players all feed the net the state from their pov, their position and the positions of the 
  other 4 detectives, the last known pos of x and all the transports since then i guess.
- When training per segment, should double moves "prolong the segment" or not? Right now they do, think on this...

---

# Latest Changes
Greedy player updates.
- `GreedyPlayer` now supports both detectives and Mr. X.
- Improved detective logic in `GreedyPlayer`.
- `Graph.get_reachable_neighbors()` now supports `from_node` argument.
- Updated TODO.