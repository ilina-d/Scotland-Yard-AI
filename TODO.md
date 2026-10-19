# TODO
- Implement a greedy Detective player that moves towards the closest potential Mr. X position.
- Implement a greedy Mr. X that runs away.
- Implement scripted players that occasionally make bad or random moves for training purposes.
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
LR Scheduler update and small game crash fix.
- `MrXTrainer` and `DetectiveTrainer` changes:
  - Added default arguments.
  - Replaced the scheduler with a custom warmup-plateau-decay function.
  - Adjusted the logs to show the new learning rate updates.
- Fixed `Game` running move logic after a winner has been determined.
- Updated TODO.