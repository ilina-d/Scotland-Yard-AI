# TODO
- FIX: Clicking away while a ticket selection popup is open doesn't close the popup.
- FIX: Reachable station indicators don't disappear as soon as the user makes their move.
- FIX: There's currently no way for Mr. X to use a double ticket when controlled by a user.
- Visual-related changes:
  - Update `assets.UserTurn` with appropriate functions for cleaner updates from both `Game` and `Display`.
  - When Mr. X is revealed, place a permanent indicator on his location.
  - If a detective controlled by a user has no valid moves, their turn is skipped without notice.
  - Add game over screen.
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
Started work on GNNs.
- Implemented `BaseDetectiveNet`.
- Started implementation of `BaseMrXNet`.
- Added edge_type attribute to `graph.py`, for working with RGCNs.
- Altered display to make player tokens more noticeable.
- Updated TODO.