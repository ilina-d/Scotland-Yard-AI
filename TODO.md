# TODO
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
Updates and fixes to game and UI logic.
- `Display` changes:
  - Fixed popup for ticket selection not closing when clicking away.
  - Fixed reachable station indicators not disappearing immediately after a move is made.
  - Slightly decreased the size of the panel title.
- Implemented logic in `Display` and `Game` allowing Mr. X to use double tickets when controlled by a user player.
- `Graph` changes:
  - Fixed placeholder calls to `get_potential_x_pos()` not passing the game state.
  - Fixed incorrect detective positions in `_get_features_for_d()`.
  - Fixed incorrect feature indices for detectives.
- Updated TODO.