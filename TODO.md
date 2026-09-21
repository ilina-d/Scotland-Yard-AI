# TODO
- FIX: Clicking away while a ticket selection popup is open doesn't close the popup.
- FIX: Reachable station indicators don't disappear as soon as the user makes their move.
- FIX: There's currently no way for Mr. X to use a double ticket when controlled by a user.
- Visual-related changes:
  - Update `assets.UserTurn` with appropriate functions for cleaner updates from both `Game` and `Display`.
  - When Mr. X is revealed, place a permanent indicator on his location.
  - Add white outline on player tokens.
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
Implemented game visuals.
- Implemented `utils/display`:
  - Added `assets.py` for variables related to visuals and a helper `UserTurn` class for handling user input.
  - Added `Camera` for moving around and zooming in/out of the game board.
  - Added `Display` for all visuals logic.
- Updated `Game` to support the new visuals.
- Added `always_show_x` argument to `Game` for debugging.
- If any of the players are controlled by a user, the game must be run with `visuals` on.
- `Game` and `Display` now handle `UserPlayer` logic for making moves.
- `State` now tracks the `current_player` and `winner`.
- Updated TODO.