# TODO
- Implement Neo4j Aura support:
  - Check if graph is loaded into database, load it if not.
  - Replace existing `graph.py` logic with neo4j queries.
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
Started implementation of Neo4j for graph operations.
- Implemented `ExplorerPlayer`.
- Began implementing Neo4j Aura support.
  - Added `.env` file for credentials.
  - Updated `Graph` to check credentials and connect to the database on initialization.
- Updated `requirements.txt`.
- Updated TODO.
