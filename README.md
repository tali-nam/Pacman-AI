This is a two-part project developed for Duke's CS370 class, Introduction to AI. Relevant code I developed can be found in search.py & multiagents.py

Part 1: Search Algorithms
This project explores various graph search algorithms applied to a Pac-Man game environment:

Depth-First Search (DFS): Explores paths deeply before backtracking, using a stack to manage the search fringe.
Breadth-First Search (BFS): Explores paths level by level using a queue to ensure the shortest path is found.
Uniform Cost Search (UCS): Uses a priority queue to expand the least-cost path first, ensuring an optimal solution.
A Search:* Enhances UCS with a heuristic (Manhattan distance) to efficiently find the optimal path.
Additionally, custom search problems and heuristics were designed for Pac-Man:

CornersProblem: Requires Pac-Man to reach all designated corner dots, using a heuristic to guide the search.
FoodSearchProblem: Involves finding the optimal path for Pac-Man to consume all food pellets on the board.

Part 2: Multi-Agent Strategies
This section implements AI strategies for Pac-Man against adversarial ghosts:

ReflexAgent: A simple agent that evaluates actions based on food proximity and ghost positions, using reciprocal distances as features.
MinimaxAgent: Implements minimax decision-making with alternating layers for Pac-Man (max) and ghosts (min), sometimes choosing self-sacrifice if it leads to a better long-term outcome.
AlphaBetaAgent: Optimizes minimax search with alpha-beta pruning, reducing unnecessary computations.
ExpectimaxAgent: Uses expected values instead of worst-case assumptions, allowing Pac-Man to act probabilistically against ghosts.
This project applies fundamental AI search and decision-making techniques to enhance Pac-Man’s gameplay strategy.







