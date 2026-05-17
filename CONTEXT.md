# CONTEXT.md — Course Algorithms Reference
# Fundamentals of AI — Connect-4 Agent Project

> **HOW TO USE THIS FILE:** Before implementing any agent, read this entire file.
> At least 60% of your agent's core logic must be grounded in algorithms described here.
> External techniques are allowed but cannot exceed 40% of the design.

---

## Connect-4 Problem Definition (Read First)

- **Board:** 6 rows × 7 columns, gravity applies (pieces fall to lowest empty row)
- **State:** The full board configuration + whose turn it is
- **Actions:** 7 possible (one per column), only valid if column is not full
- **Transition:** Deterministic — P(s'|s,a) = 1 for all legal moves
- **Terminal states:** 4-in-a-row (horizontal, vertical, diagonal) → win/loss; full board → draw
- **Reward:** Win = +1, Loss = -1, Draw = 0 (only at terminal states, γ = 1)
- **Players:** 2, alternating turns (Red and Yellow)
- **Key constraint:** Because Connect-4 is deterministic, transition probabilities are trivial
  but the opponent's behavior creates the uncertainty.

---

## Module 1: Environments, Agents, and Rationality

### Environment and Agent Framework
- **What it does:** Defines the vocabulary for all AI problems. An environment has states,
  events, and transition rules. An agent perceives the state and selects actions. Rationality
  means maximizing expected performance.
- **Formal loop:**
  1. Environment initializes at state s₀
  2. Agent receives percept st (current state)
  3. Agent computes action at = f(st) using its agent function f: P → A
  4. Environment transitions: st+1 = ψ(st, at)
- **Key formulas:**
  - ψ: S × E → S  (state transition function)
  - f: P → A      (agent function, maps percepts to actions)
- **Environment types (PEAS analysis):**
  - Fully observable vs partially observable
  - Deterministic vs stochastic
  - Static vs dynamic
  - Discrete vs continuous
  - Single-agent vs multi-agent
- **Notes for Connect-4:** Connect-4 is fully observable, deterministic, static, discrete,
  sequential, and competitive multi-agent. The agent function maps board states to column choices.

---

## Module 2: Agents and Generic Decision Optimization under Certainty

### General Constructive Search (GCS)
- **What it does:** Systematically builds solutions step-by-step by expanding partial plans.
  The order of expansion determines the search strategy (BFS, DFS, Best-First, etc.)
- **Pseudocode:**
  ```
  OPEN = [empty_plan]
  best = None
  while OPEN is not empty and budget allows:
      W = remove_first(OPEN)
      for o in expand(W):
          W' = W + [o]
          if is_goal(W'):
              if better(W', best): best = W'
          else:
              OPEN.append(W')
  return best
  ```
- **Key insight:** The selection strategy for OPEN determines everything:
  - FIFO → BFS (completeness guaranteed)
  - LIFO → DFS (memory efficient)
  - Priority queue by score → Best-First / Greedy
- **Key formula:** argmax_{x ∈ X} g(x) — find the plan maximizing the score function
- **Notes for Connect-4:** Each move sequence is a partial construction plan. The board
  state after k moves is the partial plan of length k. Used as the backbone for game tree search.

---

## Module 3: Fixed Size Decision Problems under Certainty

### Constraint Networks
- **What it does:** Prunes the search space by detecting dead-ends early using constraints
  between variables. Arc consistency and forward checking eliminate impossible assignments.
- **Notes for Connect-4:** Can detect impossible board configurations early. Less directly
  applicable but useful for pruning illegal moves.

### Genetic Algorithms (Transformative Search)
- **What it does:** Evolves a population of solutions using selection, crossover, and mutation.
  Combines existing good solutions instead of building from scratch.
- **Pseudocode:**
  ```
  population = initialize_random(N)
  while not converged:
      parents = select_by_fitness(population)
      offspring = []
      for p1, p2 in pairs(parents):
          child = crossover(p1, p2)   # combine two solutions
          child = mutate(child)        # small random change
          offspring.append(child)
      population = offspring
  return best_in(population)
  ```
- **Crossover types:**
  - 1-Point: split at one position, swap tails
  - 2-Point: swap middle segment between two positions
  - Uniform: bit-mask b ∈ {0,1}ⁿ, p' = (p ∧ b) ∨ (q ∧ ¬b)
- **Notes for Connect-4:** Genetic algorithms can be used to evolve heuristic weight vectors
  for board evaluation functions (e.g., weights for "threats", "center control", "open lines").
  Each chromosome encodes a set of weights; fitness = win rate against a reference opponent.

---

## Module 4: Sequential Decision Problems under Certainty

### Graph Search & Uniform Cost Search (UCS)
- **What it does:** Treats states as nodes and transitions as edges. UCS finds optimal paths
  by always expanding the lowest cumulative-cost path first.
- **Pseudocode (UCS):**
  ```
  OPEN = priority_queue([(start, cost=0)])
  CLOSED = set()
  while OPEN:
      p = OPEN.pop_min_cost()
      if is_goal(p.head): return p
      if p.head in CLOSED: continue
      CLOSED.add(p.head)
      for successor s' of p.head:
          new_cost = p.cost + c(p.head, s')
          if s' not in OPEN or new_cost < OPEN[s'].cost:
              OPEN.push((p + s', cost=new_cost))
  ```
- **Key formula:** p.g = p'.g + c(p'.head, p.head)  (additive cumulative cost)
- **Notes for Connect-4:** Connect-4 is a tree (no cycles since pieces only stack up),
  so CLOSED list is less critical. However, **transposition tables** (caching already-evaluated
  board states reached by different move sequences) are the direct equivalent and a crucial
  optimization. A win is a win regardless of move order → path-independent quality.

---

## Module 5: Atomic Decisions under Risk

### Expected Utility (Bernoulli Principle)
- **What it does:** For one-shot decisions with probabilistic outcomes, the rational agent
  picks the action maximizing expected utility, not expected raw outcome. A utility function
  U encodes preferences (e.g., risk aversion).
- **Pseudocode:**
  ```
  for each action a:
      q(a) = sum over s': U(s') * P(s'|s0, a)
  return argmax_a q(a)
  ```
- **Key formulas:**
  - Expected Utility: q(a) = E[U(S')|s₀, a] = Σ_{s'∈S} U(s') · P(s'|s₀, a)
  - Rational decision: a* = argmax_{a∈A} q(a)
- **Notes for Connect-4:** Against a probabilistic opponent (e.g., random player), our agent's
  move leads to a distribution of opponent responses. The utility function is:
  U(win)=1, U(loss)=-1, U(draw)=0. We pick the move maximizing expected utility.

---

## Module 6: Sequential Decisions under Risk — MDPs

### Markov Decision Processes (MDP)
- **What it does:** Formalizes sequential decisions under uncertainty. Instead of a fixed path,
  the solution is a **policy** π (a mapping from states to actions) maximizing expected total
  discounted reward.
- **Formal definition:** An MDP is a triple (S, A, P) where:
  - S: discrete set of agent internal states
  - A := ∪_{s∈S} Aₛ, where Aₛ is the discrete set of actions in state s
  - P(s', r | s, a): transition distribution (joint distribution over next state and reward)
  - Simplified view (used in the course): rewards depend only on state r: S → R,
    so transitions simplify to P(s'|s,a)
- **Loop semantics:**
  1. Agent observes state st
  2. Agent picks action at ∈ A_{st}
  3. MDP samples (st+1, rt+1) ~ P(·|st, at)
- **Key formulas:**
  - Trial utility: U(τ) = Σ_{t=0}^∞ γᵗ r(sₜ)
  - Optimal policy: π* = argmax_π E^{s₀,π}[Σ_{t=0}^∞ γᵗ r(Sₜ)]
  - γ ∈ (0,1]: discount factor (γ=1 means only final result counts)
- **Notes for Connect-4:** We use γ=1 (only win/loss matters, not how fast).
  The opponent's strategy induces the transition distribution P(s'|s,a).
  Since Connect-4 is deterministic, given our move and the opponent's response,
  P(s'|s,a) = 1 for that specific s'. Uncertainty comes from not knowing which
  column the opponent will choose.

---

## Module 7: Policy Evaluation

### State Values and the Bellman Equation
- **What it does:** Assigns a numerical value to each state under a given policy π.
  The value represents the expected total discounted reward starting from that state
  following π forever.
- **Key formula — Bellman Equation:**
  ```
  v^π(s) = r(s) + γ · Σ_{s'∈S} P(s'|s, π(s)) · v^π(s')
  ```
  - Terminal states: v^π(s⊥) = 0, so v^π(s) = r(s) for terminal states

### Exact Policy Evaluation
- **What it does:** Solves the system of |S| linear equations (one per state) exactly.
- **Complexity:** O(|S|³) — infeasible for Connect-4's enormous state space (~4.5 trillion states)

### Iterative (Incremental) Policy Evaluation
- **What it does:** Iteratively applies the Bellman update rule until values converge.
- **Pseudocode:**
  ```
  initialize v_hat(s) for all s (e.g., 0)
  repeat:
      v_prev = copy(v_hat)
      for each state s:
          v_hat(s) = r(s) + γ · Σ_{s'} P(s'|s, π(s)) · v_prev(s')
  until max_s |v_hat(s) - v_prev(s)| < ε·(1-γ)/γ
  return v_hat
  ```
- **Convergence:** Guaranteed because the Bellman operator is a contraction with constant γ
- **Notes for Connect-4:** Exact evaluation is impossible. Use iterative evaluation with
  function approximation (e.g., a heuristic or neural network) to estimate board values.
  Terminal states have known values (win=1, loss=-1, draw=0).

---

## Module 8: Policy Iteration

### Policy Iteration Algorithm
- **What it does:** Finds the optimal policy by alternating between Policy Evaluation
  (computing v^π for the current policy) and Policy Improvement (updating π greedily
  based on Q-values). Guaranteed to converge to π* in finite MDPs.
- **Pseudocode:**
  ```
  initialize π randomly
  loop:
      # Policy Evaluation
      v_hat = iterative_policy_evaluation(π)
      # Policy Improvement
      π_new(s) = argmax_a q^π(s,a) for all s
          where q^π(s,a) = r(s) + γ · Σ_{s'} P(s'|s,a) · v_hat(s')
      if π_new == π: break
      π = π_new
  return π
  ```
- **Key formulas:**
  - Q-value (state-action value): q^π(s,a) = r(s) + γ · Σ_{s'} P(s'|s,a) · v^π(s')
  - Policy Improvement: π'(s) = argmax_a q^π(s,a)
  - Bellman Optimality: v^{π*}(s) = max_a q^{π*}(s,a)
- **Policy Improvement Theorem:** π' is always at least as good as π (never worse)
- **Notes for Connect-4:** Starting with a heuristic π (e.g., "always block opponent's
  3-in-a-row"), one step of Policy Improvement gives a provably better π'. By examining all
  7 possible drops, computing Q-values, and greedily picking max, we get a stronger policy.

---

## Module 9: General Policy Iteration (GPI)

### Value Iteration
- **What it does:** Merges Policy Evaluation and Improvement into one step by applying
  the Bellman Optimality Equation directly as an update rule.
- **Pseudocode:**
  ```
  initialize v_hat arbitrarily
  repeat:
      v_prev = copy(v_hat)
      for each state s:
          v_hat(s) = r(s) + γ · max_a Σ_{s'} P(s'|s,a) · v_prev(s')
  until converged
  return greedy policy: π(s) = argmax_a Σ_{s'} P(s'|s,a) · v_hat(s')
  ```
- **Key formula:** v_hat(s) ← r(s) + γ · max_a Σ_{s'} P(s'|s,a) · v_prev(s')
- **Note:** Value Iteration = Policy Iteration with u=1 evaluation step per improvement

### The Combined Operator (Modified Policy Iteration)
- **What it does:** Unifies Policy Iteration (u=∞ evaluation steps) and Value Iteration
  (u=1 step) under a single framework parameterized by u (number of Bellman updates per
  improvement step).
- **Update rule:**
  ```
  v_hat_{t+1}(s) ← r(s) + γ · max_a Σ_{s'} P(s'|s,a) · v_tilde_t(s')
  ```
  where v_tilde_t is the result of u consecutive Bellman updates initialized with v_hat_t
- **Convergence:** The combined operator is a contraction with constant γ^{u+1}
- **Modified Policy Iteration algorithm:**
  ```
  1. v_hat(s) = r(s) for terminal states, 0 otherwise
  2. π(s) = argmax_a Σ_{s'} P(s'|s,a) · v_hat(s')
  3. Apply u incremental PE steps using π → v_hat
  4. If v_hat changed by less than ε·(1-γ^{u+1})/γ^{u+1}: stop
     Else: go to step 2
  ```
- **Key insight:** u can be changed across iterations without losing convergence guarantees
- **Notes for Connect-4:** GPI provides the theoretical backbone for RL agents.
  We can train by playing games (updating policy asynchronously) while simultaneously
  improving the board evaluation function, without needing the full state space.

---

## Module 10: Bandits (Exploration vs Exploitation)

### Multi-Armed Bandit Problem
- **What it does:** Decides between exploiting the currently-known best action and exploring
  less-tried actions that might be better. Formalizes the exploration-exploitation tradeoff.
- **Formal setup:** At each round t, pick action At ∈ {1,..,K}. Observe reward Rt ~ P_{At}.
  Unknown: q(a) = E[R|A=a]. Goal: maximize cumulative reward.

### ε-Greedy
- **Pseudocode:**
  ```
  with probability 1-ε_t: pick a* = argmax_a q_hat(a)  # exploit
  with probability ε_t:   pick random a                  # explore
  update q_hat and N after observing reward
  ```
- **Issue:** ε-greedy explores uniformly, wasting pulls on clearly bad actions

### UCB1 (Upper Confidence Bound)
- **What it does:** Selects actions that either have high estimated value OR have been
  tried few times (high uncertainty). Automatically balances exploration and exploitation.
- **Pseudocode:**
  ```
  for each action a: maintain q_hat(a) and N(a)
  at step t:
      A_t = argmax_a [ q_hat(a) + c · sqrt(ln(t) / N(a)) ]
  observe reward, update q_hat(A_t) and N(A_t)
  ```
- **Key formula:** UCB score = q_hat(a) + c · sqrt(ln(t) / N(a))
  - First term: exploitation (prefer high-reward actions)
  - Second term: exploration (prefer rarely-tried actions)
- **Notes for Connect-4:** UCB1 is the foundation of MCTS (Module 13). In the game tree,
  each node is a bandit problem — which child to explore next? UCB1 (called UCT in this
  context) elegantly concentrates computation on promising moves.

---

## Module 11: Reinforcement Learning Basics

### RL Setting
- **What it does:** The agent does not know the MDP. It only observes: current state,
  available actions, chosen action's successor state, and reward. Learns from experience.
- **Trial interface:**
  1. Tells the agent available actions in state s
  2. Receives the agent's action choice
  3. Returns successor state and observed reward

### First-Visit Monte Carlo Policy Evaluation (FVMC)
- **What it does:** Estimates Q-values by averaging returns observed after the **first visit**
  to each (state, action) pair in a trial. Requires Exploring Starts to ensure all pairs
  are visited.
- **Pseudocode:**
  ```
  initialize q_hat(s,a) = 0, N(s,a) = 0 for all s,a
  repeat:
      generate trial τ using Exploring Starts + policy π
      # compute returns backwards from end of trial
      U = 0
      for t from T-1 down to 0:
          U = r(s_t) + γ · U
          if (s_t, a_t) is first visit in τ:
              N(s_t, a_t) += 1
              q_hat(s_t, a_t) += (U - q_hat(s_t, a_t)) / N(s_t, a_t)
      π = greedy policy from q_hat
  ```
- **Key formula (incremental update):**
  q_hat_n(s,a) = q_hat_{n-1}(s,a) + (1/n) · (U_n - q_hat_{n-1}(s,a))
- **Exploring Starts:** Begin each trial from a random (s,a) pair to guarantee all pairs
  are explored. In Connect-4: start games from random board configurations.
- **Notes for Connect-4:** With no prior knowledge, let the agent play thousands of games.
  After each game, update Q-values of every (board_state, column) pair visited, based on
  whether the game was won (+1) or lost (-1). Returns are computed backwards.

### Adaptive Dynamic Programming (ADP)
- **What it does:** Learns the MDP model (transition probabilities and rewards) from experience,
  then runs standard DP (Policy Evaluation or Value Iteration) on the learned model.
- **Pseudocode:**
  ```
  initialize counts C(s,a,s') = 0, reward_sum R(s) = 0
  for each trial:
      for each (s, a, s', r) in trial:
          C(s,a,s') += 1
          R(s) += r
      # update model
      P_hat(s'|s,a) = C(s,a,s') / Σ_{s''} C(s,a,s'')
      r_hat(s) = R(s) / visit_count(s)
      # run DP on learned model
      run_value_iteration(P_hat, r_hat)
  ```
- **Notes for Connect-4:** ADP is more sample-efficient than pure Monte Carlo but
  requires storing the transition model. For Connect-4's huge state space, this is
  only feasible with function approximation.

---

## Module 12: Competitive MDPs — Alternating Markov Games

### CRITICAL MODULE FOR CONNECT-4

### Ceteris Paribus Approach
- **What it does:** To find a generally optimal policy, we should learn against the
  **strongest possible opponent** — one who plays optimally against us. If we can beat
  the strongest opponent, we can beat anyone weaker.
- **Key insight:** A policy π₁ optimal against the strongest opponent π* is also optimal
  against all weaker opponents (transitivity assumption).
- **Implication:** Use **self-play** — the agent plays against itself, because self-play
  approximates playing against the strongest possible version of the agent.

### Alternating Markov Game (AMG) Formal Definition
- **Formal structure:** G = ⟨S, {Aᵢ}_{i=1}^k, P, {rᵢ}_{i=1}^k, γ, η⟩
  - Aᵢ: action space of player i
  - rᵢ: S → R reward function for player i
  - η: S → {1,...,k} determines the active player in each state
- **Loop:** At step t:
  1. Active player i = η(sₜ) picks action aₜ ∈ Aᵢ according to πᵢ(aₜ|sₜ)
  2. Environment transitions to sₜ₊₁ ~ P(·|sₜ,aₜ)
  3. All players receive rewards rⱼ(sₜ)

### Zero-Sum Self-Play Learning
- **What it does:** For 2-player zero-sum games (like Connect-4), learn a **single Q-function**
  for both players simultaneously. A win for one = loss for the other.
- **Key constraint:** γ = 1 (only the final result — win/loss — counts)
- **Pseudocode:**
  ```
  initialize q_hat(s,a) = 0, N(s,a) = 0 for all s,a
  repeat:
      generate shared trial (both players use UCB exploration)
      # compute returns backwards, FLIPPING SIGN at each step
      U = r(terminal_state)  # e.g., +1 if current player won
      for t from T-1 down to 0:
          U = -γ · U  # FLIP: what's good for me is bad for you
          if (s_t, a_t) is first visit:
              N(s_t, a_t) += 1
              q_hat(s_t, a_t) += (U - q_hat(s_t, a_t)) / N(s_t, a_t)
  ```
- **Key formula (sign flip):** U ← -γ · U at each backward step
- **CRITICAL: No Exploring Starts in competitive settings**
  - Exploring Starts (random initial state) is NOT used in competitive MDPs
  - Reason: in self-play, we need realistic game trajectories — starting from random
    board positions would bias learning toward unreachable states
  - **Replacement:** Use UCB-based exploration policy to ensure all relevant (s,a) pairs
    are visited while maintaining competitive play quality

### UCB Exploration for Competitive Settings
- **Formula:**
  ```
  u(s,a) = q_hat(s,a) + sqrt( ln(Σ_{a'} N[s,a']) / N[s,a] )
  ```
- **Why this works:** The exploring policy π' (UCB-based) and the greedy policy π
  (argmax q_hat) converge to the same policy for sufficiently large sample sizes.
  So even though we technically evaluate π' during trials, we converge to q^π.
- **Notes for Connect-4:**
  - State s encodes full board + whose turn it is
  - Actions: columns 0-6 (only non-full columns are valid)
  - r(s) = +1 if active player just won, -1 if lost, 0 if draw
  - Single Q-table serves both Red and Yellow (same function, perspective flips)

---

## Module 13: Online Policy Improvement — MCTS

### Motivation: Why Standard RL Fails for Sparse Rewards
- **Problem:** In standard trial-based GPI, the probability of reaching a specific state
  at depth d with branching factor b is 1/bᵈ
  - Example: b=10, 1000 trials → 63% chance of seeing a depth-3 state, only 9.5% for depth-4
  - Deep states are almost never revisited → their Q-values are unreliable noise
- **Solution:** Only update Q-values for **frequent states** (states close to the root)

### Trial-Based Online Policy Improvement
- **What it does:** Upgrades trial generation. Instead of using a fixed policy π, in each
  state sₜ during trial generation, run a **mini sub-RL process** with limited budget to
  approximate q(sₜ, a) for all a. Then pick the next action proportional to these estimates.
- **Key idea:** Improve the policy *during* trial generation (online), not after.

### Monte Carlo Tree Search (MCTS)
- **What it does:** Builds an explicit game tree rooted at the current state. Concentrates
  computation (simulations) on the most promising branches. Uses a **tree policy** (UCB1)
  for known nodes and a **default policy** (random) for unknown nodes.
- **Algorithm — 4 phases per iteration:**
  ```
  repeat until budget exhausted:

  1. SELECTION:
     Starting from root, use tree policy (UCB1) to traverse DOWN
     the existing tree until reaching a leaf node (unexplored or terminal)

  2. EXPANSION:
     If leaf is not terminal: add the leaf node to the tree
     (now it becomes a "known" node for future selections)

  3. SIMULATION (Rollout):
     From the new node, complete the game using the DEFAULT POLICY
     (typically: random moves for both players)
     Record the outcome U (win=+1, loss=-1, draw=0)

  4. BACKPROPAGATION:
     Walk back UP the path from new node to root
     For each node on the path:
         N[s][a] += 1
         q_hat(s,a) += (U - q_hat(s,a)) / N[s][a]
         U = -U  # flip perspective at each level (zero-sum)

  return action a* = argmax_a N[root][a]  # most visited = most promising
  ```
- **Tree policy (UCB1 / UCT):**
  ```
  a = argmax_a [ q_hat(s,a) + c · sqrt(ln(Σ_{a'} N[s,a']) / N[s,a]) ]
  ```
- **Why return most-visited action?** N[s][a] is more robust than q_hat(s,a) for the root
  because q_hat can be noisy from early simulations.
- **Computational budget:** Iterations (simulations) per move. More iterations = stronger play.
  Typical values: 100–10000 depending on time constraint.
- **Key parameters:**
  - c: exploration constant (higher c = more exploration; typical: c = √2)
  - Budget: number of simulations per move decision
- **Notes for Connect-4:**
  - Root = current board state
  - Children = board states after each of the 7 possible column drops (filtering full columns)
  - Default policy = random column selection (uniform over valid moves)
  - Backpropagation must flip U at each level (zero-sum: my win = your loss)
  - MCTS requires no domain knowledge about Connect-4 — it learns purely by simulation
  - Can be improved by using a smarter default policy (e.g., "always block opponent's
    winning move during rollout") → called "informed rollouts"
  - **Summary:** MCTS is the single most powerful algorithm for Connect-4 in this course.
    It combines UCB1 (Module 10), zero-sum self-play (Module 12), and online GPI (Module 13).

---

## Algorithm Selection Guide for Connect-4 Agents

| Approach | Core Module | Strength | Weakness |
|---|---|---|---|
| Pure MCTS | 13 | No domain knowledge needed, very strong | Needs many simulations = slow |
| Self-play Q-learning | 11 + 12 | Offline training, fast at inference | Needs many training games |
| UCB-based self-play | 10 + 12 | Clean theory, good exploration | Convergence can be slow |
| Policy Iteration + heuristic | 7 + 8 | Provably improves any starting heuristic | Needs hand-crafted heuristic |
| Genetic heuristic weights | 3 | Optimizes heuristic weights automatically | Indirect, slow convergence |
| MCTS + learned rollout | 10 + 13 | Best of both worlds | More complex to implement |

## Common Heuristic Features for Board Evaluation (if needed)
- Number of 2-in-a-rows with open ends (both sides)
- Number of 3-in-a-rows with open ends
- Immediate winning move available (+inf)
- Immediate blocking move needed (-inf if not taken)
- Center column control (center columns are more valuable)
- Total number of potential winning lines still open