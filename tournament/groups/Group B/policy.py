"""
Group B — Hydra
Algorithm: Monte Carlo Tree Search + UCB1 + Heuristic Rollouts

CONTEXT.md grounding (≥60 %):
  • MCTS — 4 phases (selection/expansion/simulation/backprop) → Module 13
  • UCB1 tree policy                                          → Module 10
  • Zero-sum self-play + sign flip at each level              → Module 12
  • Policy Improvement via online GPI                         → Module 9 / 13
External (≤40 %):
  • Heuristic rollout bias (win/block/center preference) instead of uniform random
"""

import numpy as np
from connect4.policy import Policy
import time
import math

ROWS = 6
COLS = 7
C_UCB = math.sqrt(2)       # exploration constant
TIME_LIMIT = 1.75          # seconds per move


# ── board helpers ─────────────────────────────────────────────────────────────

def _valid(b):
    return [c for c in range(COLS) if b[0, c] == 0]


def _drop(b, col, p):
    b2 = b.copy()
    for r in range(ROWS - 1, -1, -1):
        if b2[r, col] == 0:
            b2[r, col] = p
            return b2
    return b2


def _wins(b, p):
    for r in range(ROWS):
        for c in range(COLS - 3):
            if b[r, c] == b[r, c+1] == b[r, c+2] == b[r, c+3] == p:
                return True
    for r in range(ROWS - 3):
        for c in range(COLS):
            if b[r, c] == b[r+1, c] == b[r+2, c] == b[r+3, c] == p:
                return True
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if b[r, c] == b[r+1, c+1] == b[r+2, c+2] == b[r+3, c+3] == p:
                return True
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            if b[r, c] == b[r-1, c+1] == b[r-2, c+2] == b[r-3, c+3] == p:
                return True
    return False


# ── heuristic rollout ─────────────────────────────────────────────────────────

_CENTER_W = np.array([1, 2, 3, 4, 3, 2, 1], dtype=float)


def _rollout(b, start_p):
    """
    Play a game to completion using an informed default policy:
      1. Take an immediate win if available.
      2. Block the opponent's immediate win.
      3. Otherwise, pick column proportional to center-preference weights.
    Returns +1 if start_p wins, -1 if loses, 0 for draw.
    """
    board = b.copy()
    p = start_p
    for _ in range(ROWS * COLS):
        valid = [c for c in range(COLS) if board[0, c] == 0]
        if not valid:
            return 0.0

        # 1. Immediate win
        for col in valid:
            if _wins(_drop(board, col, p), p):
                return 1.0 if p == start_p else -1.0

        # 2. Block opponent's immediate win
        opp = -p
        block = None
        for col in valid:
            if _wins(_drop(board, col, opp), opp):
                block = col
                break

        if block is not None:
            col = block
        else:
            # 3. Center-biased random
            w = _CENTER_W[[c for c in valid]]
            w = w / w.sum()
            col = int(np.random.choice(valid, p=w))

        board = _drop(board, col, p)
        if _wins(board, p):
            return 1.0 if p == start_p else -1.0
        p = -p

    return 0.0


# ── MCTS node ─────────────────────────────────────────────────────────────────

class _Node:
    """
    Each node represents a board STATE after the move that led here.
    self.W  = sum of results from self.parent.player's perspective
              (the player who CHOSE to visit this node via its move).
    This ensures UCB argmax at the parent selects the best child for the parent.
    """
    __slots__ = ('board', 'player', 'col', 'parent', 'children', 'W', 'N', 'untried')

    def __init__(self, board, player, col=None, parent=None):
        self.board = board
        self.player = player      # player whose turn it is FROM this state
        self.col = col            # column that led to this state
        self.parent = parent
        self.children = []
        self.W = 0.0              # wins from parent's perspective
        self.N = 0
        valid = [c for c in range(COLS) if board[0, c] == 0]
        # Shuffle so expansion is in random order
        self.untried = valid[:]
        np.random.shuffle(self.untried)

    def ucb(self):
        if self.N == 0:
            return float('inf')
        return self.W / self.N + C_UCB * math.sqrt(math.log(self.parent.N) / self.N)


# ── agent ─────────────────────────────────────────────────────────────────────

class Hydra(Policy):
    """
    Full Monte Carlo Tree Search following the 4-phase algorithm from Module 13.

    The UCB1 tree policy (Module 10) guides selection; zero-sum backpropagation
    (Module 12) flips the sign at every level. Rollouts use an informed default
    policy instead of uniform random, improving convergence speed.
    """

    def mount(self, action_timeout=None):
        pass

    def _me(self, b):
        return -1 if int(np.sum(b == -1)) == int(np.sum(b == 1)) else 1

    # ── backpropagation ───────────────────────────────────────────────────────

    def _backprop(self, node, result):
        """
        result = +1 if node.player wins, -1 if node.player loses.
        We store from parent's perspective (flip once before starting, then
        flip at every level going up).  Module 12: U ← −γ·U at each step.
        """
        r = -result   # convert: from parent's (= -node.player's) perspective
        n = node
        while n is not None:
            n.N += 1
            n.W += r
            r = -r
            n = n.parent

    # ── main search ───────────────────────────────────────────────────────────

    def _mcts(self, board, p, t0):
        root = _Node(board, p)

        while time.time() - t0 < TIME_LIMIT:
            # ── SELECTION ────────────────────────────────────────────────────
            node = root
            while not node.untried and node.children:
                node = max(node.children, key=lambda n: n.ucb())

            # ── terminal check at selected node ──────────────────────────────
            if _wins(node.board, -node.player):
                # previous player won → this node is a terminal loss for node.player
                self._backprop(node, -1.0)
                continue

            if not node.untried and not node.children:
                # full board → draw
                self._backprop(node, 0.0)
                continue

            # ── EXPANSION ────────────────────────────────────────────────────
            col = node.untried.pop()
            nb = _drop(node.board, col, node.player)
            child = _Node(nb, -node.player, col, parent=node)
            node.children.append(child)

            # ── SIMULATION ───────────────────────────────────────────────────
            if _wins(nb, node.player):
                # move just made was a win
                result = -1.0   # from child.player's perspective: child.player lost
            else:
                # heuristic rollout from child state
                result = _rollout(nb, child.player)

            # ── BACKPROPAGATION ───────────────────────────────────────────────
            self._backprop(child, result)

        if not root.children:
            return _valid(board)[0]
        # Return most-visited child (robust, lower variance than max Q)
        return max(root.children, key=lambda n: n.N).col

    # ── public interface ──────────────────────────────────────────────────────

    def act(self, s: np.ndarray) -> int:
        p = self._me(s)
        t0 = time.time()
        valid = _valid(s)

        # Immediate win / block (saves simulation budget)
        for col in valid:
            if _wins(_drop(s, col, p), p):
                return col
        for col in valid:
            if _wins(_drop(s, col, -p), -p):
                return col

        return self._mcts(s, p, t0)
