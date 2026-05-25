"""
Group B — Hydra
Algorithm: MCTS + UCB1 + Heuristic Rollouts + Persistent Tree Reuse + Open-Three Block

CONTEXT.md grounding (≥60 %):
  • MCTS — 4 phases (selection/expansion/simulation/backprop) → Module 13
  • UCB1 tree policy                                          → Module 10
  • Zero-sum self-play + sign flip at each level              → Module 12
  • Policy Improvement via online GPI (tree accumulates)      → Module 9 / 13
External (≤40 %):
  • Heuristic rollout bias (win/block/center preference)
  • Persistent tree reuse between moves within a game
  • Open-three block (scan all 4-windows, block any 3+empty where empty is playable now)
"""

import numpy as np
from connect4.policy import Policy
import time
import math

ROWS = 6
COLS = 7
C_UCB = math.sqrt(2)
TIME_LIMIT = 1.75


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


# ── open-three threat detection ──────────────────────────────────────────────

def _next_row(b, col):
    """Row where a piece would land if dropped in col (-1 if full)."""
    for r in range(ROWS - 1, -1, -1):
        if b[r, col] == 0:
            return r
    return -1


def _open_three_block_cols(b, p):
    """
    Columns where dropping our piece blocks opponent p's 3-in-a-row threat.
    Scans every 4-cell window; if p has 3 pieces and 1 empty cell, and that
    empty cell is the next playable position in its column, that column is a threat.
    """
    threats = set()
    opp = p  # we want to block opp = p
    valid_set = set(_valid(b))

    def _check_window(cells):
        # cells = list of (row, col) positions in this 4-window
        vals = [b[r, c] for r, c in cells]
        if vals.count(opp) == 3 and vals.count(0) == 1:
            idx = vals.index(0)
            er, ec = cells[idx]
            if ec in valid_set and _next_row(b, ec) == er:
                threats.add(ec)

    for r in range(ROWS):
        for c in range(COLS - 3):
            _check_window([(r, c+i) for i in range(4)])
    for r in range(ROWS - 3):
        for c in range(COLS):
            _check_window([(r+i, c) for i in range(4)])
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            _check_window([(r+i, c+i) for i in range(4)])
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            _check_window([(r-i, c+i) for i in range(4)])

    return list(threats)


# ── heuristic rollout ─────────────────────────────────────────────────────────

_CENTER_W = np.array([1, 2, 3, 4, 3, 2, 1], dtype=float)


def _rollout(b, start_p):
    """
    Informed rollout:
      1. Take immediate win.
      2. Block opponent's immediate win.
      3. Otherwise center-biased random.
    Returns +1 if start_p wins, -1 if loses, 0 draw.
    """
    board = b.copy()
    p = start_p
    for _ in range(ROWS * COLS):
        valid = [c for c in range(COLS) if board[0, c] == 0]
        if not valid:
            return 0.0
        for col in valid:
            if _wins(_drop(board, col, p), p):
                return 1.0 if p == start_p else -1.0
        opp = -p
        block = None
        for col in valid:
            if _wins(_drop(board, col, opp), opp):
                block = col
                break
        if block is not None:
            col = block
        else:
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
    __slots__ = ('board', 'player', 'col', 'parent', 'children', 'W', 'N', 'untried')

    def __init__(self, board, player, col=None, parent=None):
        self.board = board
        self.player = player
        self.col = col
        self.parent = parent
        self.children = []
        self.W = 0.0
        self.N = 0
        valid = [c for c in range(COLS) if board[0, c] == 0]
        self.untried = valid[:]
        np.random.shuffle(self.untried)

    def ucb(self):
        if self.N == 0:
            return float('inf')
        return self.W / self.N + C_UCB * math.sqrt(math.log(self.parent.N) / self.N)


# ── agent ─────────────────────────────────────────────────────────────────────

class Hydra(Policy):
    """
    MCTS (Module 13) + UCB1 (Module 10) + zero-sum backprop (Module 12).
    Heuristic rollouts guide simulation. The tree is reused across moves:
    after each act() the root is saved; the next call warm-starts from the
    matching grandchild node, keeping all prior simulations.
    """

    def mount(self, action_timeout=None):
        self._prev_root = None

    def _me(self, b):
        return -1 if int(np.sum(b == -1)) == int(np.sum(b == 1)) else 1

    # ── backpropagation ───────────────────────────────────────────────────────

    def _backprop(self, node, result):
        r = -result
        n = node
        while n is not None:
            n.N += 1
            n.W += r
            r = -r
            n = n.parent

    # ── tree reuse ────────────────────────────────────────────────────────────

    def _find_reuse_node(self, board_key):
        """
        Search 2 levels into _prev_root for a node whose board matches
        board_key (= current board after our move + opponent's response).
        """
        if self._prev_root is None:
            return None
        for c1 in self._prev_root.children:
            for c2 in c1.children:
                if c2.board.tobytes() == board_key:
                    return c2
        return None

    # ── main search ───────────────────────────────────────────────────────────

    def _mcts(self, board, p, t0):
        reuse = self._find_reuse_node(board.tobytes())
        if reuse is not None:
            root = reuse
            root.parent = None
        else:
            root = _Node(board, p)

        self.last_iterations = 0

        while time.time() - t0 < TIME_LIMIT:
            self.last_iterations += 1
            # ── SELECTION ────────────────────────────────────────────────────
            node = root
            while not node.untried and node.children:
                node = max(node.children, key=lambda n: n.ucb())

            # ── terminal check ───────────────────────────────────────────────
            if _wins(node.board, -node.player):
                self._backprop(node, -1.0)
                continue

            if not node.untried and not node.children:
                self._backprop(node, 0.0)
                continue

            # ── EXPANSION ────────────────────────────────────────────────────
            col = node.untried.pop()
            nb = _drop(node.board, col, node.player)
            child = _Node(nb, -node.player, col, parent=node)
            node.children.append(child)

            # ── SIMULATION ───────────────────────────────────────────────────
            if _wins(nb, node.player):
                result = -1.0
            else:
                result = _rollout(nb, child.player)

            # ── BACKPROPAGATION ───────────────────────────────────────────────
            self._backprop(child, result)

        self._prev_root = root

        if not root.children:
            return _valid(board)[0]
        return max(root.children, key=lambda n: n.N).col

    # ── public interface ──────────────────────────────────────────────────────

    def act(self, s: np.ndarray) -> int:
        if not hasattr(self, '_prev_root'):
            self._prev_root = None
        p = self._me(s)
        t0 = time.time()
        valid = _valid(s)

        # 1. Immediate win
        for col in valid:
            if _wins(_drop(s, col, p), p):
                return col

        # 2. Block immediate loss
        for col in valid:
            if _wins(_drop(s, col, -p), -p):
                return col

        # 3. Block opponent open-three threats (3-in-a-row completable this turn)
        opp_threes = _open_three_block_cols(s, -p)
        if len(opp_threes) == 1:
            return opp_threes[0]

        return self._mcts(s, p, t0)
