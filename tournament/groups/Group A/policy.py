"""
Group A — Titan
Algorithm: Iterative-Deepening Negamax + Alpha-Beta Pruning + Transposition Table

CONTEXT.md grounding (≥60 %):
  • GCS / constructive search backbone          → Module 2
  • Bellman optimality / value iteration on game tree → Module 9
  • Transposition table (path-independent quality)   → Module 4
  • Policy Improvement: greedy argmax q*(s,a)         → Module 8
  • Q-value definition: q(s,a) = r + γ·v(s')         → Module 7
External (≤40 %):
  • Alpha-beta pruning (efficiency trick, not a course algorithm)
  • Iterative deepening for anytime search
"""

import numpy as np
from connect4.policy import Policy
import time

ROWS = 6
COLS = 7
INF = 10_000_000
TIME_LIMIT = 1.75          # seconds per move
# Center-out move ordering improves pruning significantly
COL_ORDER = [3, 2, 4, 1, 5, 0, 6]


# ── board helpers ─────────────────────────────────────────────────────────────

def _valid(b):
    return [c for c in COL_ORDER if b[0, c] == 0]


def _drop(b, col, p):
    b2 = b.copy()
    for r in range(ROWS - 1, -1, -1):
        if b2[r, col] == 0:
            b2[r, col] = p
            return b2
    return b2


def _wins(b, p):
    # Horizontal
    for r in range(ROWS):
        for c in range(COLS - 3):
            if b[r, c] == b[r, c+1] == b[r, c+2] == b[r, c+3] == p:
                return True
    # Vertical
    for r in range(ROWS - 3):
        for c in range(COLS):
            if b[r, c] == b[r+1, c] == b[r+2, c] == b[r+3, c] == p:
                return True
    # Diagonal ↘
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if b[r, c] == b[r+1, c+1] == b[r+2, c+2] == b[r+3, c+3] == p:
                return True
    # Diagonal ↙
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            if b[r, c] == b[r-1, c+1] == b[r-2, c+2] == b[r-3, c+3] == p:
                return True
    return False


# ── static evaluation ─────────────────────────────────────────────────────────

def _score_window(w, p):
    opp = -p
    pc = int(np.sum(w == p))
    oc = int(np.sum(w == opp))
    if pc > 0 and oc > 0:
        return 0          # window is blocked
    if pc == 4:
        return 100
    if pc == 3:
        return 5
    if pc == 2:
        return 2
    if oc == 3:
        return -80        # urgent block bonus (defensive weight higher)
    if oc == 2:
        return -2
    return 0


def _evaluate(b, p):
    score = int(np.sum(b[:, COLS // 2] == p)) * 4
    opp = -p

    # Horizontal windows
    for r in range(ROWS):
        for c in range(COLS - 3):
            w = b[r, c:c + 4]
            score += _score_window(w, p)

    # Vertical windows
    for c in range(COLS):
        for r in range(ROWS - 3):
            w = b[r:r + 4, c]
            score += _score_window(w, p)

    # Diagonal ↘
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            w = np.array([b[r+i, c+i] for i in range(4)])
            score += _score_window(w, p)

    # Diagonal ↙
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            w = np.array([b[r-i, c+i] for i in range(4)])
            score += _score_window(w, p)

    return score


# ── agent ─────────────────────────────────────────────────────────────────────

class Titan(Policy):
    """
    Iterative-deepening negamax with alpha-beta pruning and a transposition table.

    The negamax formulation is a direct application of the Bellman optimality
    equation on the game tree (Module 9): v*(s) = max_a q*(s,a), with the
    zero-sum sign flip (Module 12) at every level.
    """

    def mount(self, action_timeout=None):
        self.tt = {}

    # ── helpers ──────────────────────────────────────────────────────────────

    def _me(self, b):
        """Determine whose turn it is from piece counts (Red=-1 always goes first)."""
        return -1 if int(np.sum(b == -1)) == int(np.sum(b == 1)) else 1

    # ── negamax ──────────────────────────────────────────────────────────────

    def _negamax(self, b, depth, alpha, beta, p, t0):
        # Timeout guard — return heuristic immediately
        if time.time() - t0 > TIME_LIMIT:
            return _evaluate(b, p), None

        # TT lookup
        key = b.tobytes()
        if key in self.tt:
            ts, td, tf, tc = self.tt[key]
            if td >= depth:
                if tf == 0:
                    return ts, tc
                if tf == 1 and ts >= beta:
                    return ts, tc
                if tf == -1 and ts <= alpha:
                    return ts, tc

        # Terminal: previous player won → current player loses
        if _wins(b, -p):
            return -INF - depth, None

        valid = _valid(b)
        if not valid:
            return 0, None

        if depth == 0:
            return _evaluate(b, p), None

        # Move ordering: check immediate wins first (fail-high early)
        for col in valid:
            nb = _drop(b, col, p)
            if _wins(nb, p):
                score = INF + depth
                self.tt[key] = (score, depth, 0, col)
                return score, col

        orig_alpha = alpha
        best_score = -INF - 1
        best_col = valid[0]

        for col in valid:
            nb = _drop(b, col, p)
            sc, _ = self._negamax(nb, depth - 1, -beta, -alpha, -p, t0)
            sc = -sc
            if sc > best_score:
                best_score = sc
                best_col = col
            alpha = max(alpha, sc)
            if alpha >= beta:
                break

        # Store TT entry with bound type
        if best_score <= orig_alpha:
            flag = -1   # upper bound (failed low)
        elif best_score >= beta:
            flag = 1    # lower bound (cut-off)
        else:
            flag = 0    # exact
        self.tt[key] = (best_score, depth, flag, best_col)

        return best_score, best_col

    # ── public interface ──────────────────────────────────────────────────────

    def act(self, s: np.ndarray) -> int:
        if not hasattr(self, 'tt'):
            self.tt = {}
        self.last_depth = 0
        p = self._me(s)
        t0 = time.time()
        valid = _valid(s)

        # Immediate win
        for col in valid:
            if _wins(_drop(s, col, p), p):
                return col
        # Immediate block
        for col in valid:
            if _wins(_drop(s, col, -p), -p):
                return col

        best_col = valid[0]

        # Iterative deepening — Bellman optimality applied level by level
        for depth in range(1, 14):
            if time.time() - t0 > TIME_LIMIT:
                break
            sc, col = self._negamax(s, depth, -INF, INF, p, t0)
            if col is not None:
                best_col = col
                self.last_depth = depth
            if sc >= INF:   # forced win found
                break

        return best_col
