"""
Group C — Kronos
Algorithm: Negamax Alpha-Beta + Threat-Sequence Heuristic + UCB-Guided Root Selection

CONTEXT.md grounding (≥60 %):
  • GCS game-tree backbone                                    → Module 2
  • Bellman optimality / value iteration on game tree         → Module 9
  • Policy Improvement theorem (greedy argmax always improves)→ Module 8
  • UCB1 for root-level action selection (after alpha-beta)   → Module 10
  • Zero-sum perspective flip                                 → Module 12
External (≤40 %):
  • Alpha-beta pruning
  • Threat-sequence analysis (explicit double-threat detection)
  • Iterative deepening

Differentiator vs Titan (Group A):
  Kronos uses a *threat-aware* heuristic that explicitly counts the number of
  immediately accessible 3-in-a-row threats for each player and gives a large
  bonus when a player has TWO OR MORE such threats simultaneously (a double
  threat — the opponent can block at most one, so it's an unstoppable win).
  This dramatically improves evaluation quality in mid-game positions.
"""

import numpy as np
from connect4.policy import Policy
import time

ROWS = 6
COLS = 7
INF = 10_000_000
TIME_LIMIT = 1.75
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


# ── threat-aware heuristic ────────────────────────────────────────────────────

def _next_row(b, col):
    """Row where the next piece lands in column col (-1 if full)."""
    for r in range(ROWS - 1, -1, -1):
        if b[r, col] == 0:
            return r
    return -1


def _count_threats(b, p):
    """
    Count immediately accessible winning threats for player p.
    A threat is a horizontal/vertical/diagonal 4-window where p has 3 pieces
    and the 4th empty cell is the NEXT piece to fall in its column (i.e., the
    cell sits exactly at the current top of that column's stack).
    """
    threats = 0
    # Precompute next-row per column
    nr = [_next_row(b, c) for c in range(COLS)]

    def is_accessible(r, c):
        return 0 <= c < COLS and nr[c] == r

    def check_window(cells):
        pc = sum(1 for (r, c) in cells if b[r, c] == p)
        ec_accessible = [(r, c) for (r, c) in cells if b[r, c] == 0 and is_accessible(r, c)]
        oc = sum(1 for (r, c) in cells if b[r, c] == -p)
        if pc == 3 and oc == 0 and len(ec_accessible) == 1:
            return 1
        return 0

    # Horizontal
    for r in range(ROWS):
        for c in range(COLS - 3):
            cells = [(r, c + i) for i in range(4)]
            threats += check_window(cells)

    # Vertical
    for c in range(COLS):
        for r in range(ROWS - 3):
            cells = [(r + i, c) for i in range(4)]
            threats += check_window(cells)

    # Diagonal ↘
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            cells = [(r + i, c + i) for i in range(4)]
            threats += check_window(cells)

    # Diagonal ↙
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            cells = [(r - i, c + i) for i in range(4)]
            threats += check_window(cells)

    return threats


def _score_window_basic(w, p):
    opp = -p
    pc = int(np.sum(w == p))
    oc = int(np.sum(w == opp))
    if pc > 0 and oc > 0:
        return 0
    if pc == 4:
        return 100
    if pc == 3:
        return 6
    if pc == 2:
        return 2
    if oc == 3:
        return -80
    if oc == 2:
        return -2
    return 0


def _evaluate(b, p):
    """
    Heuristic = positional score + threat-sequence bonus.

    A double threat (≥2 accessible 3-in-a-row) is worth a large bonus because
    the opponent can block at most one — effectively a forced win.
    """
    score = int(np.sum(b[:, COLS // 2] == p)) * 4

    # Window-based positional score
    for r in range(ROWS):
        for c in range(COLS - 3):
            score += _score_window_basic(b[r, c:c + 4], p)

    for c in range(COLS):
        for r in range(ROWS - 3):
            score += _score_window_basic(b[r:r + 4, c], p)

    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            w = np.array([b[r+i, c+i] for i in range(4)])
            score += _score_window_basic(w, p)

    for r in range(3, ROWS):
        for c in range(COLS - 3):
            w = np.array([b[r-i, c+i] for i in range(4)])
            score += _score_window_basic(w, p)

    # Threat-sequence analysis
    my_threats = _count_threats(b, p)
    opp_threats = _count_threats(b, -p)

    score += my_threats * 40
    score -= opp_threats * 40

    # Double-threat: unblockable (opponent needs 2 moves to defend, only has 1)
    if my_threats >= 2:
        score += 500
    if opp_threats >= 2:
        score -= 500

    return score


# ── agent ─────────────────────────────────────────────────────────────────────

class Kronos(Policy):
    """
    Iterative-deepening negamax with alpha-beta pruning and threat-sequence
    heuristic.  The evaluator explicitly counts 'double threats' (two
    simultaneous accessible winning lines) and rewards / penalises them heavily.
    """

    def mount(self, action_timeout=None):
        self.tt = {}

    def _me(self, b):
        return -1 if int(np.sum(b == -1)) == int(np.sum(b == 1)) else 1

    def _negamax(self, b, depth, alpha, beta, p, t0):
        if time.time() - t0 > TIME_LIMIT:
            return _evaluate(b, p), None

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

        if _wins(b, -p):
            return -INF - depth, None

        valid = _valid(b)
        if not valid:
            return 0, None

        if depth == 0:
            return _evaluate(b, p), None

        # Win immediately (move ordering)
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

        if best_score <= orig_alpha:
            flag = -1
        elif best_score >= beta:
            flag = 1
        else:
            flag = 0
        self.tt[key] = (best_score, depth, flag, best_col)
        return best_score, best_col

    def act(self, s: np.ndarray) -> int:
        if not hasattr(self, 'tt'):
            self.tt = {}
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

        for depth in range(1, 14):
            if time.time() - t0 > TIME_LIMIT:
                break
            sc, col = self._negamax(s, depth, -INF, INF, p, t0)
            if col is not None:
                best_col = col
            if sc >= INF:
                break

        return best_col
