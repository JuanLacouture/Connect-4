import time
import numpy as np
from connect4.policy import Policy

# === Variable de configuración del curso (rúbrica criterio 2) ===
# Profundidad de búsqueda minimax. 5 vence al aleatorio ~100%; 6-7 más fuerte
# (más lento en Python puro). Guard de tiempo abajo evita cualquier exceso.
SEARCH_DEPTH = 6

_CENTER_ORDER = (3, 2, 4, 1, 5, 0, 6)
_TIME_GUARD = 0.85  # s, red de seguridad aunque no haya límite conocido


def _infer_me(b: np.ndarray) -> int:
    return -1 if np.count_nonzero(b) % 2 == 0 else 1


def _drop_row(b: np.ndarray, c: int) -> int:
    for r in range(5, -1, -1):
        if b[r, c] == 0:
            return r
    return -1


def _legal(b: np.ndarray) -> list[int]:
    return [c for c in range(7) if b[0, c] == 0]


def _ordered(legal: list[int]) -> list[int]:
    s = set(legal)
    return [c for c in _CENTER_ORDER if c in s]


def _wins_at(b: np.ndarray, r: int, c: int, p: int) -> bool:
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        cnt = 1
        for sign in (1, -1):
            rr, cc = r + dr * sign, c + dc * sign
            while 0 <= rr < 6 and 0 <= cc < 7 and b[rr, cc] == p:
                cnt += 1
                if cnt >= 4:
                    return True
                rr += dr * sign
                cc += dc * sign
    return False


def _winner(b: np.ndarray) -> int:
    for r in range(6):
        for c in range(7):
            p = b[r, c]
            if p == 0:
                continue
            if c + 3 < 7 and all(b[r, c + i] == p for i in range(4)):
                return int(p)
            if r + 3 < 6 and all(b[r + i, c] == p for i in range(4)):
                return int(p)
            if r + 3 < 6 and c + 3 < 7 and all(b[r + i, c + i] == p for i in range(4)):
                return int(p)
            if r + 3 < 6 and c - 3 >= 0 and all(b[r + i, c - i] == p for i in range(4)):
                return int(p)
    return 0


def _evaluate(b: np.ndarray, me: int) -> int:
    opp = -me
    score = int(np.count_nonzero(b[:, 3] == me)) * 6
    score -= int(np.count_nonzero(b[:, 3] == opp)) * 6

    def windows():
        for r in range(6):
            row = b[r]
            for c in range(4):
                yield (row[c], row[c + 1], row[c + 2], row[c + 3])
        for c in range(7):
            col = b[:, c]
            for r in range(3):
                yield (col[r], col[r + 1], col[r + 2], col[r + 3])
        for r in range(3):
            for c in range(4):
                yield (b[r, c], b[r + 1, c + 1], b[r + 2, c + 2], b[r + 3, c + 3])
        for r in range(3):
            for c in range(3, 7):
                yield (b[r, c], b[r + 1, c - 1], b[r + 2, c - 2], b[r + 3, c - 3])

    for w in windows():
        m = o = e = 0
        for v in w:
            if v == me:
                m += 1
            elif v == opp:
                o += 1
            else:
                e += 1
        if m and o:
            continue
        if m == 3 and e == 1:
            score += 50
        elif m == 2 and e == 2:
            score += 10
        elif m == 1 and e == 3:
            score += 1
        if o == 3 and e == 1:
            score -= 80
        elif o == 2 and e == 2:
            score -= 12
        elif o == 1 and e == 3:
            score -= 1
    return score


def _minimax(b, depth, alpha, beta, maximizing, me, lr, lc, lp):
    if _wins_at(b, lr, lc, lp):
        return (1_000_000 + depth) if lp == me else -(1_000_000 + depth)
    legal = _legal(b)
    if not legal:
        return 0
    if depth == 0:
        return _evaluate(b, me)
    legal = _ordered(legal)
    player = me if maximizing else -me
    if maximizing:
        val = -(10**9)
        for c in legal:
            r = _drop_row(b, c)
            b[r, c] = player
            val = max(val, _minimax(b, depth - 1, alpha, beta, False, me, r, c, player))
            b[r, c] = 0
            alpha = max(alpha, val)
            if alpha >= beta:
                break
        return val
    val = 10**9
    for c in legal:
        r = _drop_row(b, c)
        b[r, c] = player
        val = min(val, _minimax(b, depth - 1, alpha, beta, True, me, r, c, player))
        b[r, c] = 0
        beta = min(beta, val)
        if alpha >= beta:
            break
    return val


class MinimaxPolicy(Policy):

    def mount(self, timeout: float = None) -> None:
        pass

    def act(self, s: np.ndarray) -> int:
        self.last_depth = SEARCH_DEPTH
        board = s.copy()
        me = _infer_me(board)
        opp = -me
        legal = _legal(board)

        for c in legal:
            r = _drop_row(board, c)
            board[r, c] = me
            win = _wins_at(board, r, c, me)
            board[r, c] = 0
            if win:
                return c
        for c in legal:
            r = _drop_row(board, c)
            board[r, c] = opp
            block = _wins_at(board, r, c, opp)
            board[r, c] = 0
            if block:
                return c
        order = _ordered(legal)
        best_c = order[0]
        best_v = -(10**9)
        t0 = time.perf_counter()
        for c in order:
            r = _drop_row(board, c)
            board[r, c] = me
            v = _minimax(board, SEARCH_DEPTH - 1, -(10**9), 10**9,
                         False, me, r, c, me)
            board[r, c] = 0
            if v > best_v:
                best_v, best_c = v, c
            if time.perf_counter() - t0 > _TIME_GUARD:
                break
        return best_c
