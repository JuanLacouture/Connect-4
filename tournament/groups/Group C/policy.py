import pathlib
import numpy as np
from connect4.policy import Policy
from typing import override

# === Variable de configuración del curso (rúbrica criterio 2) ===
# El nº de episodios de entrenamiento se fija en train_group_c.py
# (TRAIN_EPISODES). Este agente solo CARGA la Q-table ya entrenada.

_QTABLE = pathlib.Path(__file__).parent / "qtable.npz"
_CENTER_ORDER = (3, 2, 4, 1, 5, 0, 6)
_Q_MIN_CONF = 1e-6  # bajo esto, usar piso heurístico


def _infer_me(b: np.ndarray) -> int:
    return -1 if np.count_nonzero(b) % 2 == 0 else 1


def _drop_row(b: np.ndarray, c: int) -> int:
    for r in range(5, -1, -1):
        if b[r, c] == 0:
            return r
    return -1


def _legal(b: np.ndarray) -> list[int]:
    return [c for c in range(7) if b[0, c] == 0]


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


def _encode_half(flat21: np.ndarray) -> int:
    # 21 dígitos base-3 -> int64 (3^21 ~ 1.05e10, cabe en int64).
    v = 0
    for d in flat21:
        v = v * 3 + int(d)
    return v


def canonical_key(board: np.ndarray, me: int) -> tuple[tuple[int, int], bool]:
    # Marco "desde-el-que-mueve": rel = board*me (lado a mover siempre +1).
    # Clave canónica = min(rel, espejo izq-der). Devuelve (clave, usó_espejo).
    rel = (board * me + 1).astype(np.int64)  # {-1,0,1} -> {0,1,2}
    mir = rel[:, ::-1]
    rf, mf = rel.reshape(-1), mir.reshape(-1)
    k_norm = (_encode_half(rf[:21]), _encode_half(rf[21:]))
    k_mir = (_encode_half(mf[:21]), _encode_half(mf[21:]))
    if k_mir < k_norm:
        return k_mir, True
    return k_norm, False


def _evaluate(b: np.ndarray, me: int) -> int:
    # Piso heurístico (mismo criterio que el agente Minimax de Group A).
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


def _heuristic_move(b: np.ndarray, me: int, legal: list[int]) -> int:
    best_c, best_v = legal[0], -(10**9)
    for c in sorted(legal, key=lambda x: _CENTER_ORDER.index(x)):
        r = _drop_row(b, c)
        b[r, c] = me
        v = _evaluate(b, me)
        b[r, c] = 0
        if v > best_v:
            best_v, best_c = v, c
    return best_c


_Q_CACHE: dict[tuple[int, int], np.ndarray] | None = None


def _load_qtable() -> dict[tuple[int, int], np.ndarray]:
    global _Q_CACHE
    if _Q_CACHE is None:
        if _QTABLE.exists():
            data = np.load(_QTABLE)
            hi, lo, qv = data["keys_hi"], data["keys_lo"], data["qvals"]
            _Q_CACHE = {(int(hi[i]), int(lo[i])): qv[i]
                        for i in range(len(hi))}
        else:
            _Q_CACHE = {}
    return _Q_CACHE


class QLearningPolicy(Policy):

    def __init__(self) -> None:
        self.Q: dict[tuple[int, int], np.ndarray] = {}

    @override
    def mount(self) -> None:
        self.Q = _load_qtable()

    @override
    def act(self, s: np.ndarray) -> int:
        board = s.copy()
        me = _infer_me(board)
        opp = -me
        legal = _legal(board)

        # Capa 1: ganar ya
        for c in legal:
            r = _drop_row(board, c)
            board[r, c] = me
            w = _wins_at(board, r, c, me)
            board[r, c] = 0
            if w:
                return c
        # Capa 2: bloquear victoria inmediata del rival
        for c in legal:
            r = _drop_row(board, c)
            board[r, c] = opp
            w = _wins_at(board, r, c, opp)
            board[r, c] = 0
            if w:
                return c

        # Capa 3: Q-table aprendida (frame canónico)
        key, mirrored = canonical_key(board, me)
        row = self.Q.get(key)
        if row is not None:
            best_c, best_q = None, -np.inf
            for c in legal:
                qc = 6 - c if mirrored else c
                if row[qc] > best_q:
                    best_q, best_c = row[qc], c
            if best_c is not None and best_q > _Q_MIN_CONF:
                return best_c

        # Piso heurístico (garantiza el gate aunque el entreno sea pobre)
        return _heuristic_move(board, me, legal)
