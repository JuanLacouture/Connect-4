import math
import time
import numpy as np
from connect4.policy import Policy
from typing import override

# === Variables de configuración del curso (rúbrica criterio 2) ===
MCTS_SIMULATIONS = 3000
UCT_C = math.sqrt(2)

_SEED = 1234
_CENTER_ORDER = (3, 2, 4, 1, 5, 0, 6)
_TIME_GUARD = 0.85  # s, red de seguridad


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


def _tactical_move(b: np.ndarray, p: int, legal: list[int]) -> int | None:
    # Si p puede ganar ya, jugarlo; si no, bloquear victoria inmediata del rival.
    for c in legal:
        r = _drop_row(b, c)
        b[r, c] = p
        w = _wins_at(b, r, c, p)
        b[r, c] = 0
        if w:
            return c
    for c in legal:
        r = _drop_row(b, c)
        b[r, c] = -p
        w = _wins_at(b, r, c, -p)
        b[r, c] = 0
        if w:
            return c
    return None


class _Node:
    __slots__ = ("board", "to_move", "parent", "move", "children",
                 "untried", "N", "W", "terminal")

    def __init__(self, board, to_move, parent, move):
        self.board = board
        self.to_move = to_move          # jugador a mover en este estado
        self.parent = parent
        self.move = move                # columna jugada desde el padre
        self.children: dict[int, _Node] = {}
        self.untried = _ordered(_legal(board))
        self.N = 0
        self.W = 0.0                    # valor para el que movió HACIA este nodo
        self.terminal: int | None = None  # ganador (-1/0/1) si terminal


class MCTSPolicy(Policy):

    def __init__(self) -> None:
        self.rng = np.random.default_rng(_SEED)

    @override
    def mount(self) -> None:
        pass

    @override
    def act(self, s: np.ndarray) -> int:
        board = s.copy()
        me = _infer_me(board)
        legal = _legal(board)

        tac = _tactical_move(board, me, legal)
        if tac is not None:
            return tac

        root = _Node(board.copy(), me, None, None)
        t0 = time.perf_counter()
        for _ in range(MCTS_SIMULATIONS):
            if time.perf_counter() - t0 > _TIME_GUARD:
                break
            node = self._select(root)
            node = self._expand(node)
            result = self._rollout(node)
            self._backprop(node, result)

        best = max(root.children.values(), key=lambda n: n.N)
        return best.move

    def _select(self, node: _Node) -> _Node:
        while node.terminal is None and not node.untried and node.children:
            logN = math.log(node.N)
            best_uct = -math.inf
            best = None
            for ch in node.children.values():
                if ch.N == 0:
                    best = ch
                    break
                uct = ch.W / ch.N + UCT_C * math.sqrt(logN / ch.N)
                if uct > best_uct:
                    best_uct, best = uct, ch
            node = best
        return node

    def _expand(self, node: _Node) -> _Node:
        if node.terminal is not None:
            return node
        w = _winner(node.board)
        if w != 0 or not node.untried:
            node.terminal = w if w != 0 else 0
            if not node.untried and not _legal(node.board):
                node.terminal = w
            return node
        c = node.untried.pop()
        nb = node.board.copy()
        r = _drop_row(nb, c)
        nb[r, c] = node.to_move
        child = _Node(nb, -node.to_move, node, c)
        if _wins_at(nb, r, c, node.to_move):
            child.terminal = node.to_move
        elif not _legal(nb):
            child.terminal = 0
        node.children[c] = child
        return child

    def _rollout(self, node: _Node) -> int:
        if node.terminal is not None:
            return node.terminal
        b = node.board.copy()
        cur = node.to_move
        while True:
            legal = _legal(b)
            if not legal:
                return 0
            mv = _tactical_move(b, cur, legal)
            if mv is None:
                mv = int(self.rng.choice(legal))
            r = _drop_row(b, mv)
            b[r, mv] = cur
            if _wins_at(b, r, mv, cur):
                return cur
            cur = -cur

    def _backprop(self, node: _Node, result: int) -> None:
        while node is not None:
            node.N += 1
            mover = -node.to_move  # quien movió hacia 'node'
            if result == mover:
                node.W += 1.0
            elif result == -mover:
                node.W -= 1.0
            node = node.parent
