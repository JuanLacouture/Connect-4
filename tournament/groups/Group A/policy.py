"""
AGENTE CONNECT-4 CON MCTS MEJORADO


"""

import numpy as np
from connect4.policy import Policy
import math
import random
import time


ROWS = 6
COLS = 7


# ---------------------------------------------------------------------------
# Utilidades de tablero (funciones puras, sin clases, para máxima velocidad)
# ---------------------------------------------------------------------------

def get_free_cols(board: np.ndarray) -> list:
    return [c for c in range(COLS) if board[0, c] == 0]


def make_move(board: np.ndarray, col: int, player: int) -> np.ndarray:
    """Devuelve un nuevo tablero con el movimiento aplicado."""
    new_board = board.copy()
    for row in range(ROWS - 1, -1, -1):
        if new_board[row, col] == 0:
            new_board[row, col] = player
            return new_board
    return new_board  # no debería llegar aquí


def check_winner(board: np.ndarray) -> int:
    """Retorna -1, 1 o 0."""
    # Filas
    for r in range(ROWS):
        for c in range(COLS - 3):
            v = board[r, c]
            if v != 0 and v == board[r, c+1] == board[r, c+2] == board[r, c+3]:
                return v
    # Columnas
    for c in range(COLS):
        for r in range(ROWS - 3):
            v = board[r, c]
            if v != 0 and v == board[r+1, c] == board[r+2, c] == board[r+3, c]:
                return v
    # Diagonal ↘
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            v = board[r, c]
            if v != 0 and v == board[r+1, c+1] == board[r+2, c+2] == board[r+3, c+3]:
                return v
    # Diagonal ↙
    for r in range(ROWS - 3):
        for c in range(3, COLS):
            v = board[r, c]
            if v != 0 and v == board[r+1, c-1] == board[r+2, c-2] == board[r+3, c-3]:
                return v
    return 0


def is_terminal(board: np.ndarray) -> bool:
    return check_winner(board) != 0 or len(get_free_cols(board)) == 0


def winning_move(board: np.ndarray, player: int) -> int:
    """Retorna la columna ganadora para `player`, o -1 si no existe."""
    for col in get_free_cols(board):
        tmp = make_move(board, col, player)
        if check_winner(tmp) == player:
            return col
    return -1


def heuristic_rollout(board: np.ndarray, player: int) -> float:
    """
    Simulación de rollout con heurística simple:
    - Si puede ganar inmediatamente, lo hace.
    - Si el oponente puede ganar, lo bloquea.
    - Si no, juega aleatorio con preferencia al centro.
    Retorna la utilidad desde la perspectiva de player=-1 (RED).
    """
    current = player
    b = board.copy()

    for _ in range(ROWS * COLS):
        free = get_free_cols(b)
        if not free:
            break

        # 1. Mover ganador propio
        win_col = winning_move(b, current)
        if win_col != -1:
            b = make_move(b, win_col, current)
            current = -current
            winner = check_winner(b)
            if winner != 0:
                return 1.0 if winner == -1 else -1.0
            continue

        # 2. Bloquear al oponente
        block_col = winning_move(b, -current)
        if block_col != -1:
            col = block_col
        else:
            # 3. Jugar aleatoriamente con sesgo al centro
            weights = [5 - abs(c - 3) ** 2 for c in free]   # 4,3,4,4,3,4,4 → más peso al centro
            total = sum(weights)
            r = random.random() * total
            acc = 0
            col = free[-1]
            for c, w in zip(free, weights):
                acc += w
                if r <= acc:
                    col = c
                    break

        b = make_move(b, col, current)
        if is_terminal(b):
            w = check_winner(b)
            return 1.0 if w == -1 else (-1.0 if w == 1 else 0.0)
        current = -current

    return 0.0


# ---------------------------------------------------------------------------
# Nodo MCTS
# ---------------------------------------------------------------------------

class MCTSNode:
    __slots__ = ('board', 'player', 'parent', 'children',
                 'visits', 'value', 'untried_moves')

    def __init__(self, board: np.ndarray, player: int, parent=None):
        self.board = board
        self.player = player          # jugador que VA A MOVER desde este nodo
        self.parent = parent
        self.children: dict = {}      # col -> MCTSNode
        self.visits: int = 0
        self.value: float = 0.0
        self.untried_moves: list = get_free_cols(board) if not is_terminal(board) else []

    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0

    def ucb_score(self, c: float = 1.0) -> float:
        if self.visits == 0:
            return float('inf')
        exploitation = self.value / self.visits
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def best_child(self) -> 'MCTSNode':
        return max(self.children.values(), key=lambda n: n.ucb_score())

    def expand(self) -> 'MCTSNode':
        col = self.untried_moves.pop(random.randrange(len(self.untried_moves)))
        new_board = make_move(self.board, col, self.player)
        child = MCTSNode(new_board, -self.player, parent=self)
        self.children[col] = child
        return child

    def update(self, result: float):
        self.visits += 1
        self.value += result


# ---------------------------------------------------------------------------
# Agente principal
# ---------------------------------------------------------------------------

class MCTSAgentTorneo(Policy):
    """
    Agente MCTS para el torneo de Connect-4.

    Implementa la interfaz Policy:
      - mount(timeout=None): reinicia el agente antes de cada partida.
      - act(s): elige la columna a jugar dado el tablero s.
    """

    def __init__(self, simulations: int = 2000, time_limit: float = 5.5):
        self.simulations = simulations
        self.time_limit = time_limit   # segundos máximos por movimiento

    # ------------------------------------------------------------------
    # Interfaz Policy
    # ------------------------------------------------------------------

    def mount(self, timeout: float = None) -> None:
        """
        Inicializa el agente antes de cada partida.
        Acepta un argumento de timeout opcional (requerido por el grader).
        """
        if timeout is not None:
            # Usar el timeout para calibrar las simulaciones; reservamos
            # un margen de seguridad del 10 %.
            self.time_limit = timeout * 0.90
        else:
            self.time_limit = 4.5

    def act(self, s: np.ndarray) -> int:
        """
        Elige un movimiento para el tablero `s`.

        s:  array (6, 7)
            0  = vacío
           -1  = RED  (nuestro jugador)
            1  = YELLOW (oponente)

        Retorna: columna (0-6)
        """
        free = get_free_cols(s)

        if not free:
            raise ValueError("No hay columnas disponibles.")

        # Caso trivial: un solo movimiento posible
        if len(free) == 1:
            return free[0]

        # --- Jugada inmediata ganadora ---
        w = winning_move(s, -1)
        if w != -1:
            return w

        # --- Bloqueo inmediato de derrota ---
        b = winning_move(s, 1)
        if b != -1:
            return b

        # --- MCTS ---
        root = MCTSNode(s.copy(), player=-1)   # siempre somos RED (-1)
        deadline = time.time() + self.time_limit
        sims = 0

        while sims < self.simulations and time.time() < deadline:
            node = self._select(root)
            result = heuristic_rollout(node.board, node.player)
            self._backpropagate(node, result)
            sims += 1

        # Elegir el hijo con más visitas (más robusto que el mejor UCB)
        if not root.children:
            # Si nunca se expandió, preferir centro
            for col in [3, 2, 4, 1, 5, 0, 6]:
                if col in free:
                    return col
            return free[0]

        best_col = max(root.children.items(), key=lambda x: x[1].visits)[0]
        return int(best_col)

    # ------------------------------------------------------------------
    # Fases MCTS internas
    # ------------------------------------------------------------------

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Selection + Expansion en un solo paso."""
        while not is_terminal(node.board):
            if not node.is_fully_expanded():
                return node.expand()          # Expansion
            node = node.best_child()          # Selection
        return node

    def _backpropagate(self, node: MCTSNode, result: float):
        """Backpropagation: propagar el resultado hacia la raíz."""
        current = node
        # `result` está expresado desde la perspectiva de RED (-1).
        # Cada nodo guarda la utilidad del jugador que ACABA de mover,
        # por eso alternamos el signo al subir.
        # El nodo hoja tiene player = jugador que va a mover A CONTINUACIÓN,
        # así que el jugador que movió para llegar aquí es -node.player.
        # Usamos la convención: value positivo = bueno para el nodo.
        sign = 1.0 if node.player == 1 else -1.0   # nodo hoja: player = próximo a mover
        while current is not None:
            current.update(result * sign)
            sign = -sign
            current = current.parent


# Alias para compatibilidad con descubrimiento automático del torneo
class AgenteMCTS(MCTSAgentTorneo):
    pass
