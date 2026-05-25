"""
POLÍTICA MINIMAX PARA GROUP B
Usa el algoritmo Minimax con poda alfa-beta para encontrar el mejor movimiento.
Evalúa el tablero buscando en profundidad y minimizando/maximizando valores.
"""

import numpy as np
from connect4.policy import Policy
from connect4.connect_state import ConnectState
from typing import override


class MinimaxPolicy(Policy):
    """Política inteligente basada en Minimax con evaluación heurística."""
    
    def __init__(self, depth: int = 5):
        """
        Args:
            depth: Profundidad de búsqueda (más = más inteligente pero lento)
                   Recomendado: 4-6
        """
        self.depth = depth
        self.max_player = -1  # Nuestro jugador (rojo)
        self.min_player = 1   # Oponente (amarillo)
    
    @override
    def mount(self) -> None:
        """Llamado antes de cada partida."""
        pass
    
    @override
    def act(self, board: np.ndarray) -> int:
        """
        Elige el mejor movimiento usando Minimax.
        """
        state = ConnectState(board, self.max_player)
        available = state.get_free_cols()
        
        if not available:
            return available[0]
        
        best_value = float('-inf')
        best_move = available[0]
        
        # Evaluar cada movimiento disponible
        for col in available:
            next_state = state.transition(col)
            value = self._minimax(next_state, self.depth - 1, float('-inf'), float('inf'), False)
            
            if value > best_value:
                best_value = value
                best_move = col
        
        return best_move
    
    def _minimax(self, state: ConnectState, depth: int, alpha: float, beta: float, is_max: bool) -> float:
        """
        Minimax con poda alfa-beta.
        
        Args:
            state: Estado actual del juego
            depth: Profundidad restante para buscar
            alpha: Mejor valor encontrado para maximizador
            beta: Mejor valor encontrado para minimizador
            is_max: True si es turno del maximizador (nosotros)
        
        Returns:
            Valor evaluado del estado
        """
        # Condiciones de parada
        winner = state.get_winner()
        if winner == self.max_player:
            return 1000 + depth  # Ganar rápido es mejor
        elif winner == self.min_player:
            return -1000 - depth  # Perder lento es mejor que rápido
        elif depth == 0 or state.is_final():
            return self._evaluate_board(state.board)
        
        available = state.get_free_cols()
        
        if is_max:  # Maximizador (nosotros: -1)
            max_value = float('-inf')
            for col in available:
                next_state = state.transition(col)
                value = self._minimax(next_state, depth - 1, alpha, beta, False)
                max_value = max(max_value, value)
                alpha = max(alpha, value)
                
                # Poda beta
                if beta <= alpha:
                    break
            
            return max_value
        
        else:  # Minimizador (oponente: 1)
            min_value = float('inf')
            for col in available:
                next_state = state.transition(col)
                value = self._minimax(next_state, depth - 1, alpha, beta, True)
                min_value = min(min_value, value)
                beta = min(beta, value)
                
                # Poda alfa
                if beta <= alpha:
                    break
            
            return min_value
    
    def _evaluate_board(self, board: np.ndarray) -> float:
        """
        Evalúa el tablero sin ganador (heurística).
        Valora:
        - Posiciones con 3 en línea (casi ganar)
        - Posiciones centrales (mejor control)
        - Más piezas = mejor
        """
        score = 0.0
        
        # Evaluar todas las líneas posibles de 4
        for row in range(6):
            for col in range(7):
                # Horizontal
                if col + 3 < 7:
                    line = [board[row, col + i] for i in range(4)]
                    score += self._evaluate_line(line)
                
                # Vertical
                if row + 3 < 6:
                    line = [board[row + i, col] for i in range(4)]
                    score += self._evaluate_line(line)
                
                # Diagonal derecha
                if row + 3 < 6 and col + 3 < 7:
                    line = [board[row + i, col + i] for i in range(4)]
                    score += self._evaluate_line(line)
                
                # Diagonal izquierda
                if row + 3 < 6 and col - 3 >= 0:
                    line = [board[row + i, col - i] for i in range(4)]
                    score += self._evaluate_line(line)
        
        # Bonus por posiciones centrales (columna 3 es mejor)
        for row in range(6):
            for col in range(7):
                if board[row, col] == -1:  # Nuestra pieza
                    distance_from_center = abs(col - 3)
                    score += (3 - distance_from_center) * 10
                elif board[row, col] == 1:  # Pieza oponente
                    distance_from_center = abs(col - 3)
                    score -= (3 - distance_from_center) * 10
        
        return score
    
    def _evaluate_line(self, line: list) -> float:
        """
        Evalúa una línea de 4 posiciones.
        
        Puntuación:
        - [1, 1, 1, 1] = +400 (ganar)
        - [1, 1, 1, 0] = +50 (amenaza)
        - [1, 1, 0, 0] = +10 (potencial)
        - [-1, -1, -1, -1] = -400 (ganar para oponente)
        - [-1, -1, -1, 0] = -50 (amenaza oponente)
        - etc.
        """
        count_us = sum(1 for x in line if x == -1)
        count_them = sum(1 for x in line if x == 1)
        count_empty = sum(1 for x in line if x == 0)
        
        # No se puede contar como amenaza si hay oponente en la línea
        if count_us > 0 and count_them > 0:
            return 0
        
        # Evaluar nuestras piezas
        if count_us == 4:
            return 400
        elif count_us == 3 and count_empty == 1:
            return 50
        elif count_us == 2 and count_empty == 2:
            return 10
        elif count_us == 1 and count_empty == 3:
            return 1
        
        # Evaluar piezas del oponente (defensivamente)
        if count_them == 4:
            return -400
        elif count_them == 3 and count_empty == 1:
            return -50
        elif count_them == 2 and count_empty == 2:
            return -10
        elif count_them == 1 and count_empty == 3:
            return -1
        
        return 0
