"""
POLÍTICA HEURÍSTICA INTELIGENTE PARA GROUP C
Combina:
- Búsqueda de movimientos ganadores
- Bloqueo de amenazas del oponente
- Estrategia ofensiva/defensiva por puntaje
- Preferencia por posiciones centrales
"""

import numpy as np
from connect4.policy import Policy
from connect4.connect_state import ConnectState
from typing import override


class SmartHeuristicPolicy(Policy):
    """
    Política que usa heurísticas inteligentes sin búsqueda profunda.
    Más rápida que Minimax pero más inteligente que Random.
    """
    
    def __init__(self):
        self.our_player = -1
        self.opponent_player = 1
    
    @override
    def mount(self) -> None:
        """Llamado antes de cada partida."""
        pass
    
    @override
    def act(self, board: np.ndarray) -> int:
        """
        Decide el movimiento usando prioridades heurísticas.
        """
        state = ConnectState(board, self.our_player)
        available = state.get_free_cols()
        
        if not available:
            return available[0]
        
        # PRIORIDAD 1: ¿Podemos ganar en este turno?
        winning_move = self._find_winning_move(state, available)
        if winning_move is not None:
            return winning_move
        
        # PRIORIDAD 2: ¿El oponente puede ganar próximo turno? Bloquear
        blocking_move = self._find_blocking_move(state, available)
        if blocking_move is not None:
            return blocking_move
        
        # PRIORIDAD 3: Crear amenazas (dos en línea sin bloqueo)
        threatening_move = self._find_threatening_move(state, available)
        if threatening_move is not None:
            return threatening_move
        
        # PRIORIDAD 4: Defender de amenazas del oponente
        defensive_move = self._find_defensive_move(state, available)
        if defensive_move is not None:
            return defensive_move
        
        # PRIORIDAD 5: Expandir control central
        central_move = self._find_central_move(available)
        if central_move is not None:
            return central_move
        
        # FALLBACK: Cualquier movimiento disponible
        return available[0]
    
    def _find_winning_move(self, state: ConnectState, available: list) -> int | None:
        """Busca un movimiento que nos haga ganar inmediatamente."""
        for col in available:
            next_state = state.transition(col)
            if next_state.get_winner() == self.our_player:
                return col
        return None
    
    def _find_blocking_move(self, state: ConnectState, available: list) -> int | None:
        """Bloquea un movimiento ganador del oponente."""
        opponent_state = ConnectState(state.board, self.opponent_player)
        
        for col in available:
            next_state = opponent_state.transition(col)
            if next_state.get_winner() == self.opponent_player:
                return col
        
        return None
    
    def _find_threatening_move(self, state: ConnectState, available: list) -> int | None:
        """
        Busca crear una amenaza (3 en línea con espacio).
        Esto fuerza al oponente a bloquear.
        """
        best_threat = None
        best_threat_score = 0
        
        for col in available:
            next_state = state.transition(col)
            threat_score = self._count_threats(next_state.board, self.our_player)
            
            if threat_score > best_threat_score:
                best_threat_score = threat_score
                best_threat = col
        
        return best_threat if best_threat_score > 0 else None
    
    def _find_defensive_move(self, state: ConnectState, available: list) -> int | None:
        """
        Previene que el oponente cree amenazas.
        """
        best_defense = None
        best_defense_score = float('inf')
        
        for col in available:
            opponent_state = ConnectState(state.board, self.opponent_player)
            next_opponent_state = opponent_state.transition(col)
            threat_score = self._count_threats(next_opponent_state.board, self.opponent_player)
            
            if threat_score < best_defense_score:
                best_defense_score = threat_score
                best_defense = col
        
        return best_defense if best_defense_score < float('inf') else None
    
    def _find_central_move(self, available: list) -> int | None:
        """Prefiere movimientos en columnas centrales (mejor estrategia)."""
        # Ordenar por proximidad al centro (columna 3)
        central_preference = [3, 2, 4, 1, 5, 0, 6]
        
        for col in central_preference:
            if col in available:
                return col
        
        return available[0] if available else None
    
    def _count_threats(self, board: np.ndarray, player: int) -> int:
        """
        Cuenta cuántas amenazas (3 en línea) tiene un jugador.
        Más amenazas = más peligroso.
        """
        threat_count = 0
        
        for row in range(6):
            for col in range(7):
                # Horizontal
                if col + 3 < 7:
                    line = [board[row, col + i] for i in range(4)]
                    if self._is_threat(line, player):
                        threat_count += 1
                
                # Vertical
                if row + 3 < 6:
                    line = [board[row + i, col] for i in range(4)]
                    if self._is_threat(line, player):
                        threat_count += 1
                
                # Diagonal derecha
                if row + 3 < 6 and col + 3 < 7:
                    line = [board[row + i, col + i] for i in range(4)]
                    if self._is_threat(line, player):
                        threat_count += 1
                
                # Diagonal izquierda
                if row + 3 < 6 and col - 3 >= 0:
                    line = [board[row + i, col - i] for i in range(4)]
                    if self._is_threat(line, player):
                        threat_count += 1
        
        return threat_count
    
    def _is_threat(self, line: list, player: int) -> bool:
        """
        Una línea es amenaza si tiene:
        - 3 piezas del jugador y 1 vacío
        - Y el vacío es accesible
        """
        count_player = sum(1 for x in line if x == player)
        count_opponent = sum(1 for x in line if x == -player)
        count_empty = sum(1 for x in line if x == 0)
        
        return count_player == 3 and count_opponent == 0 and count_empty == 1
    
    def _evaluate_board(self, board: np.ndarray) -> float:
        """
        Evaluación simple del tablero para comparaciones.
        Retorna un score de cuán favorable es el tablero.
        """
        score = 0.0
        
        # Contar nuestras piezas vs piezas del oponente con bonus por posición
        for row in range(6):
            for col in range(7):
                if board[row, col] == self.our_player:
                    # Bonus por estar en el centro
                    distance = abs(col - 3)
                    score += 10 - distance
                elif board[row, col] == self.opponent_player:
                    # Penalidad por oponente en el centro
                    distance = abs(col - 3)
                    score -= (10 - distance)
        
        return score
