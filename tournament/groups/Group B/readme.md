# Agente Minimax con Poda Alfa-Beta — Group B

**Autor:** Andrés Sánchez  
**Curso:** Fundamentos de Inteligencia Artificial — Universidad de La Sabana (2026-1)  
**Profesor:** Felix Mohr

## Descripción

Agente Connect-4 basado en **Minimax con poda alfa-beta** y evaluación heurística.
Arquitectura de 3 capas de prioridad descendente:

1. **Victoria inmediata** — detecta si alguna columna completa 4 en línea propias (O(7)).
2. **Bloqueo inmediato** — detecta si el rival gana en su próximo turno y lo bloquea (O(7)).
3. **Búsqueda Minimax** — explora el árbol de juego con poda α-β hasta `SEARCH_DEPTH` niveles.

## Parámetros principales

| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| `SEARCH_DEPTH` | 6 | Profundidad máxima del árbol de búsqueda |
| `_CENTER_ORDER` | `(3,2,4,1,5,0,6)` | Orden de exploración de columnas (centro primero) |
| `_TIME_GUARD` | 0.85 s | Tiempo máximo por jugada (red de seguridad) |

## Archivos

| Archivo | Descripción |
|---|---|
| `policy.py` | Código del agente (`MinimaxPolicy`) |
| `entrega.ipynb` | Notebook con estudio completo: gate, análisis paramétrico, self-play, propuestas de mejora |
| `depth_sweep_data.npz` | Datos precalculados del barrido de profundidad (depths 1-8) |

## Ejecución

```bash
# Desde la raíz del repositorio
cd tournament
python main.py
```

El agente se auto-descubre en `tournament/groups/Group B/policy.py`.

## Gate

- **0 derrotas** contra jugador aleatorio (100 partidas por color).
- **Winrate ≥ 50%** en ambos colores (Rojo y Amarillo).
- Cumple para cualquier `SEARCH_DEPTH ≥ 1`.

## Diferenciación

Este agente usa **búsqueda constructiva con heurística de evaluación explícita** (slides 2, 3, 12).
Se diferencia de los otros agentes del grupo:
- **Group A:** Negamax (variante de Minimax sin heurística de ventanas)
- **Group C:** Q-Learning con self-play offline (200k episodios, tabla Q)

## Enlace al código

- **Branch:** [`Andres-Sanchez`](https://github.com/JuanLacouture/Connect-4/tree/Andres-Sanchez/tournament/groups/Group%20B)
- **Agente final:** [`policy.py`](https://github.com/JuanLacouture/Connect-4/blob/Andres-Sanchez/tournament/groups/Group%20B/policy.py)
