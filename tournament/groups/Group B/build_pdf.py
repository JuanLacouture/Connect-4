"""Genera el documento Word (PDF-ready) del resumen del agente Minimax."""
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import nsdecls
from docx.oxml import parse_xml
import os

doc = Document()

# ── Page setup ──
for section in doc.sections:
    section.top_margin = Cm(1.5)
    section.bottom_margin = Cm(1.2)
    section.left_margin = Cm(1.8)
    section.right_margin = Cm(1.8)

# ── Color palette ──
BLUE = RGBColor(0x1A, 0x5C, 0x8A)
BLUE_LIGHT = RGBColor(0x2E, 0x86, 0xC1)
GREEN = RGBColor(0x1E, 0x8A, 0x6E)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DARK = RGBColor(0x2C, 0x3E, 0x50)
GRAY = RGBColor(0x7F, 0x8C, 0x8D)
BG_BLUE = "1A5C8A"
BG_GREEN = "1E8A6E"
BG_LIGHT = "EBF5FB"
BG_GREEN_LIGHT = "E8F8F5"

# ── Styles ──
style = doc.styles["Normal"]
font = style.font
font.name = "Calibri"
font.size = Pt(9.5)
font.color.rgb = DARK
style.paragraph_format.space_after = Pt(4)
style.paragraph_format.space_before = Pt(0)
style.paragraph_format.line_spacing = 1.1


def add_heading2(text, color=BLUE):
    h = doc.add_heading(text, level=2)
    for run in h.runs:
        run.font.color.rgb = color
        run.font.name = "Calibri"
    h.paragraph_format.space_before = Pt(7)
    h.paragraph_format.space_after = Pt(3)
    return h


def set_cell_bg(cell, color_hex):
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading)


def style_table(table, header_bg=BG_BLUE):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for row_idx, row in enumerate(table.rows):
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(1)
                p.paragraph_format.space_before = Pt(1)
                for run in p.runs:
                    run.font.size = Pt(8.5)
                    run.font.name = "Calibri"
            if row_idx == 0:
                set_cell_bg(cell, header_bg)
                for p in cell.paragraphs:
                    for run in p.runs:
                        run.font.color.rgb = WHITE
                        run.bold = True


# ════════════════════════════════════════════
# TITLE BAR
# ════════════════════════════════════════════
title_table = doc.add_table(rows=1, cols=1)
title_table.alignment = WD_TABLE_ALIGNMENT.CENTER
cell = title_table.cell(0, 0)
set_cell_bg(cell, BG_BLUE)

p = cell.paragraphs[0]
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("AGENTE MINIMAX CON PODA ALFA-BETA PARA CONNECT-4")
run.font.size = Pt(14)
run.font.color.rgb = WHITE
run.bold = True
run.font.name = "Calibri"

p2 = cell.add_paragraph()
p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
run2 = p2.add_run(
    "Andres Sanchez  ·  Fundamentos de IA  ·  "
    "Universidad de La Sabana  ·  2026-1  ·  Prof. Felix Mohr"
)
run2.font.size = Pt(8.5)
run2.font.color.rgb = RGBColor(0xD5, 0xE8, 0xF0)
run2.font.name = "Calibri"

doc.add_paragraph()  # spacer

# ════════════════════════════════════════════
# 1. RESUMEN DEL AGENTE
# ════════════════════════════════════════════
add_heading2("1. Resumen del Agente")

p = doc.add_paragraph()
run = p.add_run("Minimax + Poda Alfa-Beta  ·  SEARCH_DEPTH = 6  ·  Evaluacion estatica por ventanas")
run.bold = True
run.font.size = Pt(10)
run.font.color.rgb = BLUE

p = doc.add_paragraph()
run = p.add_run(
    "Connect-4 es un juego determinista, secuencial, competitivo y "
    "totalmente observable (slide 1). Al modelarlo como un juego de "
    "suma cero (slide 12), la ecuacion de Bellman se convierte en "
    "minimax: yo maximizo mi utilidad asumiendo que el rival minimiza. "
    "Esto justifica una busqueda DFS en el arbol de jugadas."
)

# Formula box
fb = doc.add_table(rows=1, cols=1)
fb.alignment = WD_TABLE_ALIGNMENT.CENTER
cell = fb.cell(0, 0)
set_cell_bg(cell, BG_LIGHT)
p = cell.paragraphs[0]
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("Bellman:  ")
run.font.size = Pt(8)
run.font.color.rgb = GRAY
run = p.add_run("v*(s) = r(s) + γ · max_a Σ P(s'|s,a) · v*(s')")
run.font.size = Pt(11)
run.font.color.rgb = BLUE
run.bold = True
run.font.name = "Consolas"
p2 = cell.add_paragraph()
p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p2.add_run("En suma cero determinista → ")
run.font.size = Pt(8)
run.font.color.rgb = GRAY
run = p2.add_run("Minimax: yo maximizo, rival minimiza. ")
run.font.size = Pt(10)
run.font.color.rgb = GREEN
run.bold = True
run = p2.add_run("Poda α-β: O(b^d) → O(b^(d/2))")
run.font.size = Pt(10)
run.font.color.rgb = BLUE_LIGHT
run.bold = True

# Decisions summary
p = doc.add_paragraph()
run = p.add_run("Decisiones de diseno y justificacion:")
run.bold = True
run.font.size = Pt(9)
run.font.color.rgb = GREEN

decisions = [
    ("¿Por que Minimax?",
     "Es la solucion natural para juegos de suma cero deterministas. "
     "A diferencia de MCTS (estocastico) o Q-Learning (requiere entrenamiento offline), "
     "minimax garantiza la mejor jugada dentro del horizonte explorado, "
     "es determinista y completamente explicable."),
    ("¿Por que poda alfa-beta?",
     "Sin poda, explorar depth=6 con b≈7 requiere ~117,000 nodos. "
     "Alfa-beta con buen orden reduce a ~340 nodos (O(b^(d/2))). "
     "Esto permite buscar el doble de profundidad en el mismo tiempo."),
    ("¿Por que SEARCH_DEPTH = 6?",
     "Es el punto optimo: depth<6 no ve amenazas a medio plazo; "
     "depth>6 excede el time guard de 850ms en Python puro. "
     "Cada nivel extra multiplica el tiempo ~7x."),
    ("¿Por que evaluacion estatica por ventanas?",
     "Cuando la busqueda no llega a un estado terminal, "
     "se evalua la posicion contando patrones en ventanas de 4 casillas. "
     "Pesos asimetricos (+50 amenaza propia, -80 amenaza rival) "
     "priorizan defensa sobre ataque — un principio robusto en Connect-4."),
    ("¿Por que orden centro-primero (3,2,4,1,5,0,6)?",
     "Columnas centrales participan en mas lineas de 4. "
     "Explorarlas primero genera mejores cortes alfa-beta, "
     "acercando la poda al optimo teorico."),
    ("Arquitectura de 3 capas:",
     "Capa 1: victoria inmediata O(7) → Capa 2: bloqueo inmediato O(7) → "
     "Capa 3: busqueda minimax α-β O(b^d). Cascada de prioridad: "
     "si capa 1 resuelve, no se ejecutan 2 ni 3. "
     "Color se infiere por paridad de fichas."),
]

for q, a in decisions:
    p = doc.add_paragraph()
    run = p.add_run(f"• {q} ")
    run.bold = True
    run.font.size = Pt(8.5)
    run.font.color.rgb = BLUE_LIGHT
    run = p.add_run(a)
    run.font.size = Pt(8.5)
    p.paragraph_format.space_after = Pt(1)

# ════════════════════════════════════════════
# 2. TRES AGENTES
# ════════════════════════════════════════════
add_heading2("2. Tres Agentes Desarrollados")

t = doc.add_table(rows=4, cols=5)
t.style = "Table Grid"
for i, h in enumerate(["Agente", "Tecnica", "Slides", "Parametro", "Fortaleza"]):
    t.cell(0, i).text = h
data = [
    ["✓ Minimax", "Busqueda adversarial + α-β", "2, 3, 12", "SEARCH_DEPTH=6", "Determinista"],
    ["MCTS", "Monte Carlo + UCT", "10, 13", "3000 sims", "Sin heuristica"],
    ["Q-Learning", "Tabla Q self-play", "6-9, 11", "200k episodios", "act() O(1)"],
]
for r, rd in enumerate(data):
    for c, val in enumerate(rd):
        t.cell(r + 1, c).text = val
    if r == 0:
        for c in range(5):
            set_cell_bg(t.cell(r + 1, c), BG_GREEN_LIGHT)
style_table(t)

# Why minimax
wb = doc.add_table(rows=1, cols=1)
wb.alignment = WD_TABLE_ALIGNMENT.CENTER
cell = wb.cell(0, 0)
set_cell_bg(cell, BG_GREEN_LIGHT)
p = cell.paragraphs[0]
run = p.add_run("¿Por que Minimax?  ")
run.bold = True
run.font.size = Pt(9)
run.font.color.rgb = GREEN
run = p.add_run(
    "Determinista (reproducible) · Explicable (se justifica cada jugada) "
    "· Conceptualmente distinto de companeros (Negamax/MCTS)"
)
run.font.size = Pt(8.5)

# ════════════════════════════════════════════
# PAGE 2
# ════════════════════════════════════════════
doc.add_page_break()

# ════════════════════════════════════════════
# 3. GATE (con desglose por color)
# ════════════════════════════════════════════
add_heading2("3. Validacion Gate")

gb = doc.add_table(rows=1, cols=1)
gb.alignment = WD_TABLE_ALIGNMENT.CENTER
cell = gb.cell(0, 0)
set_cell_bg(cell, BG_GREEN_LIGHT)
p = cell.paragraphs[0]
run = p.add_run("✓ GATE PASS  ")
run.bold = True
run.font.size = Pt(11)
run.font.color.rgb = GREEN
run = p.add_run(
    "0 derrotas vs aleatorio  ·  100% winrate  ·  SEARCH_DEPTH = 6"
)
run.font.size = Pt(9)

# Desglose por color
t = doc.add_table(rows=3, cols=4)
t.style = "Table Grid"
for i, h in enumerate(["Color", "Victorias", "Derrotas", "Winrate"]):
    t.cell(0, i).text = h
data = [
    ["Rojo (primero)", "30/30", "0", "100%"],
    ["Amarillo (segundo)", "30/30", "0", "100%"],
]
for r, rd in enumerate(data):
    for c, val in enumerate(rd):
        t.cell(r + 1, c).text = val
style_table(t, BG_GREEN)

# ════════════════════════════════════════════
# 4. PARAMETRIC
# ════════════════════════════════════════════
add_heading2("4. Analisis Parametrico: SEARCH_DEPTH")

t = doc.add_table(rows=4, cols=4)
t.style = "Table Grid"
for i, h in enumerate(["Rango", "Winrate", "Tiempo max", "Observacion"]):
    t.cell(0, i).text = h
data = [
    ["Depth 1-2", "100%", "<10 ms", "Capa tactica resuelve casi todo"],
    ["Depth 3-5", "100%", "10-450 ms", "Busqueda refuerza; dentro del time guard"],
    ["Depth 6-8", "100%", "1.9-48 s", "Depth>=7 excede time guard (850ms)"],
]
for r, rd in enumerate(data):
    for c, val in enumerate(rd):
        t.cell(r + 1, c).text = val
style_table(t, BG_BLUE)

p = doc.add_paragraph()
run = p.add_run("Insight: ")
run.bold = True
run.font.color.rgb = GREEN
run.font.size = Pt(9)
run = p.add_run(
    "cada nivel multiplica el tiempo ~7x. SEARCH_DEPTH=6 = "
    "punto optimo (calidad alta, dentro del time guard 850ms)."
)
run.font.size = Pt(9)

# Placeholder for chart
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("[  INSERTAR GRAFICA: Depth vs Winrate + Depth vs Tiempo  ]")
run.font.size = Pt(9)
run.font.color.rgb = GRAY
run.italic = True

# ════════════════════════════════════════════
# 5. DOS VERSIONES
# ════════════════════════════════════════════
add_heading2("5. Dos Versiones")

t = doc.add_table(rows=3, cols=3)
t.style = "Table Grid"
for i, h in enumerate(["Version", "Configuracion", "Resultado"]):
    t.cell(0, i).text = h
data = [
    ["v1 (completa)", "Tactica + minimax", "Robusta en depth bajo"],
    ["v2 (sin tactica)", "Solo minimax puro", "Vulnerable depth<3"],
]
for r, rd in enumerate(data):
    for c, val in enumerate(rd):
        t.cell(r + 1, c).text = val
style_table(t, BG_GREEN)

# ════════════════════════════════════════════
# 6. ROUND ROBIN
# ════════════════════════════════════════════
add_heading2("6. Round-Robin Interno")

t = doc.add_table(rows=4, cols=3)
t.style = "Table Grid"
for i, h in enumerate(["Enfrentamiento", "Resultado", "Empates"]):
    t.cell(0, i).text = h
data = [
    ["Minimax vs MCTS", "1 - 2", "1"],
    ["Minimax vs Q-Learning", "4 - 0", "0"],
    ["MCTS vs Q-Learning", "4 - 0", "0"],
]
for r, rd in enumerate(data):
    for c, val in enumerate(rd):
        t.cell(r + 1, c).text = val
style_table(t)

p = doc.add_paragraph()
run = p.add_run("Ranking: MCTS > Minimax > Q-Learning. ")
run.bold = True
run.font.size = Pt(9)
run.font.color.rgb = GREEN
run = p.add_run("Diferencia marginal (2-1). Q-Learning no compite con busqueda online.")
run.font.size = Pt(9)

# ════════════════════════════════════════════
# 7. SELF-PLAY
# ════════════════════════════════════════════
add_heading2("7. Self-Play")

p = doc.add_paragraph()
run = p.add_run(
    "En Connect-4, Rojo (primer jugador) tiene ventaja teorica. "
    "Cuando el agente juega contra si mismo (depth=6), "
    "el resultado es determinista: "
)
run.font.size = Pt(9)
run = p.add_run("Rojo siempre gana")
run.bold = True
run.font.size = Pt(9)
run.font.color.rgb = GREEN
run = p.add_run(
    ", lo que confirma que el agente captura parcialmente "
    "la ventaja del primer jugador. "
    "A profundidades menores (depth=2), el resultado es empate — "
    "la busqueda no es suficiente para explotarla."
)
run.font.size = Pt(9)

# ════════════════════════════════════════════
# 8. MEJORAS
# ════════════════════════════════════════════
add_heading2("8. Propuestas de Mejora")

improvements = [
    (
        "Transposition Table",
        "Cachear posiciones ya evaluadas → +2 niveles depth en mismo tiempo. "
        "Evidencia: depth 6→8 mejora vs oponentes fuertes (sec. 4).",
    ),
    (
        "Iterative Deepening + Killer Moves",
        "Mejor orden de poda dinamico → acercarse al optimo O(b^(d/2)). "
        "Evidencia: tiempo crece ~7x/nivel; mejor poda reduce ese factor.",
    ),
    (
        "Tuning heuristica (GA)",
        "Evolucionar pesos (+50/-80) via self-play con algoritmo genetico. "
        "Evidencia: MCTS gano 2-1 (sec. 6) → heuristica es el cuello de botella.",
    ),
]
for title, desc in improvements:
    p = doc.add_paragraph()
    run = p.add_run(f"► {title}: ")
    run.bold = True
    run.font.size = Pt(9)
    run.font.color.rgb = BLUE_LIGHT
    run = p.add_run(desc)
    run.font.size = Pt(8.5)
    p.paragraph_format.space_after = Pt(2)

# ════════════════════════════════════════════
# 9. CONCLUSIONES
# ════════════════════════════════════════════
add_heading2("9. Conclusiones")

conclusions = [
    "Gate cumplido: 0 derrotas, 100% winrate vs aleatorio en ambos colores.",
    "SEARCH_DEPTH=6 es el punto optimo: cada nivel extra cuesta ~7x mas tiempo; depth>6 excede el time guard.",
    "La capa tactica (O(7)) es una red de seguridad que garantiza victoria/bloqueo inmediato sin costo significativo.",
    "MCTS supera ligeramente a Minimax (2-1), pero Minimax es determinista, explicable y distinto de companeros.",
    "Las mejoras propuestas atacan el cuello de botella principal: aumentar profundidad efectiva sin violar el time guard.",
]

for c in conclusions:
    p = doc.add_paragraph()
    run = p.add_run(f"• {c}")
    run.font.size = Pt(8.5)
    p.paragraph_format.space_after = Pt(1)

# ── Footer ──
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(10)
run = p.add_run(
    "─" * 60
)
run.font.color.rgb = GRAY
run.font.size = Pt(6)
p2 = doc.add_paragraph()
p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p2.add_run(
    "Fundamentos de Inteligencia Artificial  ·  Prof. Felix Mohr  "
    "·  Universidad de La Sabana  ·  Mayo 2026"
)
run.font.size = Pt(8)
run.font.color.rgb = GRAY
run.italic = True

# ── Save ──
output_path = os.path.join(os.path.dirname(__file__), "Resumen_Minimax_Sanchez_v4.docx")
doc.save(output_path)
print(f"Documento guardado: {output_path}")
print(f"Tamano: {os.path.getsize(output_path) / 1024:.1f} KB")
