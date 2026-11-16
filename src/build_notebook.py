"""Convierte el código en formato «percent» (# %% / # %% [markdown]) en un notebook .ipynb."""
import re
import sys
from pathlib import Path
import nbformat as nbf

BASE = Path(__file__).resolve().parent
if len(sys.argv) >= 3:
    destino, fuentes = Path(sys.argv[1]), [Path(f) for f in sys.argv[2:]]
else:
    destino = BASE / "Entrega_3_NLP.ipynb"
    fuentes = [
        BASE / "e3_parte1_datos_shallow.py",
        BASE / "e3_parte2_embeddings.py",
        BASE / "e3_parte3_deep_evaluacion.py",
    ]

texto = "\n".join(f.read_text(encoding="utf-8") for f in fuentes)

celdas, tipo, buffer = [], None, []
for linea in texto.splitlines():
    m = re.match(r"^# %%(\s*\[markdown\])?\s*$", linea)
    if m:
        if tipo is not None:
            celdas.append((tipo, buffer))
        tipo, buffer = ("markdown" if m.group(1) else "code"), []
    elif tipo is not None:
        buffer.append(linea)
if tipo is not None:
    celdas.append((tipo, buffer))

nb = nbf.v4.new_notebook()
for tipo, lineas in celdas:
    if tipo == "markdown":
        lineas = [re.sub(r"^# ?", "", l) for l in lineas]
    cuerpo = "\n".join(lineas).strip("\n")
    if not cuerpo.strip():
        continue
    nb.cells.append(nbf.v4.new_markdown_cell(cuerpo) if tipo == "markdown" else nbf.v4.new_code_cell(cuerpo))

nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}
nbf.write(nb, destino)
print(f"{destino}: {sum(c.cell_type == 'markdown' for c in nb.cells)} celdas de texto, "
      f"{sum(c.cell_type == 'code' for c in nb.cells)} de código")
