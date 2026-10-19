"""Paso 1: tokenización y división en oraciones.

Entrada : data/tickets/dataprocessed/dataprocessed.csv  (salida de load_data.py)
Salida  : data/tickets/tokenized/tickets_tokens.jsonl   (un ticket por línea)
"""
import json
import re
from pathlib import Path

import pandas as pd
import spacy

PATH = Path("data") / "tickets"
IN_PATH = PATH / "dataprocessed" / "dataprocessed.csv"
OUT_PATH = PATH / "tokenized" / "tickets_tokens.jsonl"

# Pipeline vacío de inglés (solo tokenizador) + detector de oraciones por reglas.
# No necesita descargar ningún modelo; todo el dataset está en inglés (alemán eliminado).
nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")


def limpiar_texto(texto: str) -> str:
    """Normaliza el texto crudo antes de tokenizar."""
    # En el CSV los saltos de línea vienen como el texto literal '\n' (barra + n)
    texto = texto.replace("\\n", "\n")
    # Espacios y saltos repetidos -> uno solo (los saltos separan párrafos, los dejamos como espacio)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def procesar(doc) -> dict:
    """De un Doc de spaCy saca oraciones y tokens (minúsculas, sin puntuación/espacios)."""
    oraciones = [s.text.strip() for s in doc.sents if s.text.strip()]
    tokens = [t.lower_ for t in doc if not (t.is_punct or t.is_space)]
    return {"sentences": oraciones, "tokens": tokens}


def main() -> None:
    df = pd.read_csv(IN_PATH)
    # Un solo texto por ticket: asunto + cuerpo (NO 'answer', no existe al clasificar)
    textos = (df["subject"].fillna("") + ". " + df["body"].fillna("")).map(limpiar_texto)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for i, doc in enumerate(nlp.pipe(textos, batch_size=500)):
            reg = {"id": i, "text": textos.iloc[i], **procesar(doc)}
            f.write(json.dumps(reg, ensure_ascii=False) + "\n")

    print(f"{len(df)} tickets tokenizados -> {OUT_PATH}")


if __name__ == "__main__":
    main()
