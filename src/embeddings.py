"""Paso 3: word embeddings preentrenados (GloVe) + análisis de palabras sin vector (OOV).

Entrada : data/tickets/tokenized/tickets_tokens.jsonl  (salida de preprocess.py)
Salida  : data/tickets/embeddings/ticket_vectors.npy   (un vector por ticket = media de sus palabras)
          data/tickets/embeddings/oov_words.csv        (palabras sin vector y su frecuencia)
"""
import json
from collections import Counter
from pathlib import Path

import gensim.downloader as api
import numpy as np
import pandas as pd

PATH = Path("data") / "tickets"
TOKENS_PATH = PATH / "tokenized" / "tickets_tokens.jsonl"
OUTDIR = PATH / "embeddings"
OUTDIR.mkdir(parents=True, exist_ok=True)

MODELO = "glove-wiki-gigaword-100"  # 400k palabras, 100 dimensiones, en minúsculas

with TOKENS_PATH.open(encoding="utf-8") as f:
    tokens_tickets = [json.loads(linea)["tokens"] for linea in f]

kv = api.load(MODELO)  # la primera vez descarga (~130 MB) y queda en caché

# ---- Análisis de OOV ----
freq = Counter(t for toks in tokens_tickets for t in toks)
oov = {w: c for w, c in freq.items() if w not in kv}

n_tipos, n_oov_tipos = len(freq), len(oov)
n_tokens, n_oov_tokens = sum(freq.values()), sum(oov.values())
print(f"Modelo: {MODELO} ({kv.vector_size} dimensiones, {len(kv)} palabras)")
print(f"Palabras distintas: {n_tipos} | sin vector: {n_oov_tipos} ({n_oov_tipos / n_tipos:.1%})")
print(f"Tokens totales:     {n_tokens} | sin vector: {n_oov_tokens} ({n_oov_tokens / n_tokens:.1%})")

oov_df = (
    pd.DataFrame(oov.items(), columns=["word", "count"])
    .sort_values("count", ascending=False)
    .reset_index(drop=True)
)
oov_df.to_csv(OUTDIR / "oov_words.csv", index=False)
print("\nTop 30 OOV más frecuentes:")
print(oov_df.head(30).to_string(index=False))

# ---- Vector por ticket: media de los vectores de sus palabras (las OOV se ignoran) ----
vectores = np.zeros((len(tokens_tickets), kv.vector_size), dtype=np.float32)
sin_vector = 0
for i, toks in enumerate(tokens_tickets):
    vs = [kv[t] for t in toks if t in kv]
    if vs:
        vectores[i] = np.mean(vs, axis=0)
    else:
        sin_vector += 1  # ticket sin ninguna palabra con vector -> vector de ceros

np.save(OUTDIR / "ticket_vectors.npy", vectores)
print(f"\nVectores de tickets: {vectores.shape} | tickets sin ninguna palabra con vector: {sin_vector}")
