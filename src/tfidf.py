from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from spacy.lang.de.stop_words import STOP_WORDS as STOP_DE
from spacy.lang.en.stop_words import STOP_WORDS as STOP_EN

PATH = Path("data") / "tickets"
DATAPATH = PATH / "dataprocessed" / "dataprocessed.csv"  # salida del script anterior
OUTDIR = PATH / "features"
OUTDIR.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(DATAPATH)

# Un string por ticket: asunto + cuerpo (NO se usa 'answer': no existe al clasificar)
corpus = (
    (data["subject"].fillna("") + " " + data["body"].fillna(""))
    .str.strip()
    .to_numpy()
)

# El dataset solo trae 'train', así que el split lo hacemos aquí
idx = np.arange(len(data))
idx_train, idx_test = train_test_split(
    idx,
    test_size=0.2,
    random_state=42,
    stratify=data["queue"],  # quita esta línea si alguna cola tiene muy pocos tickets
)

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words=sorted(STOP_EN | STOP_DE),  # dataset en inglés + alemán
    min_df=2,
    max_df=0.9,
    ngram_range=(1, 2),
)

# Ajustar SOLO con train para evitar fuga de información
X_train = vectorizer.fit_transform(corpus[idx_train])
X_test = vectorizer.transform(corpus[idx_test])

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Vocabulario: {len(vectorizer.get_feature_names_out())}")

# Guardar para reutilizar
sparse.save_npz(OUTDIR / "X_train.npz", X_train)
sparse.save_npz(OUTDIR / "X_test.npz", X_test)
np.save(OUTDIR / "idx_train.npy", idx_train)  # para recuperar queue/priority/tags
np.save(OUTDIR / "idx_test.npy", idx_test)
joblib.dump(vectorizer, OUTDIR / "tfidf_vectorizer.joblib")




# TOP PALABRAS
TOP_N = 20
feature_names = vectorizer.get_feature_names_out()

# 1) Top global: peso TF-IDF medio por término en train
mean_tfidf = np.asarray(X_train.mean(axis=0)).ravel()
top_global = np.argsort(mean_tfidf)[::-1][:TOP_N]

print(f"\nTop {TOP_N} palabras (TF-IDF medio, global):")
for i in top_global:
    print(f"  {feature_names[i]:<30} {mean_tfidf[i]:.4f}")

# 2) Top por cola (queue): TF-IDF medio solo con los tickets de esa cola
y_train = data["queue"].iloc[idx_train].to_numpy()
top_por_cola = {}

for cola in np.unique(y_train):
    mask = y_train == cola
    media_cola = np.asarray(X_train[mask].mean(axis=0)).ravel()
    top_idx = np.argsort(media_cola)[::-1][:TOP_N]
    top_por_cola[cola] = [(feature_names[i], media_cola[i]) for i in top_idx]

    print(f"\nTop {TOP_N} palabras para la cola '{cola}':")
    for palabra, peso in top_por_cola[cola]:
        print(f"  {palabra:<30} {peso:.4f}")
