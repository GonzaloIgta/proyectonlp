# %% [markdown]
# # E3 · Primera iteración de resolución del problema
# ## Gestión de Tickets: clasificación automática de tickets de soporte
#
# **Fundamentos del Procesamiento de Lenguaje Natural** · David Trueba y Gonzalo Iglesias
#
# En E1 planteamos un sistema que, a partir del asunto y el cuerpo de un ticket, prediga a qué
# departamento va (**cola**), su **prioridad** y su **tipo**. En E2 dejamos preparado el texto y
# las representaciones en `data/tickets/` (`src/tickets_pipeline.ipynb`). En esta entrega entrenamos
# los primeros modelos y los comparamos entre sí.
#
# | Bloque | Qué hacemos | Requisito del enunciado |
# |---|---|---|
# | 1 | Carga, limpieza y **detección de duplicados** | Dataset de E2 |
# | 2 | Preprocesado de E2 y partición train / validación / test | Features de E2 |
# | 3 | Métricas y registro de resultados | Accuracy, precision, recall y F1 |
# | 4 | **Shallow learning** con TF-IDF: Naive Bayes, Regresión logística y SVM | 2 técnicas con representación dispersa |
# | 5 | **Deep learning**: CNN y BiLSTM con atención, comparando embeddings | 2 técnicas comparando embeddings y fine-tuning |
# | 6 | Prioridad y tipo con las mejores configuraciones | Las otras dos salidas de E1 |
# | 7 | Tabla comparativa final y análisis de errores | Tabla comparativa |
#
# Extras que el enunciado valora: **atención** (5.7) y **combinar features no textuales** (4.4 y 5.8).
#
# > El notebook se puede ejecutar tanto desde `src/` como desde la raíz del proyecto y tarda unos
# > 45 minutos con GPU. Con la variable de entorno `E3_FAST=1` hace una pasada rápida de prueba con
# > menos datos.

# %% [markdown]
# ## 0. Configuración

# %%
import os, re, json, time, random, hashlib, warnings, urllib.request
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML

from scipy import sparse
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, f1_score,
                             recall_score, confusion_matrix, classification_report)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 90)

SEED = 42
FAST = os.environ.get("E3_FAST", "0") == "1"       # pasada rápida de prueba
SEEDS = [42] if FAST else [42, 43, 44]             # semillas para los modelos deep
MAX_EPOCHS = 2 if FAST else 40
N_JOBS = max(1, min(8, (os.cpu_count() or 2) - 2))

ROOT = Path.cwd().parent if Path.cwd().name == "src" else Path.cwd()
DATA_DIR = ROOT / "data" / "tickets"
RAW_CSV = DATA_DIR / "raw" / "dataraw.csv"
ART_DIR = ROOT / "artifacts"
FIG_DIR = ART_DIR / "figuras"
for carpeta in (RAW_CSV.parent, ART_DIR, FIG_DIR):
    carpeta.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo:", DEVICE, "|", torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)

# %%
# Estilo común de las gráficas: superficie clara, rejilla fina y pocos colores con un papel fijo.
COL = {"fondo": "#fcfcfb", "tinta": "#0b0b0b", "tinta2": "#52514e", "tenue": "#898781",
       "rejilla": "#e1e0d9", "eje": "#c3c2b7", "gris": "#c3c2b7",
       "azul": "#2a78d6", "naranja": "#eb6834", "aqua": "#1baf7a", "amarillo": "#eda100"}
SERIE = [COL["azul"], COL["naranja"], COL["aqua"], COL["amarillo"]]

plt.rcParams.update({
    "figure.facecolor": COL["fondo"], "axes.facecolor": COL["fondo"], "savefig.facecolor": COL["fondo"],
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"], "font.size": 10,
    "text.color": COL["tinta"], "axes.labelcolor": COL["tinta2"],
    "xtick.color": COL["tenue"], "ytick.color": COL["tinta2"], "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.edgecolor": COL["eje"], "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": COL["rejilla"], "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.axisbelow": True, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "legend.fontsize": 9, "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def guardar(fig, nombre):
    """Guarda la figura en artifacts/figuras (la reutiliza el informe en PDF) y la muestra."""
    fig.savefig(FIG_DIR / f"{nombre}.png")
    plt.show()
    plt.close(fig)


def barras_h(ax, etiquetas, valores, color=None, fmt="{:.3f}", err=None, alto=0.55):
    """Barras horizontales finas con el valor escrito en la punta."""
    y = np.arange(len(etiquetas))
    ax.barh(y, valores, height=alto, color=COL["azul"] if color is None else color, xerr=err,
            error_kw={"ecolor": COL["tinta2"], "elinewidth": 1, "capsize": 2})
    ax.set_yticks(y, etiquetas)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.tick_params(length=0)
    tope = max(v + (err[i] if err is not None else 0) for i, v in enumerate(valores))
    for i, v in enumerate(valores):
        extra = (err[i] if err is not None else 0) + tope * 0.012
        ax.text(v + extra, i, fmt.format(v), va="center", ha="left", fontsize=9, color=COL["tinta"])
    ax.set_xlim(0, tope * 1.14)


# %% [markdown]
# ## 1. Carga y limpieza de los datos
#
# Partimos directamente del dataset guardado en E2 (`data/tickets/raw/dataraw.csv`), que
# concatena los tres CSV de
# [`Tobi-Bueck/customer-support-tickets`](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets).
# Si no existiera en disco, se descargan de HuggingFace. Aplicamos los dos filtros de E2: quitar
# los tickets en alemán y los que tienen el cuerpo vacío (obteniendo los mismos 28.260 tickets
# que `data/tickets/dataprocessed/dataprocessed.csv`).

# %%
HF_URL = "https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets/resolve/main/"
FICHEROS = ["aa_dataset-tickets-multi-lang-5-2-50-version.csv",
            "dataset-tickets-german_normalized_50_5_2.csv",
            "dataset-tickets-multi-lang-4-20k.csv"]

if RAW_CSV.exists():
    raw = pd.read_csv(RAW_CSV)
else:
    partes = []
    for f in FICHEROS:
        ruta = RAW_CSV.parent / f
        if not ruta.exists():
            print("Descargando", f)
            urllib.request.urlretrieve(HF_URL + f, ruta)
        partes.append(pd.read_csv(ruta))
    raw = pd.concat(partes, ignore_index=True)
    raw.to_csv(RAW_CSV, index=False)

print(f"Tickets en bruto: {len(raw):,}")

df_e2 = raw[raw["language"] != "de"].copy()
df_e2 = df_e2[df_e2["body"].fillna("").str.strip() != ""].reset_index(drop=True)
# De los dos CSV con tickets en inglés, solo el 5-2-50 trae la columna `version` informada.
df_e2["fichero"] = np.where(df_e2["version"].notna(), FICHEROS[0], FICHEROS[2])
print(f"Tras quitar alemán y cuerpos vacíos (filtros de E2): {len(df_e2):,}")

# %% [markdown]
# ### 1.1 Duplicados: una fuga de información que no vimos en E2
#
# Al cruzar los CSV aparece un problema: dos de los ficheros son versiones distintas del mismo
# dataset y **comparten tickets idénticos**. Si un ticket cae en train y su gemelo en test, el
# modelo no tiene que generalizar: le basta con haberlo memorizado. Eso es una fuga de información
# (*data leakage*) y hace que las métricas salgan mejores de lo que son en realidad.

# %%
def arreglar_saltos(texto):
    """En el dataset los saltos de línea vienen escritos como los dos caracteres '\\' y 'n'."""
    return texto.replace("\\n", "\n").replace("’", "'")


def clave_texto(texto):
    """Texto reducido a letras y números en minúscula: sirve para detectar tickets repetidos."""
    return re.sub(r"[^a-z0-9]+", " ", texto.lower()).strip()


df_e2["texto"] = [arreglar_saltos(f"{s}\n{b}").strip()
                  for s, b in zip(df_e2["subject"].fillna(""), df_e2["body"])]
df_e2["clave"] = df_e2["texto"].map(clave_texto)

repetidos = df_e2[df_e2.duplicated("clave", keep=False)]
grupos = repetidos.groupby("clave")
print(f"Tickets con al menos un gemelo exacto: {len(repetidos):,} ({len(repetidos) / len(df_e2):.1%} del total)")
print(f"Grupos de duplicados: {grupos.ngroups:,} | aparecen en ficheros distintos: {(grupos['fichero'].nunique() > 1).mean():.0%}")
for col in ("queue", "priority", "type"):
    print(f"  grupos con '{col}' distinta entre gemelos: {(grupos[col].nunique() > 1).sum()}")

df = df_e2.drop_duplicates("clave").reset_index(drop=True)
print(f"\nTickets únicos con los que trabajamos: {len(df):,}")

# %% [markdown]
# Los gemelos tienen siempre las mismas etiquetas, así que no aportan información nueva: son el
# mismo ticket dos veces. ¿Cuánto inflaban las métricas? Lo medimos con un clasificador rápido
# (TF-IDF + SVM lineal) sobre la cola, con la partición 80/20 de E2 y con la partición limpia.

# %%
def medir_fuga(datos, nombre):
    tr, te = train_test_split(datos, test_size=0.2, stratify=datos["queue"], random_state=SEED)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    clf = LinearSVC().fit(vec.fit_transform(tr["texto"]), tr["queue"])
    pred = clf.predict(vec.transform(te["texto"]))
    return {"partición": nombre, "tickets": len(datos),
            "test con gemelo en train": f"{te['clave'].isin(set(tr['clave'])).mean():.1%}",
            "accuracy": accuracy_score(te["queue"], pred),
            "f1 macro": f1_score(te["queue"], pred, average="macro")}


FUGA = pd.DataFrame([medir_fuga(df_e2, "E2 (con duplicados)"), medir_fuga(df, "E3 (sin duplicados)")])
display(FUGA.round(3))

# %% [markdown]
# Con duplicados, aproximadamente uno de cada cuatro tickets de test ya estaba en train y la
# accuracy sube unos 9 puntos sin que el modelo sea mejor. **A partir de aquí trabajamos solo con
# tickets únicos**, y por eso no reutilizamos directamente las matrices ni los splits guardados en
# `data/tickets/features/` en E2, sino que recalculamos representaciones y particiones sin fuga.

# %% [markdown]
# ### 1.2 Las tres etiquetas
#
# El objetivo principal es la **cola** (10 departamentos). La **prioridad** (3 niveles) y el
# **tipo** (4 clases) se tratan en el bloque 6.

# %%
TAREAS = {"queue": sorted(df["queue"].unique()),
          "priority": ["low", "medium", "high"],
          "type": sorted(df["type"].unique())}
NOMBRE_TAREA = {"queue": "Cola", "priority": "Prioridad", "type": "Tipo"}
for tarea, clases in TAREAS.items():
    df[f"y_{tarea}"] = df[tarea].map({c: i for i, c in enumerate(clases)}).astype(int)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1.5, 1, 1]})
for ax, tarea in zip(axes, TAREAS):
    conteo = df[tarea].value_counts()
    barras_h(ax, conteo.index.tolist(), conteo.values, fmt="{:,.0f}")
    ax.set_title(f"{NOMBRE_TAREA[tarea]} ({len(conteo)} clases)")
    ax.set_xlabel("tickets")
fig.tight_layout()
guardar(fig, "01_distribucion_clases")

# %% [markdown]
# La cola está muy desbalanceada: *Technical Support* tiene unas 20 veces más tickets que
# *General Inquiry*. Por eso la métrica de referencia será el **F1 macro**, que da el mismo peso a
# cada clase, y no la accuracy, que se puede subir acertando solo las clases grandes.

# %% [markdown]
# ## 2. Preprocesado (el de E2) y partición
#
# Reutilizamos el esquema de preprocesado de E2: saltos de línea literales convertidos en saltos
# reales, división en oraciones sobre el texto original, minúsculas y tokenización con una expresión
# regular que mantiene como un único token los marcadores `<...>`, las palabras con apóstrofo
# (`don't`) y cada signo de puntuación. Aquí conservamos la puntuación en `tokens` (para las redes
# neuronales y las meta-features de forma, filtrándola después al construir TF-IDF) y no se lematiza
# ni se quitan stopwords en este paso.

# %%
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
_TOKEN = re.compile(r"<[a-z_]+>|[^\W_]+(?:'[^\W_]+)*|[^\w\s]|_")


def procesar(texto):
    """Texto -> lista de oraciones, cada una como lista de tokens en minúscula."""
    oraciones = [_TOKEN.findall(o.lower()) for o in _SENT.split(texto) if o.strip()]
    return [o for o in oraciones if o]


df["oraciones"] = df["texto"].map(procesar)
df["tokens"] = df["oraciones"].map(lambda ors: [t for o in ors for t in o])
# La respuesta del agente NO es una feature (no existe cuando llega el ticket). Solo se usa,
# como en E2, de texto extra para entrenar el Word2Vec propio.
df["oraciones_resp"] = df["answer"].fillna("").map(lambda s: procesar(arreglar_saltos(s)))
df = df[df["tokens"].str.len() > 0].reset_index(drop=True)

print("Ejemplo:", df["tokens"].iloc[0][:28])
print(df["tokens"].str.len().describe(percentiles=[0.5, 0.95, 0.99]).round(1).to_string())

# %% [markdown]
# ### 2.1 Partición train / validación / test
#
# En E2 solo había train y test (80/20). Ahora hace falta un tercer conjunto, **validación**, porque
# vamos a elegir hiperparámetros y a decidir cuándo parar de entrenar las redes. Si esas decisiones
# se tomaran mirando test, test dejaría de ser una estimación honesta.
#
# - **train (70 %)**: ajusta los modelos, el vocabulario, el TF-IDF y el Word2Vec propio.
# - **validación (15 %)**: elige hiperparámetros y hace el *early stopping*.
# - **test (15 %)**: se usa una sola vez, para la tabla final.
#
# La partición es estratificada por cola, como en E2, y todos los modelos ven exactamente los
# mismos tickets en cada conjunto, así que la comparación es justa.

# %%
idx_train, idx_resto = train_test_split(df.index, test_size=0.30, stratify=df["queue"], random_state=SEED)
idx_val, idx_test = train_test_split(idx_resto, test_size=0.50, stratify=df.loc[idx_resto, "queue"],
                                     random_state=SEED)
df["split"] = "train"
df.loc[idx_val, "split"] = "val"
df.loc[idx_test, "split"] = "test"

if FAST:  # pasada rápida: submuestra de cada conjunto
    df = (df.groupby("split", group_keys=False).sample(frac=0.2, random_state=SEED)
            .reset_index(drop=True))

SPLITS = ("train", "val", "test")
IDX = {s: np.flatnonzero((df["split"] == s).to_numpy()) for s in SPLITS}
Y = {t: {s: df[f"y_{t}"].to_numpy()[IDX[s]] for s in SPLITS} for t in TAREAS}
print({s: len(i) for s, i in IDX.items()})

# %% [markdown]
# ## 3. Métricas y registro de resultados
#
# Para cada modelo guardamos las cuatro métricas del enunciado. Como el problema es multiclase,
# precision, recall y F1 se calculan por clase y luego se promedian:
#
# - **macro**: media simple entre clases. Una clase pequeña cuenta igual que una grande. Es la
#   que usamos para precision, recall y F1, y la que decide qué modelo es mejor.
# - **ponderada (weighted)**: media ponderada por el número de tickets de cada clase. La damos
#   como referencia para el F1.
#
# En la prioridad añadimos el **recall de la clase `high`**, que en E1 definimos como la crítica:
# un ticket urgente clasificado como no urgente es el error más caro.

# %%
RESULTADOS = []      # una fila por modelo y tarea
PREDICCIONES = {}    # (tarea, modelo) -> predicción sobre test (para matrices de confusión)


def metricas(y_true, y_pred, tarea):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    m = {"accuracy": accuracy_score(y_true, y_pred), "precision": p, "recall": r, "f1": f1,
         "f1_ponderado": f1_score(y_true, y_pred, average="weighted")}
    if tarea == "priority":
        alta = TAREAS["priority"].index("high")
        m["recall_high"] = recall_score(y_true, y_pred, labels=[alta], average="macro", zero_division=0)
    return m


def registrar(tarea, familia, modelo, preds_test, f1_val, detalle=""):
    """Guarda las métricas de test. `preds_test` es una lista con una predicción por semilla."""
    por_semilla = pd.DataFrame([metricas(Y[tarea]["test"], p, tarea) for p in preds_test])
    fila = {"tarea": tarea, "familia": familia, "modelo": modelo, "detalle": detalle,
            "semillas": len(preds_test), "f1_val": float(np.mean(f1_val)),
            **por_semilla.mean().to_dict(),
            **{f"{k}_std": v for k, v in por_semilla.std(ddof=0).to_dict().items()}}
    RESULTADOS[:] = [r for r in RESULTADOS if not (r["tarea"] == tarea and r["modelo"] == modelo)]
    RESULTADOS.append(fila)
    PREDICCIONES[(tarea, modelo)] = preds_test[int(np.argmax(np.atleast_1d(f1_val)))]
    return fila


COLS_TABLA = ["familia", "modelo", "accuracy", "precision", "recall", "f1", "f1_ponderado"]


def tabla(tarea, extra=()):
    t = pd.DataFrame([r for r in RESULTADOS if r["tarea"] == tarea])
    cols = [c for c in [*COLS_TABLA, *extra] if c in t.columns]
    return t[cols].sort_values("f1", ascending=False).reset_index(drop=True)


# %% [markdown]
# ## 4. Shallow learning con representación dispersa (TF-IDF)
#
# **TF-IDF** convierte cada ticket en un vector tan largo como el vocabulario, donde casi todas
# las posiciones valen cero (por eso se llama representación *dispersa*). El peso de una palabra
# sube si aparece mucho en el ticket (TF) y baja si aparece en casi todos los tickets (IDF).
# No tiene en cuenta el orden de las palabras, salvo lo que capturan los bigramas.
#
# Sobre esos vectores entrenamos tres clasificadores clásicos:
#
# | Modelo | Idea | Por qué lo probamos |
# |---|---|---|
# | **Naive Bayes** | Aplica Bayes suponiendo que las palabras son independientes entre sí dada la clase | Es el baseline clásico de texto: rapidísimo y difícil de sobreajustar |
# | **Regresión logística** | Aprende un peso por palabra y clase, y da probabilidades con softmax | Modelo lineal discriminativo, interpretable |
# | **SVM lineal** | Busca el hiperplano que separa las clases con el mayor margen | Suele ser el mejor modelo lineal en espacios dispersos de muchas dimensiones |
#
# Como referencia mínima añadimos un clasificador trivial que siempre responde la clase más
# frecuente: cualquier modelo útil tiene que superarlo.

# %%
def texto_tfidf(tokens):
    """Tokens ya procesados unidos por espacios, sin signos de puntuación."""
    return " ".join(t for t in tokens if t[0].isalnum() or (len(t) > 1 and t[0] == "<"))


df["texto_tfidf"] = df["tokens"].map(texto_tfidf)
TXT = {s: df["texto_tfidf"].to_numpy()[IDX[s]] for s in SPLITS}

# Cuatro formas de construir el TF-IDF. Como en E2, se ajusta solo con train (ahora sin duplicados).
VEC_CONFIGS = {
    "unigramas": dict(ngram_range=(1, 1), stop_words=None),
    "unigramas, sin stopwords": dict(ngram_range=(1, 1), stop_words=list(ENGLISH_STOP_WORDS)),
    "uni+bigramas": dict(ngram_range=(1, 2), stop_words=None),
    "uni+bigramas, sin stopwords": dict(ngram_range=(1, 2), stop_words=list(ENGLISH_STOP_WORDS)),
}
TFIDF = {}
for nombre, cfg in VEC_CONFIGS.items():
    vec = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False,
                          min_df=2, sublinear_tf=True, **cfg)
    TFIDF[nombre] = {"vec": vec, "train": vec.fit_transform(TXT["train"]),
                     "val": vec.transform(TXT["val"]), "test": vec.transform(TXT["test"])}
    print(f"{nombre:30s} -> {TFIDF[nombre]['train'].shape[1]:>6,} features")

# %% [markdown]
# ### 4.1 Optimización de hiperparámetros
#
# Hacemos una **búsqueda en rejilla** (*grid search*): se prueban todas las combinaciones de una
# lista de valores, se entrena cada una con train y se mide su **F1 macro en validación**. Gana la
# que más F1 macro saca. Test no interviene.
#
# Hiperparámetros que se exploran:
#
# - **Del TF-IDF** (4 opciones): solo unigramas o unigramas + bigramas, y quitar o no las stopwords.
# - **Naive Bayes**: el suavizado `alpha` (evita probabilidad cero para palabras no vistas en una
#   clase) y la variante, *Multinomial* o *Complement* (pensada para clases desbalanceadas).
# - **Regresión logística y SVM**: `C`, la inversa de la regularización (C alto deja al modelo
#   ajustarse más a train), y `class_weight`, que con `balanced` penaliza más los errores en las
#   clases pequeñas.

# %%
GRIDS = {
    "Naive Bayes": [{"variante": v, "alpha": a} for v in ("multinomial", "complement")
                    for a in (0.01, 0.05, 0.1, 0.3, 1.0)],
    "Regresión logística": [{"C": c, "class_weight": w} for c in (1, 10, 100) for w in (None, "balanced")],
    "SVM lineal": [{"C": c, "class_weight": w} for c in (0.03, 0.1, 0.3, 1, 3) for w in (None, "balanced")],
}
if FAST:
    GRIDS = {k: v[:2] for k, v in GRIDS.items()}


def crear_shallow(nombre, p):
    if nombre == "Naive Bayes":
        return (ComplementNB if p["variante"] == "complement" else MultinomialNB)(alpha=p["alpha"])
    if nombre == "Regresión logística":
        return LogisticRegression(C=p["C"], class_weight=p["class_weight"], max_iter=1000)
    return LinearSVC(C=p["C"], class_weight=p["class_weight"])


def _probar(nombre, p, vec_nombre, Xtr, ytr, Xva, yva):
    pred = crear_shallow(nombre, p).fit(Xtr, ytr).predict(Xva)
    return {"modelo": nombre, "tfidf": vec_nombre, **p,
            "f1_val": f1_score(yva, pred, average="macro"), "acc_val": accuracy_score(yva, pred)}


def busqueda_shallow(tarea):
    """Grid search sobre validación. Devuelve todas las combinaciones probadas."""
    trabajos = [delayed(_probar)(nombre, p, v, TFIDF[v]["train"], Y[tarea]["train"],
                                 TFIDF[v]["val"], Y[tarea]["val"])
                for nombre, grid in GRIDS.items() for p in grid for v in VEC_CONFIGS]
    return pd.DataFrame(Parallel(n_jobs=N_JOBS)(trabajos))


def evaluar_shallow(tarea, grid):
    """Reentrena la mejor combinación de cada modelo y la evalúa en test."""
    mejores = {}
    pred = DummyClassifier(strategy="most_frequent").fit(TFIDF["unigramas"]["train"], Y[tarea]["train"])
    registrar(tarea, "Baseline", "Clase mayoritaria",
              [pred.predict(TFIDF["unigramas"]["test"])],
              f1_score(Y[tarea]["val"], pred.predict(TFIDF["unigramas"]["val"]), average="macro"))
    for nombre in GRIDS:
        mejor = grid[grid["modelo"] == nombre].sort_values("f1_val", ascending=False).iloc[0]
        p = {k: (None if pd.isna(mejor[k]) else mejor[k]) for k in GRIDS[nombre][0]}
        p = {k: (v.item() if hasattr(v, "item") else v) for k, v in p.items()}   # tipos nativos de Python
        modelo = crear_shallow(nombre, p).fit(TFIDF[mejor["tfidf"]]["train"], Y[tarea]["train"])
        detalle = f"{mejor['tfidf']}; " + ", ".join(f"{k}={v}" for k, v in p.items())
        registrar(tarea, "Shallow", f"TF-IDF + {nombre}",
                  [modelo.predict(TFIDF[mejor["tfidf"]]["test"])], mejor["f1_val"], detalle)
        mejores[nombre] = {"modelo": modelo, "tfidf": mejor["tfidf"], "params": p, "f1_val": mejor["f1_val"]}
    return mejores


t0 = time.time()
GRID_SHALLOW = {"queue": busqueda_shallow("queue")}
MEJOR_SHALLOW = {"queue": evaluar_shallow("queue", GRID_SHALLOW["queue"])}
print(f"{len(GRID_SHALLOW['queue'])} combinaciones probadas en {time.time() - t0:.0f} s")
for nombre, m in MEJOR_SHALLOW["queue"].items():
    print(f"  {nombre:20s} F1 val = {m['f1_val']:.3f} | {m['tfidf']} | {m['params']}")

# %%
# Cómo cambia el F1 de validación con la regularización C, según el TF-IDF usado.
g = GRID_SHALLOW["queue"]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, nombre in zip(axes, ("Regresión logística", "SVM lineal")):
    sub = g[g["modelo"] == nombre]
    for color, v in zip(SERIE, VEC_CONFIGS):
        curva = sub[sub["tfidf"] == v].groupby("C")["f1_val"].max()   # mejor class_weight para cada C
        ax.plot(curva.index, curva.values, color=color, lw=2, marker="o", ms=6,
                markeredgecolor=COL["fondo"], markeredgewidth=1.5, label=v)
    ax.set_xscale("log")
    ax.set_title(nombre)
    ax.set_xlabel("C (más alto = menos regularización)")
axes[0].set_ylabel("F1 macro en validación")
axes[1].legend(title="TF-IDF", loc="lower right")
fig.tight_layout()
guardar(fig, "02_grid_shallow")

# %%
display(tabla("queue").round(3))

# %% [markdown]
# ### 4.2 Combinar el texto con features no textuales
#
# El dataset solo trae texto y etiquetas, así que las features no textuales las calculamos
# nosotros a partir de la **forma** del ticket, no de sus palabras: longitud, número de oraciones,
# si tiene asunto, signos de interrogación y exclamación, cifras, marcadores anonimizados
# (`<tel_num>`, `<acc_num>`...), proporción de mayúsculas y saltos de línea.
#
# No usamos como entrada `type`, `priority` ni los `tag_*`: son etiquetas que pone el agente
# después de leer el ticket, y usarlas sería otra fuga de información.

# %%
def meta_features(fila):
    texto, tokens = fila["texto"], fila["tokens"]
    asunto = fila["subject"] if isinstance(fila["subject"], str) else ""
    letras = [c for c in texto if c.isalpha()]
    return {
        "n_tokens": len(tokens),
        "n_caracteres": len(texto),
        "n_oraciones": len(fila["oraciones"]),
        "long_media_palabra": float(np.mean([len(t) for t in tokens if t[0].isalnum()] or [0])),
        "tiene_asunto": float(bool(asunto.strip())),
        "n_tokens_asunto": len(_TOKEN.findall(asunto.lower())),
        "n_interrogaciones": tokens.count("?"),
        "n_exclamaciones": tokens.count("!"),
        "n_cifras": sum(t[0].isdigit() for t in tokens),
        "n_marcadores": sum(len(t) > 1 and t[0] == "<" for t in tokens),
        "ratio_mayusculas": sum(c.isupper() for c in letras) / max(1, len(letras)),
        "n_saltos_linea": texto.count("\n"),
    }


META = pd.DataFrame([meta_features(f) for f in df.to_dict("records")])
CONTEOS = [c for c in META.columns if c.startswith("n_")]
META[CONTEOS] = np.log1p(META[CONTEOS])                  # los conteos tienen cola larga
escalador = StandardScaler().fit(META.iloc[IDX["train"]])     # se ajusta solo con train
M = {s: escalador.transform(META.iloc[IDX[s]]).astype(np.float32) for s in SPLITS}
N_META = M["train"].shape[1]
print(N_META, "meta-features:", list(META.columns))

# %% [markdown]
# Tres experimentos sobre la cola: solo meta-features (¿llevan señal por sí solas?), y el mejor
# modelo lineal con TF-IDF + meta-features. Como el TF-IDF está normalizado y las meta-features
# no, se multiplican por un peso `w` que también se elige en validación.

# %%
def con_meta(X, meta, w):
    return sparse.hstack([X, sparse.csr_matrix(meta * w)]).tocsr()


solo_meta = LogisticRegression(max_iter=2000, class_weight="balanced").fit(M["train"], Y["queue"]["train"])
registrar("queue", "Shallow", "Solo meta-features + Regresión logística",
          [solo_meta.predict(M["test"])],
          f1_score(Y["queue"]["val"], solo_meta.predict(M["val"]), average="macro"),
          f"{N_META} features de forma, sin texto")

mejor_lineal = max(("Regresión logística", "SVM lineal"), key=lambda n: MEJOR_SHALLOW["queue"][n]["f1_val"])
base = MEJOR_SHALLOW["queue"][mejor_lineal]
X = TFIDF[base["tfidf"]]
pruebas = []
for w in (0.03, 0.1, 0.3, 1.0):
    m = crear_shallow(mejor_lineal, base["params"]).fit(con_meta(X["train"], M["train"], w), Y["queue"]["train"])
    f1v = f1_score(Y["queue"]["val"], m.predict(con_meta(X["val"], M["val"], w)), average="macro")
    pruebas.append((f1v, w, m))
    print(f"  w = {w:<5} F1 val = {f1v:.4f}   (solo texto: {base['f1_val']:.4f})")
f1v, w, m = max(pruebas, key=lambda t: t[0])
registrar("queue", "Shallow", f"TF-IDF + meta-features + {mejor_lineal}",
          [m.predict(con_meta(X["test"], M["test"], w))], f1v, f"{base['tfidf']}; w={w}")
display(tabla("queue").round(3))
