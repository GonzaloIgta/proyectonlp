# %% [markdown]
# ## 5. Deep learning: CNN y BiLSTM con atención
#
# Los modelos anteriores tratan el ticket como una bolsa de palabras. Las redes de este bloque
# leen el ticket como una **secuencia**: cada token se sustituye por un vector denso de 300
# dimensiones (*embedding*) y la red aprende a combinar esos vectores teniendo en cuenta el orden.
#
# ### 5.1 Vocabulario y secuencias
#
# Partimos de la clase `Vocabulary` de E2, adaptada con `<PAD>` (índice 0) para el relleno de
# secuencias: se construye solo con train, reserva `<UNK>` para las palabras desconocidas y descarta
# las que aparecen menos de 2 veces. Cada ticket pasa a ser una lista de índices, recortada o
# rellenada con `<PAD>` hasta 200 tokens (el 99 % de los tickets tiene menos de 180).

# %%
class Vocabulary:
    PAD, UNK = "<PAD>", "<UNK>"

    def __init__(self, docs, min_freq=2):
        self.freq = Counter(t for doc in docs for t in doc)
        self.itos = [self.PAD, self.UNK] + [t for t, n in self.freq.most_common() if n >= min_freq]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        return [self.stoi.get(t, 1) for t in tokens]


_tokens = df["tokens"].tolist()
TOKENS = {s: [_tokens[i] for i in IDX[s]] for s in SPLITS}
vocab = Vocabulary(TOKENS["train"], min_freq=2)
MAX_LEN, EMB_DIM = 200, 300


def a_matriz(docs):
    X = np.zeros((len(docs), MAX_LEN), dtype=np.int64)
    L = np.zeros(len(docs), dtype=np.int64)
    for i, doc in enumerate(docs):
        ids = vocab.encode(doc)[:MAX_LEN]
        X[i, :len(ids)], L[i] = ids, len(ids)
    return X, L


DATOS = {}
for s in SPLITS:
    X_ids, L_ids = a_matriz(TOKENS[s])
    DATOS[s] = {"X": torch.tensor(X_ids, device=DEVICE), "L": torch.tensor(L_ids),
                "M": torch.tensor(M[s], device=DEVICE)}
YT = {t: {s: torch.tensor(Y[t][s], device=DEVICE) for s in SPLITS} for t in TAREAS}
print(f"Vocabulario: {len(vocab):,} tokens | tipos distintos en train: {len(vocab.freq):,}")

# %% [markdown]
# ### 5.2 Tres formas de inicializar los embeddings
#
# El enunciado pide comparar tipos de embeddings y el efecto del *fine-tuning*. Probamos cinco
# variantes, que salen de combinar tres orígenes con dos formas de usarlos:
#
# | Origen | Qué es |
# |---|---|
# | **Aleatorio** | Vectores al azar que la red aprende desde cero, solo con nuestros tickets |
# | **Google News** | Word2Vec preentrenado con unos 100.000 millones de palabras de noticias. Conoce el inglés general pero no nuestro dominio |
# | **Word2Vec propio** | Mismo esquema que el `w2v_all` de E2 (tickets y respuestas), reentrenado solo sobre el `train` ya sin duplicados. Poco texto, pero del dominio |
#
# - **Congelado**: los vectores no cambian durante el entrenamiento. La red solo aprende a usarlos.
# - **Fine-tuning**: los vectores también se actualizan, así que pueden adaptarse a la tarea. Hay
#   más parámetros que aprender y, con ello, más riesgo de sobreajuste.
#
# Para Google News mantenemos los *fallbacks* de E2: si la palabra en minúscula no está, se prueba
# con mayúscula inicial, en mayúsculas y sin el posesivo `'s`. Las palabras sin vector reciben uno
# aleatorio.

# %%
def buscar_vector(kv, palabra):
    candidatos = [palabra, palabra.capitalize(), palabra.upper()]
    if palabra.endswith("'s"):
        candidatos += [palabra[:-2], palabra[:-2].capitalize()]
    for c in candidatos:
        if c in kv.key_to_index:
            return kv[c]
    return None


def matriz_desde(kv):
    """Matriz (vocabulario x 300) con el vector de cada palabra y qué palabras tienen vector."""
    rng = np.random.default_rng(SEED)
    matriz = rng.uniform(-0.25, 0.25, (len(vocab), EMB_DIM)).astype(np.float32)
    matriz[0] = 0.0                                   # <PAD>
    cubierto = np.zeros(len(vocab), dtype=bool)
    for i, palabra in enumerate(vocab.itos[2:], start=2):
        v = buscar_vector(kv, palabra)
        if v is not None:
            matriz[i], cubierto[i] = v, True
    return matriz, cubierto


# Google News: 1,6 GB de descarga y 3,6 GB en RAM. Solo se carga una vez: nos quedamos con las
# filas de nuestro vocabulario y las guardamos en disco.
huella = hashlib.md5(" ".join(vocab.itos).encode("utf-8")).hexdigest()[:10]
cache = ART_DIR / f"emb_google_news_{huella}.npz"
if cache.exists():
    z = np.load(cache)
    E_GOOGLE, CUB_GOOGLE = z["matriz"], z["cubierto"]
else:
    import gensim.downloader as api
    from gensim.models import KeyedVectors
    ruta_w2v = api.load("word2vec-google-news-300", return_path=True)
    kv = KeyedVectors.load_word2vec_format(ruta_w2v, binary=True, limit=300_000 if FAST else None)
    E_GOOGLE, CUB_GOOGLE = matriz_desde(kv)
    np.savez_compressed(cache, matriz=E_GOOGLE, cubierto=CUB_GOOGLE)
    del kv

# Word2Vec propio: tickets + respuestas, solo de train (para que test no influya en los vectores).
from gensim.models import Word2Vec

_ors, _ors_resp = df["oraciones"].tolist(), df["oraciones_resp"].tolist()
corpus_w2v = [o for i in IDX["train"] for o in _ors[i] + _ors_resp[i]]
w2v = Word2Vec(corpus_w2v, vector_size=EMB_DIM, window=5, min_count=2, sg=1, negative=10,
               epochs=10, workers=4, seed=SEED)
E_PROPIO, CUB_PROPIO = matriz_desde(w2v.wv)
print(f"Word2Vec propio: {len(corpus_w2v):,} oraciones, {sum(map(len, corpus_w2v)):,} tokens, "
      f"{len(w2v.wv):,} palabras")


def cobertura(nombre, cubierto):
    fila = {"embedding": nombre, "tipos del vocabulario con vector": cubierto[2:].mean()}
    for s in ("val", "test"):
        ids = DATOS[s]["X"].cpu().numpy()
        fila[f"tokens de {s} con vector"] = cubierto[ids[ids > 0]].mean()
    return fila


COBERTURA = pd.DataFrame([cobertura("Google News", CUB_GOOGLE), cobertura("Word2Vec propio", CUB_PROPIO)])
display(COBERTURA.set_index("embedding").style.format("{:.2%}"))

EMBEDDINGS = {   # nombre -> (matriz inicial, ¿congelado?)
    "Aleatorio": (None, False),
    "Google News congelado": (E_GOOGLE, True),
    "Google News fine-tuning": (E_GOOGLE, False),
    "Word2Vec propio congelado": (E_PROPIO, True),
    "Word2Vec propio fine-tuning": (E_PROPIO, False),
}
