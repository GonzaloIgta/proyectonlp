# E2 – Procesamiento y análisis de los datos

**Asignatura:** Fundamentos del Procesamiento de Lenguaje Natural · Deusto
**Corpus:** [`Tobi-Bueck/customer-support-tickets`](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) (tickets de soporte, anotado)
**Notebook:** `tickets_pipeline.ipynb`

> **Borrador.** Los valores marcados con `⟨XX⟩` dependen de la ejecución de las secciones 8–10 del notebook (embeddings y exportación) y hay que rellenarlos con las salidas reales. Las secciones 8–10 no se han ejecutado todavía. El resto de cifras proceden de la ejecución ya realizada de las secciones 1–7.

---

## 1. Resumen

El proyecto trabaja con un corpus de tickets de soporte al cliente en inglés. Cada ticket tiene un asunto (`subject`), un cuerpo (`body`), la respuesta del agente (`answer`) y varias anotaciones: cola de destino (`queue`), prioridad (`priority`), tipo (`type`) y etiquetas (`tag_1` … `tag_N`).

El pipeline implementado:

1. Limpia el corpus (filas sin cuerpo y tickets en alemán).
2. Lo divide en oraciones, lo normaliza y lo tokeniza.
3. Separa **train (80 %) / test (20 %)** estratificando por `queue`.
4. Construye un vocabulario con token `<UNK>` solo con train.
5. Genera una representación **TF-IDF** (unigramas y bigramas).
6. Genera una representación con **word embeddings** (Word2Vec preentrenado de Google News y Word2Vec propio) y analiza cuántas palabras quedan sin vector.
7. Exporta todo a `corpus.jsonl` + `meta.json` para reutilizarlo en las siguientes entregas.

| Concepto | Valor |
|---|---|
| Filas originales → tras limpieza | 61 765 → **28 260** |
| Tickets train / test | **22 608 / 5 652** |
| Oraciones por ticket (media) | 4,8 |
| Tokens por ticket (media) | 60,8 (respuesta: 58,6) |
| Tipos distintos en train | 5 923 |
| Tamaño del vocabulario (`MIN_FREQ = 2`, incluye `<UNK>`) | **4 897** |
| Tokens de test caen en `<UNK>` | 551 / 345 339 (**0,16 %**) |
| Características TF-IDF (1–2-gramas) | **71 191** |

---

## 2. Tokenización, división en oraciones y preprocesado

### 2.1 Limpieza (`limpieza_datos`)

Se aplican dos filtros y el resultado se guarda en `dataprocessed/dataprocessed.csv`:

- **Se eliminan los tickets con `body` vacío**: sin texto no hay nada que representar.
- **Se eliminan los tickets en alemán** (`language == "de"`): mezclar idiomas complicaría el vocabulario, las stopwords y los embeddings (el modelo de embeddings preentrenado es solo inglés).

### 2.2 Texto de entrada

Para cada ticket se concatena `subject + "\n" + body`. **`answer` no se incluye** como texto del ticket: la respuesta del agente no existe en el momento de clasificar, así que usarla sería fuga de información. Se procesa con las mismas funciones, pero en columnas aparte (`answer_*`), y solo se utiliza (únicamente con tickets de train) como texto adicional para entrenar Word2Vec.

**Saltos de línea literales.** En el dataset los saltos de línea de `body` y `answer` vienen como los dos caracteres `\n` (barra + n), no como saltos reales. Sin corregirlo, el divisor de oraciones no corta y el tokenizador genera tokens basura (`\n\nI` → `n` + `ni`). La función `fix_newlines` los convierte en saltos reales **antes** de dividir en oraciones.

### 2.3 Pasos del procesamiento

| Paso | Función | Qué hace |
|---|---|---|
| 1. Oraciones | `split_sentences` | Parte en saltos de línea y tras `.`, `!` o `?`. Se hace sobre el texto original, antes de minúsculas. |
| 2. Normalización | `normalize` | URLs → `<url>`, emails → `<email>`, números → `<num>`; después minúsculas. |
| 3. Tokenización | `tokenize` | Expresión regular que conserva como un solo token: marcadores `<...>` (incluidos los del dataset, como `<name>`), palabras con apóstrofo (`don't`) y cada signo de puntuación suelto. |

```python
URL_RE        = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE      = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
NUM_RE        = re.compile(r"\b\d+(?:[.,]\d+)*\b")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
TOKEN_RE      = re.compile(r"<[a-z_]+>|\w+(?:'\w+)*|[^\w\s]")
```

**Decisiones y justificación**

- **Marcadores `<url>`, `<email>`, `<num>`**: URLs, emails y números casi nunca se repiten, así que sin sustituirlos inflarían el vocabulario con tokens únicos sin valor semántico.
- **Minúsculas**: reduce el vocabulario. El Word2Vec preentrenado distingue mayúsculas, pero contiene la gran mayoría de palabras comunes en minúscula; para el resto se prueba la forma capitalizada (apartado 4.2).
- **No se lematiza ni se eliminan stopwords en el preprocesado.** Se conservan para los embeddings (el contexto importa) y las stopwords solo se filtran más tarde en TF-IDF y al promediar vectores de ticket.
- **Puntuación**: se tokeniza como token independiente (columna `text_tok`, formato de la práctica de Perceptron/Yelp) y se descarta en `tokens`.

### 2.4 Columnas resultantes

| Columna | Contenido |
|---|---|
| `sentences` | Lista de oraciones (JSON) |
| `n_sentences`, `n_tokens` | Recuentos |
| `text_tok` | Tokens con puntuación, separados por espacios |
| `tokens` | Tokens sin puntuación (entrada del vocabulario, de TF-IDF y de los embeddings) |
| `answer_sentences`, `answer_tokens`, `n_tokens_answer` | Lo mismo para la respuesta. **No se usa como feature.** |
| `split` | `train` / `test` |

### 2.5 Ejemplo

```text
RAW        : Account Disruption | Dear Customer Support Team, | | I am writing to report a significant problem ...
ORACIONES  : ['Account Disruption', 'Dear Customer Support Team,', 'I am writing to report a significant problem ...']
TEXT_TOK   : account disruption dear customer support team , i am writing to report a significant problem ...
TOKENS     : account disruption dear customer support team i am writing to report a significant problem ...
```

### 2.6 Partición train / test

El dataset solo trae `train`, así que la partición se hace en el notebook: `train_test_split(test_size=0.2, random_state=42, stratify=queue)`. Se hace **antes** de construir el vocabulario, TF-IDF o embeddings, para que nada de test influya en ellos.

### 2.7 Vocabulario (`Vocabulary`)

Clase basada en la de la práctica (Rao & McMahan, 2019): mapea token ↔ índice y reserva `<UNK>`. Se construye **solo con train** y con **corte por frecuencia** (`MIN_FREQ = 2`): los tokens con menos de 2 apariciones se representan como `<UNK>`.

- 5 923 tipos distintos en train → 4 897 en el vocabulario.
- Cobertura en test: solo el **0,16 %** de los tokens es `<UNK>`.
- Se serializa en `features/vocab.json` (`to_serializable` / `from_serializable`).

---

## 3. Representación con TF-IDF

Se usa TF-IDF porque es una base sólida y fácil de interpretar para clasificar tickets (cola, prioridad), y permite comparar después contra los embeddings.

```python
TfidfVectorizer(
    lowercase=False, tokenizer=str.split, token_pattern=None,   # usa nuestros tokens, no vuelve a tokenizar
    stop_words=sorted(STOP_EN),                                  # stopwords de spaCy (inglés)
    min_df=2, max_df=0.9, ngram_range=(1, 2),
)
```

- **Ajuste solo con train**; test solo se transforma (sin fuga de información).
- `min_df=2` elimina términos que aparecen en un solo documento; `max_df=0.9` elimina términos presentes en casi todos.
- Resultado: matrices dispersas `X_train` (22 608 × 71 191) y `X_test` (5 652 × 71 191).
- Se guardan `X_train.npz`, `X_test.npz`, `tfidf_vectorizer.joblib`, `idx_train.npy`, `idx_test.npy` (los índices permiten recuperar `queue`, `priority` y `tags`).

**Palabras con mayor peso TF-IDF medio (global):** `data` (0,0326), `issue`, `security`, `medical`, `analytics`.

> **Observación.** `data` y `security` aparecen en el top de casi todas las colas (salvo *Billing and Payments*, donde destacan `billing` y `payment`). Esto sugiere que el vocabulario del dataset es poco discriminativo entre colas, probablemente por tratarse de un corpus sintético con muchos textos casi idénticos. Conviene tenerlo en cuenta al interpretar los resultados de clasificación.

---

## 4. Representación con Word Embeddings

### 4.1 ¿Por qué Word2Vec? (y relación con GloVe)

Word2Vec y GloVe son familias de embeddings **estáticos**: asignan un único vector fijo a cada palabra y se usan exactamente igual (una matriz `palabra → vector`). Se diferencian en cómo se entrenan:

| | Word2Vec | GloVe |
|---|---|---|
| Entrenamiento | Predictivo: una red pequeña predice una palabra a partir de su contexto (o al revés) en una ventana local | Basado en recuentos: factoriza la matriz global de coocurrencias de palabras |
| Variantes | CBOW / skip-gram | — |

Como en el curso se trabaja con Word2Vec, esta entrega usa **Word2Vec** tanto preentrenado como entrenado por nosotros, de modo que se pueda comparar el mismo algoritmo con distinto corpus (general vs. dominio).

### 4.2 Embedding preentrenado: Word2Vec Google News

Se usa **`word2vec-google-news-300`** (cargado con `gensim.downloader`): 3 millones de palabras y expresiones, 300 dimensiones, entrenado con ~100 000 millones de palabras de noticias.

**Cómo se usa**

- `emb_matrix_pretrained.npy`: matriz `[4 897 × 300]` alineada con los índices de `vocab.json` (fila *i* = vector del token con índice *i*).
- `doc_emb_pretrained_{train,test}.npy`: un vector por ticket = **media** de los vectores de sus palabras, sin stopwords ni `<UNK>`.

**Particularidades del modelo frente a nuestra tokenización**

| Caso | Problema | Solución |
|---|---|---|
| Mayúsculas | El modelo distingue mayúsculas y nuestros tokens están en minúsculas; algunos términos solo existen capitalizados | `get_pre_vec` prueba `palabra`, `Palabra` y `PALABRA` |
| Stopwords | No incluye varias (`a`, `and`, `of`, `to`…), lo que inflaría el % de OOV | La cobertura se informa **con y sin stopwords**; sin stopwords es la cifra relevante para los vectores de ticket |
| Marcadores `<name>`, `<num>`, `<url>`, `<email>` | No existen en el modelo (los generamos nosotros) | Se cuentan **aparte** (no son OOV reales); vector aleatorio propio en la matriz |
| Posesivos (`customer's`) | Pueden no existir | Se usa el vector de la palabra base (`customer`) |
| `<UNK>` | — | Vector de ceros |

**Coste:** el modelo completo ocupa ~3,6 GB en RAM. El parámetro `PRETRAINED_LIMIT` permite cargar solo las N palabras más frecuentes (el fichero está ordenado por frecuencia); en ese caso la cobertura medida es ligeramente más pesimista.

### 4.3 Análisis de palabras que no se pueden representar (OOV)

Se mide la cobertura por **tipos** (palabras distintas) y por **tokens** (apariciones), excluyendo los marcadores. Los tokens pesan más que los tipos: un OOV raro cuenta mucho en tipos y casi nada en tokens.

| Conjunto | Tipos | Tipos OOV | % tipos OOV | Tokens | Tokens OOV | % tokens OOV |
|---|---:|---:|---:|---:|---:|---:|
| Train (todos los tipos) | 5 923 | ⟨XX⟩ | ⟨XX⟩ | ⟨XX⟩ | ⟨XX⟩ | ⟨XX⟩ |
| Train (vocabulario, freq ≥ 2) | 4 897 | ⟨XX⟩ | ⟨XX⟩ | ⟨XX⟩ | ⟨XX⟩ | ⟨XX⟩ |
| Test | ⟨XX⟩ | ⟨XX⟩ | ⟨XX⟩ | 345 339 | ⟨XX⟩ | ⟨XX⟩ |
| Test (sin stopwords) | ⟨XX⟩ | ⟨XX⟩ | ⟨XX⟩ | ⟨XX⟩ | ⟨XX⟩ | ⟨XX⟩ |

*Fuente: sección 8 del notebook (`features/embedding_coverage.csv`).*

**Marcadores (no cuentan como OOV):** `<num>` ⟨XX⟩ %, `<name>` ⟨XX⟩ %, `<email>` ⟨XX⟩ %, … de los tokens de train.

**Qué palabras se quedan sin vector** (top por frecuencia): ⟨XX: pegar el top 10 de la salida⟩. Del total de tokens OOV en train, ⟨XX⟩ % son stopwords.

**Clasificación de los OOV:**

| Categoría | Tipos | Tokens |
|---|---:|---:|
| Alfabético (jerga, productos, typos, nombres propios) | ⟨XX⟩ | ⟨XX⟩ |
| Contiene dígitos | ⟨XX⟩ | ⟨XX⟩ |
| Con apóstrofo (no resuelto por el fallback) | ⟨XX⟩ | ⟨XX⟩ |
| Otros | ⟨XX⟩ | ⟨XX⟩ |

**Interpretación** *(ajustar según los números reales)*: ⟨XX: p. ej. "el modelo cubre el XX % de los tokens de test sin stopwords; los OOV son sobre todo términos de producto / typos y apenas afectan a la representación media del ticket"⟩.

### 4.4 ¿Entrenar nuestro propio Word2Vec?

**Criterio de decisión:** se considera que el número de palabras sin representar es *grande* si, en test y sin stopwords, el % de tokens OOV supera ~5 % o el % de tipos OOV supera ~15 % (parámetros `THRESH_*` del notebook).

**Resultado con el preentrenado:** ⟨XX % de tokens OOV y XX % de tipos OOV⟩ → ⟨se considera / no se considera grande⟩.

Aun así se entrena un Word2Vec propio para compararlo, porque el dominio (soporte técnico) tiene vocabulario que un modelo entrenado con noticias no recoge bien. Se entrenan dos variantes con los mismos hiperparámetros sobre `features/w2v_corpus_train.txt` (**solo train**, una oración por línea, ya sin puntuación):

| Modelo | Texto de entrenamiento | Corpus |
|---|---|---|
| `w2v_all` | Tickets + respuestas (`answer`) | 195 124 oraciones, 2 696 196 tokens |
| `w2v_tickets` | Solo tickets | 109 354 oraciones, 1 372 877 tokens |

**Hiperparámetros:** `sg=1` (skip-gram: funciona mejor con corpus pequeños y palabras poco frecuentes), `vector_size=300` (igual que el preentrenado, para poder intercambiar las matrices), `window=5`, `min_count=2` (igual que `MIN_FREQ`), `epochs=20`, `seed=42`.

**Por qué incluir `answer` en el corpus de Word2Vec** (las respuestas aportan el 49,1 % de los tokens):

- Word2Vec es no supervisado: no usa etiquetas, solo contexto de palabras.
- Las respuestas comparten dominio y léxico con los tickets, así que aumentan las apariciones de palabras que en los tickets aparecen poco.
- **Solo se usa con train**, por lo que ni el texto de test ni sus respuestas influyen en los vectores.
- **Limitación:** las respuestas son de otro "hablante" (registro más formal, plantillas repetidas), así que los vectores pueden sesgarse hacia ese estilo. Por eso se entrena también la variante `w2v_tickets` para comparar.

**Comparación de cobertura**

| Embedding | Conjunto | % tipos OOV | % tokens OOV |
|---|---|---:|---:|
| Preentrenado (Google News) | Train vocab | ⟨XX⟩ | ⟨XX⟩ |
| Preentrenado (Google News) | Test | ⟨XX⟩ | ⟨XX⟩ |
| Preentrenado (Google News) | Test sin stopwords | ⟨XX⟩ | ⟨XX⟩ |
| `w2v_all` (propio) | Train vocab | ⟨XX⟩ | ⟨XX⟩ |
| `w2v_all` (propio) | Test | ⟨XX⟩ | ⟨XX⟩ |
| `w2v_all` (propio) | Test sin stopwords | ⟨XX⟩ | ⟨XX⟩ |
| `w2v_tickets` (propio) | Train vocab | ⟨XX⟩ | ⟨XX⟩ |
| `w2v_tickets` (propio) | Test | ⟨XX⟩ | ⟨XX⟩ |
| `w2v_tickets` (propio) | Test sin stopwords | ⟨XX⟩ | ⟨XX⟩ |

> **Cómo leer esta tabla.** La cobertura del Word2Vec propio sobre *train* es casi 100 % **por construcción** (se entrena con el mismo texto que define el vocabulario y con el mismo corte de frecuencia). La cifra comparable con el preentrenado es la de **test**. Además, el propio incluye las stopwords, que el preentrenado no, por eso se comparan también las filas "sin stopwords".

**Comprobación cualitativa** (vecinos más cercanos de palabras frecuentes del dominio en cada espacio): ⟨XX: incluir 2–3 ejemplos de la salida y comentar si el espacio propio captura mejor el dominio⟩.

**Decisión final:** ⟨XX: p. ej. "Se usará el preentrenado como representación principal y el propio como alternativa / se usará el propio porque..."⟩. Las dos matrices tienen el mismo formato (`[4 897 × 300]`), de modo que se pueden intercambiar sin tocar el resto del código.

**Posible mejora (no implementada):** concatenar ambos espacios (preentrenado ‖ propio, 600 dimensiones) o continuar el entrenamiento del preentrenado con el corpus del dominio. No se mezclan los vectores directamente porque pertenecen a espacios distintos.

---

## 5. Estructura de los datos para su reutilización

Todo el texto procesado se exporta a **`features/corpus.jsonl`** (un ticket por línea) acompañado de **`features/meta.json`**. Así cualquier tarea posterior puede cargar el corpus sin repetir el preprocesado.

### 5.1 Formato de cada registro

```json
{
  "id": 0,
  "split": "train",
  "language": "en",
  "labels": {
    "queue": "Technical Support",
    "priority": "high",
    "type": "Incident",
    "tags": ["Account", "Disruption"]
  },
  "text": {
    "subject": "Account Disruption",
    "sentences": ["Account Disruption", "Dear Customer Support Team,", "..."],
    "tokens": ["account", "disruption", "dear", "..."],
    "token_ids": [112, 904, 37, "..."]
  },
  "aux": { "answer_tokens": ["thank", "you", "for", "..."] }
}
```

| Campo | Descripción |
|---|---|
| `id` | Posición en `dataprocessed.csv`; coincide con `idx_train` / `idx_test` y con las filas de las matrices `.npz` / `.npy` |
| `split` | `train` o `test` |
| `labels` | Anotaciones a predecir (`queue`, `priority`, `type`, `tags`) |
| `text.sentences` | Oraciones del texto original |
| `text.tokens` | Tokens normalizados sin puntuación |
| `text.token_ids` | Índices según `vocab.json` (fuera de vocabulario → índice de `<UNK>`) |
| `aux.answer_tokens` | Tokens de la respuesta del agente. **No usar como feature**; solo trazabilidad y corpus de Word2Vec |

`meta.json` recoge los parámetros (partición, vocabulario, TF-IDF, embeddings), los tamaños, las columnas de etiquetas y los mapeos etiqueta → entero. `labels.json` contiene esos mapeos por separado.

### 5.2 Ficheros generados

| Fichero | Contenido |
|---|---|
| `raw/dataraw.csv` | Dataset original |
| `dataprocessed/dataprocessed.csv` | Tabla completa: texto original, oraciones, tokens y split |
| `features/corpus.jsonl` | **Corpus estructurado (formato principal de reutilización)** |
| `features/meta.json`, `features/labels.json` | Parámetros, tamaños y mapeos de etiquetas |
| `features/vocab.json` | `Vocabulary` serializado |
| `features/X_train.npz`, `X_test.npz`, `tfidf_vectorizer.joblib` | TF-IDF |
| `features/idx_train.npy`, `idx_test.npy` | Índices de las particiones |
| `features/emb_matrix_pretrained.npy`, `emb_matrix_own.npy` | Matrices de embeddings (Word2Vec preentrenado / propio) alineadas con `vocab.json` |
| `features/doc_emb_{pretrained,own}_{train,test}.npy` | Un vector por ticket |
| `features/w2v_all.model`, `w2v_tickets.model` | Modelos Word2Vec propios |
| `features/w2v_corpus_train.txt` | Corpus de entrenamiento de Word2Vec |
| `features/embedding_coverage.csv` | Tabla de cobertura OOV |

### 5.3 Cómo cargarlo

```python
import json
import numpy as np
from scipy import sparse

FEAT = "../data/tickets/features"

with open(f"{FEAT}/corpus.jsonl", encoding="utf-8") as fp:
    records = [json.loads(line) for line in fp]

train = [r for r in records if r["split"] == "train"]
y_queue = [r["labels"]["queue"] for r in train]

X_train = sparse.load_npz(f"{FEAT}/X_train.npz")            # TF-IDF; fila i ↔ train[i]
D_train = np.load(f"{FEAT}/doc_emb_pretrained_train.npy")    # embedding medio por ticket
M       = np.load(f"{FEAT}/emb_matrix_pretrained.npy")       # para capas Embedding (fila = índice de vocab.json)
```

---

## 6. Decisiones de diseño y limitaciones

- **Sin fuga de información:** la partición se hace primero; vocabulario, TF-IDF y Word2Vec se ajustan solo con train; `answer` nunca es feature.
- **Corpus sintético:** el dataset parece generado (muchos textos casi idénticos, vocabulario genérico). Los resultados de clasificación pueden ser optimistas y las palabras más discriminativas poco informativas.
- **Idioma:** solo se conserva inglés; los tickets en alemán se descartan (el modelo preentrenado es inglés), lo que reduce de forma importante el tamaño del corpus.
- **Memoria:** el modelo preentrenado ocupa ~3,6 GB en RAM; se puede limitar con `PRETRAINED_LIMIT`.
- **Mayúsculas, stopwords y marcadores:** requieren tratamiento especial con el Word2Vec preentrenado (apartado 4.2).
- **Reproducibilidad de Word2Vec:** con `workers > 1` gensim no garantiza resultados idénticos entre ejecuciones aunque se fije `seed`; para reproducir exactamente se debe usar `workers=1`.
- **Un solo corpus:** este notebook procesa únicamente el corpus de tickets. ⟨XX: si el proyecto incluye más corpus (anotados o no), indicar aquí cómo se procesarán con el mismo pipeline.⟩

## 7. Referencias

- Rao, D., & McMahan, B. (2019). *Natural Language Processing with PyTorch*. O'Reilly. (Clase `Vocabulary`)
- Mikolov, T., et al. (2013). *Distributed Representations of Words and Phrases and their Compositionality*. NeurIPS. (modelo Google News)
- Mikolov, T., et al. (2013). *Efficient Estimation of Word Representations in Vector Space*.
- Řehůřek, R., & Sojka, P. (2010). *Software Framework for Topic Modelling with Large Corpora* (gensim).
