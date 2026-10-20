# Procesamiento y análisis de tickets de soporte (E2)

Corpus: [`Tobi-Bueck/customer-support-tickets`](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) (HuggingFace). Objetivo del proyecto: dado un ticket, predecir `queue` (a quién va), `priority` (importancia) y `tags`. Esta entrega (E2) prepara el texto y sus representaciones.

Todo el código y la explicación están en un único notebook: [`src/tickets_pipeline.ipynb`](src/tickets_pipeline.ipynb). Las secciones de este README siguen los apartados del enunciado y remiten a las del notebook.

## Instalación y ejecución

```
python -m venv env
.\env\Scripts\Activate.ps1
pip install -r requirements.txt        # o: pip install -e .
```

Abrir `src/tickets_pipeline.ipynb` (kernel del `env`) y ejecutar todas las celdas **en orden**. El notebook usa rutas relativas a `src/` (`../data/...`). La primera vez descarga el dataset y GloVe (~130 MB, a la caché de gensim).

## 0. Datos y limpieza (secciones 1-2 del notebook)

El dataset solo trae el split `train`, así que el train/test lo hacemos nosotros. Limpieza:

- Se eliminan los tickets con `body` vacío.
- Se eliminan los tickets en alemán (`language == "de"`). Resultado: **28.260 tickets, todos en inglés**.

**Por qué quitar el alemán:** los embeddings y las stopwords que usamos son de inglés. Con otro idioma casi todas las palabras quedarían sin vector.

**Texto de entrada:** `subject` + `body`. La columna `answer` **no se usa**: en el momento de clasificar un ticket nuevo todavía no existe respuesta, y usarla sería hacer trampa.

## 1. Tokenización y división en oraciones (sección 3)

| Paso | Qué se hace | Por qué |
|---|---|---|
| Limpieza previa | `\n` literal y `<br>`/`<br/>`/`<br />` pasan a salto de línea; se eliminan placeholders (`<name>`, `<tel_num>`, `<acc_num>`...) | El CSV trae los saltos como texto y el dataset anonimiza con placeholders. Sin esto salían "palabras" falsas (`br><br`, `ni`) |
| Oraciones | Se parte en saltos de línea y tras `.`, `!` o `?` (regex) | Suficiente para texto de tickets; no exige modelos descargables |
| Normalización | URLs, emails y números → `<url>`, `<email>`, `<num>`; minúsculas | Reduce vocabulario y evita tokens casi únicos |
| Tokens | Regex que mantiene marcadores `<...>`, palabras con apóstrofo (`don't`) y puntuación suelta; luego se descarta la puntuación | Minúsculas y sin puntuación porque no aportan a la clasificación |

**Lo que NO se hace, a propósito:**
- No se eliminan stopwords aquí: TF-IDF las filtra por su cuenta y para embeddings conviene conservar el contexto.
- No se lematiza: habría que instalar un modelo de spaCy completo y, para este corpus, la ganancia es pequeña.

**Vocabulario (sección 4):** clase `Vocabulary` (token ↔ índice, con `<UNK>`) construida solo con train y con corte por frecuencia (`MIN_FREQ = 2`). Se guarda en `features/vocab.json`.

## 2. Representación TF-IDF (secciones 5-6)

**Por qué TF-IDF y no Bag of Words:** BoW cuenta apariciones, así que palabras omnipresentes (`please`, `support`) dominan el vector. TF-IDF multiplica por la rareza de la palabra en todo el corpus, lo que destaca las palabras que distinguen una cola de otra (`invoice`, `outage`). Para clasificar tickets suele funcionar mejor.

Configuración del `TfidfVectorizer`:

| Parámetro | Valor | Motivo |
|---|---|---|
| `stop_words` | stopwords inglesas de spaCy | Quitan ruido (`the`, `and`) |
| `min_df` | 2 | Ignora palabras que salen en un solo ticket (erratas, ids) |
| `max_df` | 0.9 | Ignora palabras presentes en más del 90 % de tickets |
| `ngram_range` | (1, 2) | Conserva algo de orden (`reset password`) |
| `lowercase` | False | Los tokens ya vienen en minúsculas |

**Split:** 80 % train / 20 % test (**22.608 / 5.652** tickets), `random_state=42` y estratificado por `queue`.
**Sin fuga de información:** el vectorizador se ajusta (`fit`) **solo con train**; test solo se transforma.

Resultado: matrices dispersas de **70.948 términos**. Se guardan como `.npz` porque casi todas las celdas son ceros; densas ocuparían más de 12 GB, dispersas unos 12 MB.

## 3. Word embeddings (sección 7)

**Modelo:** GloVe `glove-wiki-gigaword-100` (gensim): 400.000 palabras, 100 dimensiones, en minúsculas.

**Adaptación de tokens:** GloVe separa las contracciones (`don't` → `do` + `n't`) y nuestro tokenizador no. Solo para buscar el vector se parten igual, y los marcadores `<num>`, `<url>`, `<email>` se ignoran. Así no se cuentan como OOV palabras que sí existen.

**Vector por ticket:** media de los vectores de sus palabras (las OOV se ignoran). Resultado: `ticket_vectors.npy`, de forma (28.260 × 100).

### Análisis de palabras sin vector (OOV)

| Medida | Resultado |
|---|---|
| Palabras distintas sin vector | **247 de 5.955 (4,1 %)** |
| Apariciones sin vector | **7.763 de 1.716.336 tokens (0,5 %)** |
| Tickets sin ninguna palabra con vector | 2 |

La lista completa está en `data/tickets/embeddings/oov_words.csv`. Las más frecuentes son nombres de producto o tecnología (`clickup`, `elasticsearch`, `smartsheet`, `redis`, `airtable`, `zapier`, `kubernetes`).

**Por qué hay OOV:** GloVe se entrenó con Wikipedia y noticias, y estas marcas técnicas casi no aparecen. Además, el modelo descarta palabras raras.

**¿Hace falta un Word2Vec propio?** El número de OOV **no es grande**: el 4,1 % de las palabras distintas y solo el 0,5 % del texto. Casi todo el contenido queda cubierto, así que entrenar un Word2Vec propio no es imprescindible. Su ventaja sería cubrir las marcas técnicas, y con 28.000 tickets habría vocabulario para hacerlo como complemento. *(Decisión pendiente de confirmar por el equipo.)*

## 4. Estructura reutilizable (sección 8)

Salida en `data/tickets/export/`:

- `tickets_processed.jsonl`: **un ticket por línea**, con `id`, `split`, `text`, `sentences`, `tokens`, `n_sentences`, `n_tokens`, `queue`, `priority`, `type` y `tags` (lista, sin vacíos).
- `manifest.json`: qué es cada archivo y cómo cargarlo.

**Por qué JSONL:** admite listas (oraciones, tokens, tags) dentro de cada registro, que un CSV no representa bien, y se lee línea a línea. Las matrices numéricas **no** van en el JSON porque son grandes; se guardan en `.npz` / `.npy` y se alinean con el JSON por `id` (la fila *i* de cada matriz es el ticket con `id == i`).

Ejemplo de carga:

```python
import numpy as np, pandas as pd
from scipy import sparse

df = pd.read_json("data/tickets/export/tickets_processed.jsonl", lines=True)
X_train = sparse.load_npz("data/tickets/features/X_train.npz")
idx_train = np.load("data/tickets/features/idx_train.npy")   # ids de las filas de X_train
y_train = df.loc[idx_train, "queue"]
vectores = np.load("data/tickets/embeddings/ticket_vectors.npy")  # (28260, 100)
```

## Estructura del proyecto

```
src/tickets_pipeline.ipynb      # todo el código y la explicación
data/tickets/
  raw/  dataprocessed/  features/  embeddings/  export/
```

## Actualizar dependencias

```
pipreqs . --scan-notebooks --savepath requirements.txt
uv add -r requirements.txt
```
