# Clasificación automática de tickets de soporte

Corpus: [`Tobi-Bueck/customer-support-tickets`](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) (HuggingFace). Objetivo del proyecto: dado el asunto (`subject`) y el cuerpo (`body`) de un ticket, predecir `queue` (departamento, 10 clases), `priority` (3 niveles) y `type` (4 clases).

---

## Instalación y ejecución

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
pip install -r requirements.txt
# Si se dispone de GPU NVIDIA (para E3):
# pip install torch --index-url https://download.pytorch.org/whl/cu126
```

- **E2:** abrir `src/tickets_pipeline.ipynb` y ejecutar todas las celdas en orden (con `src/` como directorio de trabajo).
- **E3:** abrir [`src/Entrega_3_NLP.ipynb`](src/Entrega_3_NLP.ipynb) y ejecutar todas las celdas en orden (funciona tanto desde `src/` como desde la raíz). Para una prueba rápida con el 20 % de los datos, 1 semilla y 2 épocas, definir `E3_FAST=1`.
- Para regenerar `src/Entrega_3_NLP.ipynb` tras editar los scripts `src/e3_parte*.py`:

```powershell
python src/build_notebook.py
```

---

## Entrega 3 (E3) · Primera iteración de modelado

Todo el código de E3 está en [`src/Entrega_3_NLP.ipynb`](src/Entrega_3_NLP.ipynb), generado con [`src/build_notebook.py`](src/build_notebook.py) a partir de los módulos `src/e3_parte1_datos_shallow.py`, `src/e3_parte2_embeddings.py` y `src/e3_parte3_deep_evaluacion.py`.

### Continuidad con E2 y corrección de fuga de datos

E3 parte directamente del dataset bruto guardado en E2 (`data/tickets/raw/dataraw.csv`, 61.765 filas) y aplica sus mismos filtros (`language != "de"` y `body` no vacío → **28.260 tickets**).

Al analizar el texto para entrenar modelos supervisados se detectó que dos de los ficheros originales de HuggingFace comparten **4.516 grupos de duplicados exactos** (9.032 tickets idénticos). En el split 80/20 de E2, el **26,2 %** de los tickets de test tenía un gemelo exacto en train. Por ello, E3:

1. Elimina duplicados y trabaja con **23.744 tickets únicos**.
2. Introduce una partición estratificada por `queue` **70 % train / 15 % validación / 15 % test** (`random_state=42`).
3. Recalcula TF-IDF, vocabulario (`<PAD>`, `<UNK>`) y Word2Vec propio exclusivamente sobre el `train` sin duplicados para evitar fuga hacia validación y test.

### Experimentos incluidos

- **Shallow learning (TF-IDF):** `Naive Bayes` (Multinomial / Complement), `Regresión logística` y `SVM lineal`, con búsqueda en rejilla sobre unigramas/bigramas, stopwords e hiperparámetros de regularización, más combinación con **12 meta-features de forma**.
- **Deep learning:** `TextCNN` (filtros 3, 4, 5) y `BiRNN` (LSTM/GRU bidireccional) comparando **5 variantes de embeddings** (Aleatorio, Google News congelado / fine-tuning, Word2Vec propio congelado / fine-tuning).
- **Atención y modelos híbridos:** comparación sin atención, producto escalar y atención general de Luong, inspección de pesos por token y concatenación de meta-features en las redes.
- **Tres tareas:** `queue` (F1 macro), `priority` (incluyendo `recall_high`) y `type`.

### Artefactos generados (`artifacts/`)

- `artifacts/resultados.csv` y `artifacts/resultados.json`: métricas y resúmenes para el informe.
- `artifacts/figuras/*.png`: 7 gráficas listas para la memoria.
- `artifacts/emb_google_news_*.npz`: caché de vectores de Google News filtrados al vocabulario.

> **Nota:** los resultados de `artifacts/` y las salidas guardadas en `src/Entrega_3_NLP.ipynb` corresponden a la ejecución completa (3 semillas y hasta 60 épocas por red; unas dos horas con una GPU NVIDIA RTX 3050). Con `E3_FAST=1` se hace una pasada rápida de prueba (20 % de los datos, 1 semilla y 2 épocas) en unos 2 minutos, pero sobrescribe `artifacts/`.

### Informe en PDF

El informe de la entrega es [`informe/E3_Gestion_de_Tickets.pdf`](informe/E3_Gestion_de_Tickets.pdf). Se genera a partir de `artifacts/resultados.json` y `artifacts/figuras/` con:

```powershell
python informe/generar_informe.py
```

El script construye `informe/E3_informe.html` y lo imprime a PDF con Microsoft Edge en modo *headless*, con el mismo formato que el informe de la E2. Las cifras salen del JSON; la redacción está en el propio script.

La guía de estudio (`informe/Guia_E3_para_entenderla.pdf`, no forma parte de la entrega) se genera igual con `python informe/generar_guia.py`.

---

## Entrega 2 (E2) · Procesamiento y análisis de datos

Todo el código y la explicación de E2 están en [`src/tickets_pipeline.ipynb`](src/tickets_pipeline.ipynb). La primera vez descarga el dataset y GloVe/Word2Vec a la caché de gensim.

### 0. Datos y limpieza (secciones 1-2 del notebook)

El dataset solo trae el split `train`, así que el train/test lo hacemos nosotros. Limpieza:

- Se eliminan los tickets con `body` vacío.
- Se eliminan los tickets en alemán (`language == "de"`). Resultado: **28.260 tickets, todos en inglés**.

**Por qué quitar el alemán:** los embeddings y las stopwords que usamos son de inglés. Con otro idioma casi todas las palabras quedarían sin vector.

**Texto de entrada:** `subject` + `body`. La columna `answer` **no se usa**: en el momento de clasificar un ticket nuevo todavía no existe respuesta, y usarla sería hacer trampa.

### 1. Tokenización y división en oraciones (sección 3)

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

### 2. Representación TF-IDF (secciones 5-6)

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

### 3. Word embeddings (secciones 7-9)

Se analizan GloVe (`glove-wiki-gigaword-100`), Word2Vec preentrenado (`word2vec-google-news-300`) y Word2Vec propio (`w2v_all`, `w2v_tickets`), con exportación de matrices de embeddings, vectores de documento y análisis OOV en `data/tickets/features/` y `data/tickets/embeddings/`.

### 4. Estructura reutilizable (sección 10)

Salida en `data/tickets/export/` (`tickets_processed.jsonl`, `manifest.json`) y `data/tickets/features/` (`corpus.jsonl`, `vocab.json`, `labels.json`, `meta.json`, `X_train.npz`, `X_test.npz`, `idx_train.npy`, `idx_test.npy`).

---

## Estructura del proyecto

```
src/
  tickets_pipeline.ipynb        # E2: limpieza, tokenización, TF-IDF, embeddings y exportación
  Entrega_3_NLP.ipynb           # E3: shallow learning, CNN/BiRNN con atención y evaluación
  e3_parte1_datos_shallow.py    # E3 bloque 0-4 (datos, deduplicación, métricas, shallow + meta)
  e3_parte2_embeddings.py       # E3 bloque 5.1-5.2 (secuencias y matrices de embeddings)
  e3_parte3_deep_evaluacion.py  # E3 bloque 5.3-7 (CNN, BiRNN, atención, prioridad/tipo, análisis)
  build_notebook.py             # Genera src/Entrega_3_NLP.ipynb desde los scripts e3_parte*.py
data/tickets/
  raw/  dataprocessed/  features/  embeddings/  export/
artifacts/
  resultados.csv  resultados.json  emb_google_news_*.npz  figuras/
informe/
  generar_informe.py            # Construye el informe de E3 (HTML -> PDF con Edge headless)
  E3_Gestion_de_Tickets.pdf     # Informe de la entrega
  generar_guia.py               # Construye la guía de estudio de la E3
  Guia_E3_para_entenderla.pdf   # Guía de estudio (no se entrega)
```

## Actualizar dependencias

```powershell
pipreqs . --scan-notebooks --savepath requirements.txt
uv add -r requirements.txt
```
