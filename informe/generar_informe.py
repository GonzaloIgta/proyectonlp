"""Genera el informe en PDF de la Entrega 3 a partir de artifacts/resultados.json y artifacts/figuras.

Uso (desde la raíz del proyecto):
    python informe/generar_informe.py

Construye un HTML (informe/E3_informe.html) y lo imprime a PDF con Microsoft Edge en modo headless.
Todas las cifras salen del JSON que guarda el notebook, así que basta con volver a ejecutar este
script tras reejecutar el notebook.
"""
import html
import json
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INF = ROOT / "informe"
FIG = ROOT / "artifacts" / "figuras"
R = json.loads((ROOT / "artifacts" / "resultados.json").read_text(encoding="utf-8"))
SALIDA_HTML = INF / "E3_informe.html"
SALIDA_PDF = INF / "E3_Gestion_de_Tickets.pdf"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")

MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre",
         "octubre", "noviembre", "diciembre"]
HOY = date.today()
FECHA = f"{HOY.day} de {MESES[HOY.month - 1]} de {HOY.year}"


# ----------------------------------------------------------------------------- utilidades
def n(x, d=3):
    """Número con coma decimal, como en las entregas anteriores."""
    return "–" if x is None else f"{x:.{d}f}".replace(".", ",")


def pct(x, d=1):
    return "–" if x is None else f"{100 * x:.{d}f}".replace(".", ",") + " %"


def miles(x):
    return f"{int(x):,}".replace(",", ".")


def e(s):
    return html.escape(str(s))


def tabla(cabecera, filas, clases=None, num_desde=1, resaltar=None):
    """Tabla HTML. `num_desde`: a partir de qué columna se alinea a la derecha. `resaltar`: índices de fila en negrita."""
    resaltar = set(resaltar or [])
    th = "".join(f'<th class="{"n" if i >= num_desde else ""}">{c}</th>' for i, c in enumerate(cabecera))
    trs = []
    for k, f in enumerate(filas):
        tds = "".join(f'<td class="{"n" if i >= num_desde else ""}">{c}</td>' for i, c in enumerate(f))
        trs.append(f'<tr class="{"best" if k in resaltar else ""}">{tds}</tr>')
    return f'<table class="{clases or ""}"><thead><tr>{th}</tr></thead><tbody>{"".join(trs)}</tbody></table>'


def figura(nombre, pie, ancho="100%"):
    ruta = (FIG / f"{nombre}.png").as_uri()
    return f'<figure><img src="{ruta}" style="width:{ancho}"><figcaption>{pie}</figcaption></figure>'


def res(tarea, modelo=None, familia=None):
    filas = [r for r in R["resultados"] if r["tarea"] == tarea]
    if modelo is not None:
        filas = [r for r in filas if r["modelo"] == modelo]
    if familia is not None:
        filas = [r for r in filas if r["familia"] == familia]
    return filas


def mejor(tarea, familia=None, por="f1"):
    return max(res(tarea, familia=familia), key=lambda r: r[por])


def f1_con_std(r):
    return n(r["f1"]) + (f' <span class="std">± {n(r["f1_std"])}</span>' if r.get("semillas", 1) > 1 else "")


def tabla_resultados(tarea, extra_high=False, filas=None):
    filas = sorted(filas if filas is not None else res(tarea), key=lambda r: -r["f1"])
    cab = ["Familia", "Modelo", "Accuracy", "Precision", "Recall", "F1 macro", "F1 pond."]
    if extra_high:
        cab.append("Recall <i>high</i>")
    cuerpo = []
    for r in filas:
        f = [e(r["familia"]), e(r["modelo"]), n(r["accuracy"]), n(r["precision"]), n(r["recall"]),
             f1_con_std(r), n(r["f1_ponderado"])]
        if extra_high:
            f.append(n(r.get("recall_high")))
        cuerpo.append(f)
    return tabla(cab, cuerpo, clases="res", num_desde=2, resaltar=[0])


# ----------------------------------------------------------------------------- datos derivados
P = R["particiones"]
FUGA = R["fuga"]
HP = R["hp_deep"]
NA = R["nombre_arq"]
ME = R["mejor_embedding"]
COB = {c["embedding"]: c for c in R["cobertura"]}
DEEP_Q = [d for d in R["deep_log"] if d["tarea"] == "queue"]
EMBS = ["Aleatorio", "Google News congelado", "Google News fine-tuning", "Word2Vec propio congelado",
        "Word2Vec propio fine-tuning"]


def deep(arq, emb, meta=False, tarea="queue"):
    nombre = f"{NA[arq]}{' + meta-features' if meta else ''} · {emb}"
    return next((d for d in R["deep_log"] if d["tarea"] == tarea and d["modelo"] == nombre), None)


BEST_Q = mejor("queue")
BEST_Q_SH = mejor("queue", "Shallow")
BEST_Q_DEEP = mejor("queue", "Deep")
BASE_Q = res("queue", familia="Baseline")[0]
MS = R["mejor_shallow"]

# ----------------------------------------------------------------------------- textos interpretativos
# Se rellenan a mano tras leer los resultados definitivos (ver sección "Conclusiones").
from textos_informe import TEXTOS  # noqa: E402  (fichero hermano con la redacción de cada apartado)

T = TEXTOS(R, n=n, pct=pct, miles=miles)

# ----------------------------------------------------------------------------- HTML
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Montserrat:wght@600;700&display=block');
:root { --azul: #2347b5; --tinta: #1d1d1f; --tinta2: #4a4a4f; --linea: #c8c8cc; --cab: #d9d9d9; }
@page { size: A4; margin: 20mm 19mm 20mm 19mm;
        @bottom-right { content: counter(page); font: 9pt 'Inter', 'Segoe UI', sans-serif; color: #77777c; } }
@page portada { margin: 0; @bottom-right { content: none; } }
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: 'Inter', 'Segoe UI', sans-serif; font-size: 10pt; line-height: 1.5; color: var(--tinta); margin: 0; }
.portada { page: portada; height: 297mm; width: 210mm; position: relative; break-after: page; }
.portada img.logo { position: absolute; top: 14mm; left: 50%; transform: translateX(-50%); width: 58mm; }
.portada .titulo { position: absolute; top: 112mm; width: 100%; text-align: center; }
.portada h1 { font-family: 'Montserrat', 'Segoe UI', sans-serif; font-weight: 700; font-size: 34pt; margin: 0; color: #13182a; }
.portada .sub { margin-top: 5mm; font-size: 13pt; color: var(--tinta2); }
.portada .sub2 { margin-top: 2mm; font-size: 10.5pt; color: #77777c; }
.portada .pie { position: absolute; left: 25mm; bottom: 26mm; }
.portada .pie h3 { font-family: 'Montserrat', 'Segoe UI', sans-serif; color: var(--azul); font-size: 15pt; margin: 7mm 0 2mm; }
.portada .pie p { margin: 0 0 3mm; font-size: 11pt; color: var(--tinta2); }
h2 { font-family: 'Montserrat', 'Segoe UI', sans-serif; color: var(--azul); font-size: 17pt; margin: 0 0 3mm; line-height: 1.25;
     break-after: avoid; }
h2.nueva { break-before: page; }
h3 { font-family: 'Inter', 'Segoe UI', sans-serif; font-weight: 600; color: var(--azul); font-size: 12pt; margin: 6mm 0 2mm;
     break-after: avoid; }
p { margin: 0 0 2.6mm; text-align: left; }
ul, ol { margin: 0 0 2.6mm; padding-left: 6mm; }
li { margin-bottom: 1mm; }
code { font-family: Consolas, monospace; font-size: 8.8pt; background: #f1f1f3; padding: 0 1mm; border-radius: 1mm; }
table { border-collapse: collapse; width: auto; min-width: 55%; margin: 2mm 0 4mm; font-size: 8.8pt; break-inside: avoid; }
table.res, table.ancha { width: 100%; }
table.emb td:first-child { white-space: nowrap; }
th, td { border: 0.6pt solid #8a8a8f; padding: 1.4mm 2mm; vertical-align: top; text-align: left; }
th { background: var(--cab); font-weight: 600; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
tr.best td { font-weight: 600; }
.std { color: #77777c; font-weight: 400; font-size: 8pt; }
table.res td:nth-child(2) { width: 38%; }
figure { margin: 3mm 0 5mm; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; }
figcaption { font-size: 8.6pt; color: var(--tinta2); margin-top: 1.5mm; text-align: center; }
.nota { border-left: 1mm solid var(--azul); background: #f3f5fc; padding: 2.4mm 3.5mm; margin: 3mm 0 4mm; break-inside: avoid; }
.nota p:last-child { margin-bottom: 0; }
.formula { text-align: center; font-family: 'Cambria Math', 'Cambria', serif; font-size: 11pt; margin: 2mm 0 3mm; }
.kpis { display: flex; gap: 4mm; margin: 3mm 0 5mm; break-inside: avoid; }
.kpi { flex: 1; border: 0.6pt solid var(--linea); border-radius: 2mm; padding: 3mm 3.5mm; }
.kpi .v { font-family: 'Montserrat', 'Segoe UI', sans-serif; font-size: 18pt; font-weight: 700; color: var(--tinta); line-height: 1.1; }
.kpi .l { font-size: 8.4pt; color: var(--tinta2); margin-top: 1mm; }
.bloque { break-inside: avoid; }
.refs li { font-size: 8.5pt; margin-bottom: 0.8mm; }
"""

secciones = []

# Portada
secciones.append(f"""
<section class="portada">
  <img class="logo" src="{(INF / 'deusto.png').as_uri()}">
  <div class="titulo">
    <h1>Gestión de Tickets</h1>
    <div class="sub">E3 · Primera iteración de resolución del problema</div>
    <div class="sub2">Fundamentos del Procesamiento de Lenguaje Natural</div>
  </div>
  <div class="pie">
    <h3>AUTORES</h3><p>David Trueba</p><p>Gonzalo Iglesias</p>
    <h3>FECHA</h3><p>{FECHA}</p>
  </div>
</section>""")

# Resumen
resumen_codigo = tabla(
    ["Apartado del notebook", "Qué hace"],
    [["0. Configuración", "Semillas, rutas, estilo de las gráficas y modo rápido de prueba (<code>E3_FAST=1</code>)"],
     ["1. Carga y limpieza", "Lee el CSV bruto de E2, aplica sus filtros y <b>elimina los tickets duplicados</b>"],
     ["2. Preprocesado y partición", "Tokenización de E2 y nueva partición train / validación / test (70 / 15 / 15)"],
     ["3. Métricas", "Accuracy, precision, recall y F1 (macro y ponderado); recall de la prioridad <i>high</i>"],
     ["4. Shallow learning", "TF-IDF + Naive Bayes, Regresión logística y SVM lineal con búsqueda en rejilla; meta-features"],
     ["5. Deep learning", "CNN (Kim, 2014) y RNN bidireccional con atención; 5 variantes de embeddings; análisis de la atención"],
     ["6. Prioridad y tipo", "Las otras dos salidas planteadas en E1"],
     ["7. Comparativa y errores", "Tabla final, matriz de confusión y confusiones más frecuentes"],
     ["8. Conclusiones", "Resumen de resultados y siguientes pasos"]],
    clases="ancha", num_desde=99)

secciones.append(f"""
<h2>Resumen de la entrega</h2>
<p>En E1 planteamos un sistema que, a partir del asunto y el cuerpo de un ticket de soporte, prediga el
departamento al que va (<b>cola</b>, 10 clases), su <b>prioridad</b> (3 niveles) y su <b>tipo</b> (4 clases).
En E2 preparamos el texto (tokenización, vocabulario, TF-IDF y Word2Vec). En esta entrega entrenamos y
comparamos los primeros modelos: tres clasificadores clásicos sobre TF-IDF y dos redes neuronales (una CNN y
una RNN bidireccional con atención) con cinco formas distintas de inicializar los embeddings.</p>
<div class="kpis">
  <div class="kpi"><div class="v">{miles(R['tickets_unicos'])}</div><div class="l">tickets únicos tras quitar {miles(R['tickets_e2'] - R['tickets_unicos'])} duplicados</div></div>
  <div class="kpi"><div class="v">{len(R['resultados'])}</div><div class="l">modelos evaluados en test (3 tareas)</div></div>
  <div class="kpi"><div class="v">{n(BEST_Q['f1'])}</div><div class="l">mejor F1 macro en la cola · {e(BEST_Q['modelo'])}</div></div>
  <div class="kpi"><div class="v">{n(BASE_Q['f1'])}</div><div class="l">F1 macro de la clase mayoritaria (referencia)</div></div>
</div>
{T['resumen']}
<h3>Estructura del código</h3>
<p>Todo el código está en <code>src/Entrega_3_NLP.ipynb</code>, que se genera con <code>src/build_notebook.py</code>
a partir de tres scripts (<code>src/e3_parte1_datos_shallow.py</code>, <code>e3_parte2_embeddings.py</code> y
<code>e3_parte3_deep_evaluacion.py</code>). El notebook se ejecuta de principio a fin en unas dos horas con una
GPU NVIDIA RTX 3050 y guarda las métricas en <code>artifacts/resultados.json</code> y las figuras en
<code>artifacts/figuras/</code>, que son las que usa este informe.</p>
{resumen_codigo}
""")

# 1. Datos
fuga_tabla = tabla(["Partición", "Tickets", "Test con gemelo en train", "Accuracy", "F1 macro"],
                   [[e(f["partición"]), miles(f["tickets"]), e(f["test con gemelo en train"]).replace(".", ","),
                     n(f["accuracy"]), n(f["f1 macro"])] for f in FUGA])
part_tabla = tabla(["Conjunto", "Uso", "Tickets"],
                   [["train", "Ajusta los modelos, el vocabulario, el TF-IDF, el escalador y el Word2Vec propio", miles(P["train"])],
                    ["validación", "Elige hiperparámetros y decide cuándo parar de entrenar (<i>early stopping</i>)", miles(P["val"])],
                    ["test", "Solo se usa una vez, para las tablas finales", miles(P["test"])]], clases="ancha", num_desde=2)
secciones.append(f"""
<h2 class="nueva">1. Datos: continuidad con E2 y una fuga que corregir</h2>
<p>Partimos del mismo dataset que en E1 y E2, <code>Tobi-Bueck/customer-support-tickets</code>, y de los mismos
filtros: se quitan los tickets en alemán y los de cuerpo vacío, lo que deja {miles(R['tickets_e2'])} tickets en
inglés. La entrada de los modelos es el asunto más el cuerpo; la respuesta del agente no se usa para clasificar,
porque no existe cuando llega el ticket.</p>
<h3>1.1 Tickets duplicados entre los ficheros</h3>
<p>Al preparar los modelos supervisados descubrimos algo que no vimos en E2: dos de los tres CSV del dataset son
versiones distintas del mismo conjunto y comparten <b>{miles(R['tickets_e2'] - R['tickets_unicos'])} tickets idénticos</b>
(con las mismas etiquetas). Si un ticket cae en train y su gemelo en test, el modelo no necesita generalizar: le basta
con haberlo memorizado. Es una fuga de información que infla las métricas. Lo medimos con el mismo clasificador
(TF-IDF + SVM lineal) sobre la partición 80/20 de E2 y sobre los datos sin duplicados:</p>
{fuga_tabla}
<p>Con duplicados, uno de cada cuatro tickets de test ya estaba en train y la accuracy sube
{n(100 * (FUGA[0]['accuracy'] - FUGA[1]['accuracy']), 1)} puntos sin que el modelo sea mejor. Por eso
<b>en E3 trabajamos solo con los {miles(R['tickets_unicos'])} tickets únicos</b> y recalculamos el TF-IDF, el
vocabulario y el Word2Vec propio sobre el nuevo train, en lugar de reutilizar las matrices guardadas en E2.</p>
<h3>1.2 Partición train / validación / test</h3>
<p>En E2 solo había train y test. Ahora hace falta un tercer conjunto, validación, porque vamos a elegir
hiperparámetros y a decidir cuándo parar de entrenar las redes: si esas decisiones se tomaran mirando test, test
dejaría de ser una estimación honesta. Seguimos el esquema de la Unidad 3 (70 / 15 / 15, estratificado por cola):</p>
{part_tabla}
<h3>1.3 Etiquetas y métricas</h3>
{figura('01_distribucion_clases', 'Figura 1. Distribución de las tres etiquetas en los tickets únicos.', '80%')}
<p>La cola está muy desbalanceada: <i>Technical Support</i> tiene unas veinte veces más tickets que <i>General
Inquiry</i>. Por eso la métrica que decide qué modelo es mejor es el <b>F1 macro</b>, que da el mismo peso a cada
clase. Damos las cuatro métricas del enunciado (accuracy, y precision, recall y F1 promediadas por clase) más el F1
ponderado como referencia. En la prioridad medimos además el <b>recall de la clase <i>high</i></b>, la crítica según
E1: un ticket urgente clasificado como no urgente es el error más caro.</p>
""")

# 2. Shallow
tf = R["tfidf_features"]
tf_tabla = tabla(["Configuración del TF-IDF", "Features"], [[e(k), miles(v)] for k, v in tf.items()])
grid_tabla = tabla(["Modelo", "Hiperparámetro", "Valores probados"],
                   [["todos", "TF-IDF", "unigramas / uni+bigramas, con o sin stopwords"],
                    ["Naive Bayes", "variante", "Multinomial, Complement"],
                    ["Naive Bayes", "<code>alpha</code> (suavizado)", "0,0003 · 0,001 · 0,003 · 0,01 · 0,03 · 0,1 · 0,3 · 1"],
                    ["Regresión logística", "<code>C</code> (inversa de la regularización)", "1 · 10 · 100 · 1000"],
                    ["SVM lineal", "<code>C</code>", "0,1 · 0,3 · 1 · 3 · 10 · 30 · 100"],
                    ["Regresión logística y SVM", "<code>class_weight</code>", "ninguno, <code>balanced</code>"]],
                   clases="ancha", num_desde=99)


def params_txt(p):
    return ", ".join(f"{k} = {('ninguno' if v is None else str(v).replace('.', ','))}" for k, v in p.items())


mejores_sh = tabla(["Modelo", "Mejor combinación", "F1 macro (validación)"],
                   [[e(k), f"{e(v['tfidf'])}; {e(params_txt(v['params']))}", n(v["f1_val"])] for k, v in MS["queue"].items()],
                   num_desde=2)
filas_sh = [r for r in res("queue") if r["familia"] in ("Shallow", "Baseline")]
meta_lista = ", ".join(f"<code>{m}</code>" for m in R["meta_features"])
secciones.append(f"""
<h2 class="nueva">2. Shallow learning con representación dispersa</h2>
<p><b>TF-IDF</b> convierte cada ticket en un vector tan largo como el vocabulario, casi todo ceros (por eso es una
representación <i>dispersa</i>): el peso de una palabra sube si aparece mucho en el ticket y baja si aparece en casi
todos. No tiene en cuenta el orden de las palabras salvo el que capturan los bigramas. Igual que en E2, el
vectorizador se ajusta solo con train.</p>
{tf_tabla}
<p>Sobre esos vectores entrenamos los tres clasificadores clásicos vistos en clase (Unidad 3):</p>
<ul>
<li><b>Naive Bayes</b>: aplica Bayes suponiendo que las palabras son independientes dada la clase. Es el baseline
clásico de texto, muy rápido. Probamos la variante <i>Multinomial</i> y la <i>Complement</i>, pensada para clases
desbalanceadas.</li>
<li><b>Regresión logística</b>: aprende un peso por palabra y clase y da probabilidades con softmax.</li>
<li><b>SVM lineal</b>: busca el hiperplano que separa las clases con el mayor margen; suele ser el mejor modelo
lineal en espacios dispersos de muchas dimensiones.</li>
</ul>
<p>Como referencia mínima incluimos un clasificador trivial que siempre responde la clase más frecuente.</p>
<h3>2.1 Optimización de hiperparámetros</h3>
<p>Hacemos una <b>búsqueda en rejilla</b>: se entrena cada combinación con train y se mide su F1 macro en
validación; gana la que más saca y test no interviene. Los rangos de <code>C</code> y <code>alpha</code> se
ampliaron dos veces, porque en las primeras pruebas el mejor valor caía en el extremo de la rejilla: si el óptimo
está en el borde, no sabemos si seguiría mejorando. Con los rangos finales, los tres óptimos quedan en el interior.</p>
{grid_tabla}
{mejores_sh}
{figura('02_grid_shallow', 'Figura 2. F1 macro en validación según la regularización C y la configuración del TF-IDF (para cada C, el mejor <i>class_weight</i>).')}
{T['grid_shallow']}
<h3>2.2 Combinar el texto con features no textuales</h3>
<p>El enunciado valora combinar distintos tipos de features. El dataset solo trae texto y etiquetas, así que
calculamos {len(R['meta_features'])} <b>meta-features de forma</b>, que describen cómo está escrito el ticket y no
qué palabras usa: {meta_lista}. Los conteos se pasan a escala logarítmica y se estandarizan con la media y la
desviación de train. No usamos como entrada <code>type</code>, <code>priority</code> ni los <code>tag_*</code>:
son etiquetas que pone el agente después de leer el ticket y usarlas sería otra fuga.</p>
<p>Para combinarlas con el TF-IDF se concatenan a su vector, multiplicadas por un peso <code>w</code> que también se
elige en validación (el TF-IDF está normalizado y las meta-features no).</p>
<h3>2.3 Resultados en test</h3>
{tabla_resultados('queue', filas=filas_sh)}
{T['shallow']}
""")

# 3. Deep
cob_tabla = tabla(["Embedding", "Palabras del vocabulario con vector", "Tokens de test con vector"],
                  [[e(k), pct(v["tipos del vocabulario con vector"]), pct(v["tokens de test con vector"])] for k, v in COB.items()])
emb_tabla = tabla(["Variante", "Inicialización", "¿Se entrena?"],
                  [["Aleatorio", "Vectores al azar, uniformes en [−0,25; 0,25]", "Sí, desde cero"],
                   ["Google News congelado", "Word2Vec preentrenado con noticias (300 dim.)", "No"],
                   ["Google News fine-tuning", "Word2Vec preentrenado con noticias (300 dim.)", "Sí"],
                   ["Word2Vec propio congelado", "Skip-gram entrenado con tickets y respuestas de train", "No"],
                   ["Word2Vec propio fine-tuning", "Skip-gram entrenado con tickets y respuestas de train", "Sí"]], num_desde=99)
bq = R["busqueda_deep"]
cnn_grid = tabla(["Filtros", "Dropout", "Pesos por clase", "F1 val.", "Mejor época"],
                 [[b["n_filtros"], n(b["dropout"], 1), "sí" if b["pesos_clase"] else "no", n(b["f1_val"]), f"{b['mejor_epoca']:.0f}"]
                  for b in bq["CNN"]], num_desde=0, resaltar=[0])
rnn_grid = tabla(["Celda", "Estado oculto", "F1 val.", "Mejor época", "Tiempo (s)"],
                 [[b["celda"], b["oculto"], n(b["f1_val"]), f"{b['mejor_epoca']:.0f}", f"{b['segundos']:.0f}"]
                  for b in bq["RNN"]], num_desde=1, resaltar=[0])
filas_emb = []
for emb in EMBS:
    fila = [e(emb)]
    for arq in ("CNN", "RNN"):
        d = deep(arq, emb)
        fila += [n(d["f1_val"]), f'{n(d["f1_test"])} <span class="std">± {n(d["f1_test_std"])}</span>', f"{d['mejor_epoca']:.0f}"]
    filas_emb.append(fila)
emb_res = tabla(["Embedding", f"{NA['CNN']}: F1 val.", "F1 test", "Épocas",
                 f"{e(NA['RNN'])}: F1 val.", "F1 test", "Épocas"], filas_emb, clases="ancha emb", num_desde=1)
AT = R["atencion_tabla"]
att_tabla = tabla(["Variante", "Accuracy", "F1 macro", "Peso en el primer y último token", "Peso del token más atendido"],
                  [[e(a["variante"]), n(a["accuracy"]), f'{n(a["f1"])} <span class="std">± {n(a["f1_std"])}</span>',
                    pct(a.get("atención en primer y último token")), pct(a.get("peso del token más atendido"))] for a in AT],
                  clases="ancha")
top_cols = [k for k in R["atencion_top"][0] if k != "cola"]
att_top = tabla(["Cola"] + [e(c.capitalize()) for c in top_cols],
                [[e(t["cola"])] + [e(t[c]) for c in top_cols] for t in R["atencion_top"]], clases="ancha", num_desde=99)
filas_meta = [r for r in res("queue") if "meta-features" in r["modelo"]] + \
             [r for r in res("queue") if r["modelo"] in (f"{NA[a]} · {ME[a]}" for a in ("CNN", "RNN"))] + \
             [r for r in res("queue") if r["modelo"] == BEST_Q_SH["modelo"]]
secciones.append(f"""
<h2 class="nueva">3. Deep learning: CNN y RNN con atención</h2>
<p>Los modelos anteriores tratan el ticket como una bolsa de palabras. Las redes de este apartado lo leen como una
<b>secuencia</b>: cada token se sustituye por un vector denso de 300 dimensiones (<i>embedding</i>) y la red aprende
a combinar esos vectores teniendo en cuenta el orden. Las implementamos en PyTorch.</p>
<h3>3.1 Vocabulario y secuencias</h3>
<p>Usamos la clase <code>Vocabulary</code> de E2 con dos tokens reservados, <code>&lt;PAD&gt;</code> (relleno, índice 0)
y <code>&lt;UNK&gt;</code> (palabras desconocidas). Se construye solo con train y descarta las palabras que aparecen
menos de dos veces, lo que deja {miles(R['vocabulario'])} tokens. Cada ticket pasa a ser una lista de índices
recortada o rellenada hasta 200 tokens (el 99 % de los tickets tiene menos de 180). Como en la Unidad 3, el relleno se
gestiona con <i>packed sequences</i> en la RNN y con una máscara en la atención, para que la red no lo procese.</p>
<h3>3.2 Embeddings: tres orígenes, congelados o con fine-tuning</h3>
<p>El enunciado pide comparar tipos de embeddings y el efecto del <i>fine-tuning</i>. Probamos cinco variantes:</p>
{emb_tabla}
<p><b>Congelado</b> significa que los vectores no cambian durante el entrenamiento y la red solo aprende a usarlos;
con <b>fine-tuning</b> también se actualizan, así que pueden adaptarse a la tarea, a cambio de más parámetros y más
riesgo de sobreajuste. El Word2Vec propio sigue el esquema de <code>w2v_all</code> de E2 (skip-gram, ventana 5,
tickets y respuestas) pero reentrenado solo con el train sin duplicados. Para Google News mantenemos los
<i>fallbacks</i> de E2 (mayúscula inicial, todo en mayúsculas y sin el posesivo <code>'s</code>); las palabras sin
vector reciben uno aleatorio.</p>
{cob_tabla}
{T['cobertura']}
<h3>3.3 Arquitecturas</h3>
<p><b>CNN para texto (Kim, 2014).</b> Es la arquitectura de la Unidad 3. Filtros de convolución de anchura 3, 4 y 5
recorren el ticket y cada uno se activa cuando ve un patrón de 3, 4 o 5 palabras seguidas (un n-grama aprendido). De
cada filtro se toma solo su activación máxima (<i>max-over-time pooling</i>): importa si el patrón aparece, no dónde.
Las activaciones se concatenan y pasan por una capa densa con dropout y softmax. En el artículo original nuestras
variantes se llaman <i>CNN-rand</i>, <i>CNN-static</i> y <i>CNN-non-static</i>.</p>
<p><b>RNN bidireccional (LSTM o GRU).</b> La red lee el ticket palabra a palabra manteniendo una memoria y, al ser
bidireccional, otra lo lee de derecha a izquierda, de modo que cada palabra queda representada con su contexto por
ambos lados. LSTM y GRU usan puertas para decidir qué recordar y qué olvidar, lo que evita el desvanecimiento del
gradiente de la RNN básica. Según la regla vista en clase la GRU suele ir mejor con pocos datos, así que el tipo de
celda es un hiperparámetro más.</p>
<p><b>Atención.</b> Sin atención, todo el ticket se resume en el último estado oculto, que hace de cuello de botella.
Con atención la red vuelve a mirar todos los estados. Seguimos el cálculo de la Unidad 3, con una adaptación: al
clasificar no hay decoder, así que el «estado actual» <i>q</i> es el estado final de la RNN (sus dos direcciones
concatenadas).</p>
<ol>
<li>Puntuación de cada estado oculto con el estado actual (producto escalar): <span class="formula">s<sub>t</sub> = h<sub>t</sub><sup>⊤</sup> q</span></li>
<li>Softmax de las puntuaciones para obtener los pesos de atención: <span class="formula">α<sub>t</sub> = e<sup>s<sub>t</sub></sup> / Σ<sub>k</sub> e<sup>s<sub>k</sub></sup></span></li>
<li>Suma de los estados ponderada por su peso (vector de contexto): <span class="formula">c = Σ<sub>t</sub> α<sub>t</sub> h<sub>t</sub></span></li>
<li>El contexto se concatena con el estado actual, [<i>c</i>; <i>q</i>], y eso entra en la capa de salida.</li>
</ol>
<div class="nota"><p><b>Modelo adicional no visto exactamente en clase: atención «general» (Luong, Pham y Manning, 2015).</b>
Es la misma atención con una matriz aprendida entre los dos vectores, <i>s<sub>t</sub></i> = <i>h<sub>t</sub></i><sup>⊤</sup>
<i>W q</i>. La incluimos porque el producto escalar solo puede comparar <i>h<sub>t</sub></i> y <i>q</i> tal cual, mientras que
<i>W</i> deja que la red aprenda qué buscar en cada estado. En el apartado 3.7 comparamos las dos.</p></div>
<p>Las dos redes admiten además una entrada opcional con las meta-features del apartado 2.2, que pasan por una capa
densa pequeña y se concatenan a la representación del texto antes de la capa de salida.</p>
<h3>3.4 Entrenamiento</h3>
<p>Todas las redes se entrenan igual, para que lo único que cambie entre experimentos sea lo que queremos comparar:
entropía cruzada categórica, optimizador Adam (tasa {n(R['lr'], 3)}), lotes de {R['tam_lote']} tickets de longitud
parecida, dropout antes de la capa de salida y <b><i>early stopping</i></b>: tras cada época se mide el F1 macro en
validación y, si pasan {R['paciencia']} épocas sin mejorar, se para y se recupera el mejor modelo (máximo
{R['max_epocas']} épocas). Cada configuración final se entrena con <b>{len(R['semillas'])} semillas</b> y damos la
media y la desviación típica: una red cambia de un entrenamiento a otro por la inicialización aleatoria, así que una
diferencia menor que esa desviación no es significativa.</p>
<h3>3.5 Búsqueda de hiperparámetros</h3>
<p>Igual que con los modelos clásicos, una rejilla evaluada en validación. Para no mezclarla con la comparación de
embeddings se hace con una sola variante (Google News congelado) y una semilla, y los valores elegidos se mantienen
fijos después. La RNN tarda bastante más en entrenar, así que su rejilla es más pequeña: hereda de la CNN el dropout y
la decisión sobre los pesos por clase, que dependen sobre todo del desbalanceo de los datos. La tasa de aprendizaje,
el tamaño de lote y las anchuras de filtro (3, 4, 5) se dejan en los valores habituales de la literatura.</p>
<div class="bloque"><p><b>CNN</b> (ordenada por F1 de validación; en negrita, la elegida):</p>{cnn_grid}</div>
<div class="bloque"><p><b>RNN bidireccional con atención</b> (dropout {n(HP['RNN']['dropout'], 1)} y pesos por clase
{'sí' if HP['RNN']['pesos_clase'] else 'no'}, heredados de la CNN):</p>{rnn_grid}</div>
{T['busqueda_deep']}
<h3>3.6 Comparación de embeddings</h3>
<p>Con los hiperparámetros fijados, cada arquitectura se entrena con las cinco variantes de embedding y tres
semillas. Es el experimento central de la entrega.</p>
{emb_res}
{figura('03_comparacion_embeddings', 'Figura 3. F1 macro en test de cada combinación de arquitectura y embedding (media de 3 semillas; la línea fina es la desviación típica). La línea vertical marca el mejor modelo shallow.', '92%')}
{T['embeddings']}
{figura('04_curvas_aprendizaje', 'Figura 4. Curvas de aprendizaje de la CNN (primera semilla) con tres de las variantes de embedding.')}
{T['curvas']}
<h3>3.7 ¿Qué aporta la atención?</h3>
<p>Entrenamos la misma RNN, con su mejor embedding ({e(ME['RNN'])}), de tres formas: sin atención (solo el estado
final), con la atención de producto escalar de clase y con la variante general. Además del F1 medimos <b>dónde cae la
atención</b>: qué parte del peso total se llevan el primer y el último token del ticket.</p>
{att_tabla}
{T['atencion']}
{figura('05_atencion_ejemplo', 'Figura 5. Los diez tokens con más peso de atención en un mismo ticket de test, con cada variante (entre paréntesis, la posición del token).')}
<p>Agregando los pesos sobre todo test, estas son las palabras que más atención reciben en los tickets asignados a
cada cola (media del peso relativo; solo palabras que aparecen al menos 15 veces):</p>
{att_top}
{T['atencion_palabras']}
<h3>3.8 Red híbrida: texto + meta-features</h3>
<p>La misma idea del apartado 2.2 dentro de la red, con la mejor configuración de cada arquitectura:</p>
{tabla_resultados('queue', filas=filas_meta)}
{T['meta_deep']}
""")

# 4. Prioridad y tipo
secciones.append(f"""
<h2 style="margin-top:8mm">4. Las otras dos salidas: prioridad y tipo</h2>
<p>Repetimos el proceso para las otras dos etiquetas de E1. Los modelos clásicos repiten su búsqueda en rejilla
completa (es barata). Las redes reutilizan los hiperparámetros y el mejor embedding encontrados para la cola;
optimizarlos para cada tarea queda para la siguiente iteración.</p>
<h3>4.1 Prioridad (3 clases)</h3>
{tabla_resultados('priority', extra_high=True)}
{T['prioridad']}
<h3>4.2 Tipo (4 clases)</h3>
{tabla_resultados('type')}
{T['tipo']}
""")

# 5. Comparativa y errores
INF_CL = [c for c in R["informe_clases"] if c["clase"] not in ("accuracy", "macro avg", "weighted avg")]
cl_tabla = tabla(["Cola", "Precision", "Recall", "F1", "Tickets en test"],
                 [[e(c["clase"]), n(c["precision"]), n(c["recall"]), n(c["f1-score"]), miles(c["support"])]
                  for c in sorted(INF_CL, key=lambda c: -c["f1-score"])])
conf_tabla = tabla(["Cola real", "Cola predicha", "Tickets"],
                   [[e(c["real"]), e(c["predicha"]), miles(c["tickets"])] for c in R["confusiones"]], num_desde=2)
secciones.append(f"""
<h2 class="nueva">5. Tabla comparativa final (cola)</h2>
<p>Todos los modelos de la tarea principal, ordenados por F1 macro en test. En los modelos deep, cada cifra es la media
de {len(R['semillas'])} semillas (en gris, la desviación típica).</p>
{tabla_resultados('queue')}
{figura('06_ranking_cola', 'Figura 6. F1 macro en test de todos los modelos de la cola. Azul: shallow; naranja: deep; gris: referencia.', '94%')}
{T['comparativa']}
<h2 class="nueva">6. Análisis de errores</h2>
<p>Analizamos el mejor modelo de la cola según validación, <b>{e(R['mejor_modelo_cola'])}</b>.</p>
{figura('07_matriz_confusion', 'Figura 7. Matriz de confusión normalizada por filas: cada fila muestra a qué colas se asignan los tickets de una cola real.', '80%')}
<div class="bloque">{cl_tabla}</div>
<div class="bloque"><p>Las confusiones más frecuentes:</p>{conf_tabla}</div>
{T['errores']}
""")

# 7. Conclusiones y referencias
secciones.append(f"""
<h2 class="nueva">7. Conclusiones y siguientes pasos</h2>
{T['conclusiones']}
<h3>Siguientes pasos</h3>
{T['siguientes']}
<h3>Referencias</h3>
<ul class="refs">
<li>Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. <i>EMNLP</i>.</li>
<li>Bahdanau, D., Cho, K. y Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. <i>ICLR</i>.</li>
<li>Luong, M.-T., Pham, H. y Manning, C. D. (2015). Effective Approaches to Attention-based Neural Machine Translation. <i>EMNLP</i>.</li>
<li>Mikolov, T. et al. (2013). Distributed Representations of Words and Phrases and their Compositionality. <i>NeurIPS</i>.</li>
<li>Hochreiter, S. y Schmidhuber, J. (1997). Long Short-Term Memory. <i>Neural Computation</i>, 9(8).</li>
<li>Cho, K. et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. <i>EMNLP</i>.</li>
<li>Rennie, J. et al. (2003). Tackling the Poor Assumptions of Naive Bayes Text Classifiers. <i>ICML</i> (Complement Naive Bayes).</li>
<li>Zangari, A. et al. (2023). Ticket automation: An insight into current research with applications to multi-level classification scenarios. <i>Expert Systems with Applications</i>.</li>
<li>Dataset: Bueck, T. <i>Customer Support Tickets</i>. Hugging Face, <code>Tobi-Bueck/customer-support-tickets</code>.</li>
<li>Apuntes de la asignatura, Unidades 2 (embeddings) y 3 (shallow learning, CNN, RNN y mecanismos de atención). Librerías: scikit-learn, PyTorch y gensim.</li>
</ul>
""")

documento = f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<title>E3 · Gestión de Tickets</title><style>{CSS}</style></head><body>{''.join(secciones)}</body></html>"""
SALIDA_HTML.write_text(documento, encoding="utf-8")
print("HTML:", SALIDA_HTML)

if "--solo-html" in sys.argv:
    sys.exit(0)
with tempfile.TemporaryDirectory() as perfil:
    subprocess.run([str(EDGE), "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={perfil}",
                    "--virtual-time-budget=20000", f"--print-to-pdf={SALIDA_PDF}", SALIDA_HTML.as_uri()],
                   check=True, timeout=180)
print("PDF:", SALIDA_PDF)
