"""Genera el informe en PDF de la Entrega 3 con el mismo formato que el de la E2.

Uso (desde la raíz del proyecto):
    python informe/generar_informe.py

Lee artifacts/resultados.json y artifacts/figuras/, construye informe/E3_informe.html y lo imprime a
PDF con Microsoft Edge en modo headless. Las cifras salen del JSON; si se reejecuta el notebook y
cambia el sentido de algún resultado, hay que revisar la redacción de este fichero.
"""
import html
import json
import subprocess
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
    return "–" if x is None else f"{x:.{d}f}".replace(".", ",")


def pct(x, d=1):
    return "–" if x is None else f"{100 * x:.{d}f}".replace(".", ",") + " %"


def miles(x):
    return f"{int(x):,}".replace(",", ".")


def e(s):
    return html.escape(str(s))


def tabla(cab, filas, num=(), negrita=(), clase=""):
    """Tabla como las de E2: cabecera gris centrada, números a la derecha."""
    th = "".join(f"<th>{c}</th>" for c in cab)
    trs = []
    for k, f in enumerate(filas):
        tds = "".join(f'<td class="{"n" if i in num else ""}">{c}</td>' for i, c in enumerate(f))
        trs.append(f'<tr class="{"b" if k in negrita else ""}">{tds}</tr>')
    return f'<table class="{clase}"><thead><tr>{th}</tr></thead><tbody>{"".join(trs)}</tbody></table>'


def lista(items):
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def figura(nombre, pie, ancho="100%"):
    return (f'<figure><img src="{(FIG / f"{nombre}.png").as_uri()}" style="width:{ancho}">'
            f'<figcaption>{pie}</figcaption></figure>')


def h1(texto, intro):
    return f'<div class="sec"><h1>{texto}</h1><p class="intro">{intro}</p></div>'


def res(tarea, modelo):
    return next(r for r in R["resultados"] if r["tarea"] == tarea and r["modelo"] == modelo)


def mejor(tarea, cond=lambda r: True, por="f1"):
    return max((r for r in R["resultados"] if r["tarea"] == tarea and cond(r)), key=lambda r: r[por])


def deep(modelo, tarea="queue"):
    return next(d for d in R["deep_log"] if d["tarea"] == tarea and d["modelo"] == modelo)


def f1s(r):
    return n(r["f1"]) + (f' <span class="std">± {n(r["f1_std"])}</span>' if r.get("semillas", 1) > 1 else "")


def tabla_res(tarea, filas=None, high=False):
    filas = sorted(filas if filas is not None else [r for r in R["resultados"] if r["tarea"] == tarea],
                   key=lambda r: -r["f1"])
    cab = ["Familia", "Modelo", "Accuracy", "Precision", "Recall", "F1 macro", "F1 pond."] + (["Recall high"] if high else [])
    cuerpo = [[e(r["familia"]), e(r["modelo"]), n(r["accuracy"]), n(r["precision"]), n(r["recall"]), f1s(r),
               n(r["f1_ponderado"])] + ([n(r.get("recall_high"))] if high else []) for r in filas]
    return tabla(cab, cuerpo, num=tuple(range(2, len(cab))), negrita=(0,), clase="res")


# ----------------------------------------------------------------------------- cifras
P, FUGA, NA, ME, HP = R["particiones"], R["fuga"], R["nombre_arq"], R["mejor_embedding"], R["hp_deep"]
MS = R["mejor_shallow"]
COB = {c["embedding"]: c for c in R["cobertura"]}
SVM, LR, NB = res("queue", "TF-IDF + SVM lineal"), res("queue", "TF-IDF + Regresión logística"), res("queue", "TF-IDF + Naive Bayes")
BASE = mejor("queue", lambda r: r["familia"] == "Baseline")
META_SOLA = res("queue", "Solo meta-features + Regresión logística")
META_MAS = mejor("queue", lambda r: r["modelo"].startswith("TF-IDF + meta-features"))
RNN = NA["RNN"].split()[0]                      # BiGRU o BiLSTM
EMBS = ["Aleatorio", "Google News congelado", "Google News fine-tuning", "Word2Vec propio congelado",
        "Word2Vec propio fine-tuning"]
C = {emb: deep(f"{NA['CNN']} · {emb}") for emb in EMBS}
Rn = {emb: deep(f"{NA['RNN']} · {emb}") for emb in EMBS}
CNN_Q = mejor("queue", lambda r: r["modelo"].startswith("CNN") and "meta" not in r["modelo"])
RNN_Q = mejor("queue", lambda r: r["modelo"].startswith("Bi") and "meta" not in r["modelo"])
G = R["grid_shallow_queue"]


def mejor_grid(**filtro):
    return max(x["f1_val"] for x in G if all(x.get(k) == v for k, v in filtro.items()))


AT = {a["variante"]: a for a in R["atencion_tabla"]}
AT_DOT = next(a for k, a in AT.items() if k.endswith("+ atención"))
AT_GEN = next(a for k, a in AT.items() if "general" in k)
CL = {c["clase"]: c for c in R["informe_clases"]}
COLAS = [c for c in CL if c not in ("accuracy", "macro avg", "weighted avg")]
BQ = R["busqueda_deep"]

# ----------------------------------------------------------------------------- CSS (formato de la E2)
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Montserrat:wght@700&display=block');
@page { size: A4; margin: 25.4mm; }
@page portada { margin: 0; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: Arial, Helvetica, sans-serif; font-size: 11pt; line-height: 1.32; color: #000; margin: 0; }
.portada { page: portada; width: 210mm; height: 297mm; position: relative; break-after: page; }
.portada .logo { position: absolute; top: 2mm; left: 50%; transform: translateX(-50%); width: 60mm; }
.portada h1.t { position: absolute; top: 127mm; width: 100%; text-align: center; margin: 0;
                font: 700 32pt 'Montserrat', sans-serif; color: #0f172a; }
.portada .pie { position: absolute; left: 25.4mm; top: 206mm; }
.portada .pie h3 { font: 700 14pt 'Montserrat', sans-serif; color: #1e40af; margin: 0 0 3.5mm; }
.portada .pie p { font: 12pt 'Inter', sans-serif; color: #334155; margin: 0 0 6.5mm; }
.portada .pie h3.f { margin-top: 3mm; }
h1 { font: 700 16pt/1.25 'Montserrat', sans-serif; color: #1e40af; margin: 9mm 0 1.5mm; }
h1.res { font-size: 14pt; margin-top: 0; }
.sec { break-inside: avoid; }
p.intro { font: 12pt/1.35 'Inter', sans-serif; color: #334155; margin: 0 0 4mm; }
h2 { font: 700 14pt/1.25 Arial, Helvetica, sans-serif; color: #1e40af; margin: 6mm 0 1.5mm; break-after: avoid; }
p { margin: 0 0 2.4mm; }
ul { list-style: none; margin: 0 0 2.4mm; padding-left: 12.5mm; }
ul li { position: relative; margin-bottom: 0.4mm; }
ul li::before { content: "-"; position: absolute; left: -6.5mm; }
ol { margin: 0 0 2.4mm; padding-left: 9mm; }
ol li { margin-bottom: 0.8mm; }
code { font-family: Consolas, 'Courier New', monospace; font-size: 9.5pt; }
table { border-collapse: collapse; width: 100%; margin: 3mm 0 4mm; font-size: 10pt; line-height: 1.25; break-inside: avoid; }
th, td { border: 0.75pt solid #000; padding: 1.6mm 2mm; vertical-align: top; text-align: left; }
th { background: #cccccc; color: #000; font-weight: 400; vertical-align: middle; }
td.n { text-align: right; white-space: nowrap; }
tr.b td { font-weight: 700; }
.std { color: #555; font-weight: 400; font-size: 8.5pt; }
table.res td:nth-child(2) { width: 36%; }
figure { margin: 3mm 0 4mm; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; }
figcaption { font-size: 9pt; color: #334155; margin-top: 1mm; }
.bloque { break-inside: avoid; }
.refs ul { font-size: 9pt; line-height: 1.22; }
.refs li { margin-bottom: 0; }
"""

S = []

# ----------------------------------------------------------------------------- portada
S.append(f"""<section class="portada">
<img class="logo" src="{(INF / 'deusto.png').as_uri()}">
<h1 class="t">Gestión de Tickets</h1>
<div class="pie"><h3>AUTORES</h3><p>David Trueba</p><p>Gonzalo Iglesias</p><h3 class="f">FECHA</h3><p>{FECHA}</p></div>
</section>""")

# ----------------------------------------------------------------------------- resumen de apartados
S.append(f"""<h1 class="res">Resumen apartados del código</h1>
<p>El código está en <code>src/Entrega_3_NLP.ipynb</code> y se divide en 9 apartados:</p>
{tabla(["Apartado", "Qué hace"], [
    ["0. Configuración", "Semillas, rutas, estilo de las gráficas y modo rápido de prueba (<code>E3_FAST=1</code>)"],
    ["1. Carga y limpieza", "Carga el CSV bruto de E2, aplica sus filtros y elimina los tickets duplicados"],
    ["2. Preprocesado y partición", "Tokenización de E2 y partición train / validación / test (70 / 15 / 15)"],
    ["3. Métricas", "Accuracy, precision, recall y F1 (macro y ponderado); recall de la prioridad <i>high</i>"],
    ["4. Shallow learning", "TF-IDF + Naive Bayes, Regresión logística y SVM lineal con búsqueda en rejilla; meta-features"],
    ["5. Deep learning", "Vocabulario, embeddings, CNN y RNN bidireccional con atención; comparación de embeddings"],
    ["6. Prioridad y tipo", "Las otras dos salidas que planteamos en E1"],
    ["7. Tabla comparativa y errores", "Tabla final, matriz de confusión y confusiones más frecuentes"],
    ["8. Conclusiones", "Resumen de resultados y siguientes pasos"]])}
<p>El notebook se genera con <code>src/build_notebook.py</code> a partir de tres scripts
(<code>src/e3_parte1_datos_shallow.py</code>, <code>e3_parte2_embeddings.py</code> y <code>e3_parte3_deep_evaluacion.py</code>).
La ejecución completa tarda unos 96 minutos con una GPU NVIDIA RTX 3050 y guarda las métricas en
<code>artifacts/resultados.json</code> y las figuras en <code>artifacts/figuras/</code>.</p>""")

# ----------------------------------------------------------------------------- 1. datos
S.append(h1("1. Datos, partición y métricas",
            "Esta sección cubre de dónde salen los datos de entrenamiento: los filtros de E2, un problema de duplicados "
            "que encontramos al entrenar, la nueva partición con validación y las métricas que usamos."))
S.append(f"""<h2>1.1 Datos de partida</h2>
<p>Usamos el mismo dataset de E1 y E2 (<code>Tobi-Bueck/customer-support-tickets</code>) y los mismos filtros: quitamos
los tickets en alemán y los de cuerpo vacío, lo que deja {miles(R['tickets_e2'])} tickets en inglés. El texto de
entrada es el asunto + el cuerpo. Como en E2, la respuesta del agente (<code>answer</code>) no se usa para clasificar,
porque no existe en el momento de clasificar; solo se usa como texto extra para entrenar nuestro Word2Vec.</p>
<p>Queremos predecir las tres salidas de E1: <b>queue</b> (cola o departamento, 10 clases), <b>priority</b> (3 niveles)
y <b>type</b> (4 clases). La tarea principal es la cola.</p>
<h2>1.2 Duplicados entre ficheros</h2>
<p>Al entrenar los primeros modelos vimos que dos de los tres CSV del dataset son versiones distintas del mismo conjunto
y comparten <b>{miles(R['tickets_e2'] - R['tickets_unicos'])} tickets idénticos</b>, con las mismas etiquetas. Si un
ticket está en train y su gemelo en test, el modelo no tiene que generalizar: le basta con haberlo memorizado. Es una
fuga de información que no vimos en E2. Lo medimos con el mismo modelo (TF-IDF + SVM lineal) sobre la partición 80/20
de E2 y sobre los datos sin duplicados:</p>
{tabla(["Partición", "Tickets", "Test con gemelo en train", "Accuracy", "F1 macro"],
       [[e(f["partición"]), miles(f["tickets"]), e(f["test con gemelo en train"]).replace(".", ","), n(f["accuracy"]),
         n(f["f1 macro"])] for f in FUGA], num=(1, 2, 3, 4))}
<p><b>Decisión.</b> Con duplicados, uno de cada cuatro tickets de test ya estaba en train y la accuracy sube
{n(100 * (FUGA[0]['accuracy'] - FUGA[1]['accuracy']), 1)} puntos sin que el modelo sea mejor. Trabajamos con los
<b>{miles(R['tickets_unicos'])} tickets únicos</b> y recalculamos el TF-IDF, el vocabulario y el Word2Vec propio sobre el
nuevo train, en vez de reutilizar las matrices guardadas en E2.</p>
<h2>1.3 Partición train / validación / test</h2>
<p>En E2 solo había train y test. Ahora necesitamos un conjunto de <b>validación</b> para elegir hiperparámetros y decidir
cuándo parar de entrenar las redes; si lo hiciéramos mirando test, test dejaría de ser una estimación honesta. Seguimos
el esquema de la Unidad 3 (70 / 15 / 15), estratificado por cola y con <code>random_state=42</code>:</p>
{tabla(["Conjunto", "Tickets", "Para qué se usa"],
       [["train", miles(P["train"]), "Ajustar los modelos, el vocabulario, el TF-IDF, el escalador y el Word2Vec propio"],
        ["validación", miles(P["val"]), "Elegir hiperparámetros y parar de entrenar (<i>early stopping</i>)"],
        ["test", miles(P["test"]), "Solo para las tablas finales"]], num=(1,))}
<h2>1.4 Métricas</h2>
<p>La cola está muy desbalanceada: <i>Technical Support</i> tiene unas veinte veces más tickets que <i>General
Inquiry</i>. Un modelo que siempre responde la cola más frecuente saca accuracy {n(BASE['accuracy'])} pero F1 macro
{n(BASE['f1'])}. Por eso la métrica que decide qué modelo es mejor es el <b>F1 macro</b>, que da el mismo peso a cada
clase. Para cada modelo damos:</p>
{lista(["<b>Accuracy</b>: porcentaje de aciertos.",
        "<b>Precision, recall y F1</b> calculados por clase y promediados (macro).",
        "<b>F1 ponderado</b> por el tamaño de cada clase, como referencia.",
        "En la prioridad, el <b>recall de la clase <i>high</i></b>, la crítica según E1: un ticket urgente clasificado como no urgente es el error más caro."])}
""")

# ----------------------------------------------------------------------------- 2. shallow
tf = R["tfidf_features"]
svm_uni = mejor_grid(modelo="SVM lineal", tfidf="unigramas")
svm_bi = mejor_grid(modelo="SVM lineal", tfidf="uni+bigramas")
svm_bi_sw = mejor_grid(modelo="SVM lineal", tfidf="uni+bigramas, sin stopwords")
lr_bal = mejor_grid(modelo="Regresión logística", class_weight="balanced")
lr_nobal = max(x["f1_val"] for x in G if x["modelo"] == "Regresión logística" and x["class_weight"] is None)


def params(p):
    return ", ".join(f"{k} = {'ninguno' if v is None else str(v).replace('.', ',')}" for k, v in p.items())


S.append(h1("2. Shallow learning con representación dispersa",
            "Primera familia de modelos: los clasificadores clásicos de la Unidad 3 sobre la representación TF-IDF que "
            "preparamos en E2."))
S.append(f"""<h2>2.1 Representación TF-IDF</h2>
<p>Igual que en E2, cada ticket se representa con TF-IDF ajustado solo con train. Probamos cuatro configuraciones, que
entran en la búsqueda de hiperparámetros:</p>
{tabla(["Configuración", "Features"], [[e(k), miles(v)] for k, v in tf.items()], num=(1,))}
<h2>2.2 Modelos</h2>
{lista(["<b>Naive Bayes</b>: aplica Bayes suponiendo que las palabras son independientes dada la clase. Probamos la "
        "variante <i>Multinomial</i> y la <i>Complement</i>, pensada para clases desbalanceadas.",
        "<b>Regresión logística</b>: aprende un peso por palabra y clase y da probabilidades con softmax.",
        "<b>SVM lineal</b>: busca el hiperplano que separa las clases con el mayor margen.",
        "<b>Clase mayoritaria</b>: siempre responde la cola más frecuente. Es la referencia mínima."])}
<h2>2.3 Optimización de hiperparámetros</h2>
<p>Hacemos una <b>búsqueda en rejilla</b>: se entrena cada combinación con train y se mide su F1 macro en validación; gana
la que más saca y test no interviene. En total son {len(G)} combinaciones para la cola.</p>
{tabla(["Modelo", "Hiperparámetro", "Valores probados"],
       [["todos", "TF-IDF", "unigramas / uni+bigramas, con o sin stopwords"],
        ["Naive Bayes", "variante", "Multinomial, Complement"],
        ["Naive Bayes", "<code>alpha</code> (suavizado)", "0,0003 · 0,001 · 0,003 · 0,01 · 0,03 · 0,1 · 0,3 · 1"],
        ["Regresión logística", "<code>C</code> (inversa de la regularización)", "1 · 10 · 100 · 1000"],
        ["SVM lineal", "<code>C</code>", "0,1 · 0,3 · 1 · 3 · 10 · 30 · 100"],
        ["Regresión logística y SVM", "<code>class_weight</code>", "ninguno, <code>balanced</code>"]])}
<p>Los rangos de <code>C</code> y <code>alpha</code> se ampliaron dos veces, porque en las primeras pruebas el mejor valor
caía en el extremo de la rejilla y no sabíamos si seguiría mejorando. Con los rangos finales, los tres óptimos quedan en
el interior:</p>
{tabla(["Modelo", "Mejor combinación", "F1 macro (validación)"],
       [[e(k), f"{e(v['tfidf'])}; {e(params(v['params']))}", n(v["f1_val"])] for k, v in MS["queue"].items()], num=(2,))}
<p><b>Qué sacamos de la búsqueda.</b> Las tres búsquedas eligen <b>unigramas + bigramas sin stopwords</b>. Los bigramas
son lo que más aporta: en la SVM, el F1 de validación pasa de {n(svm_uni)} con solo unigramas a {n(svm_bi)} con
bigramas. Hay expresiones que señalan el departamento mucho mejor que sus palabras por separado: todos los tickets con
<i>billing discrepancy</i> son de <b>Billing and Payments</b> y el 89 % de los que dicen <i>service outage</i> van a
<b>Service Outages and Maintenance</b>. Quitar stopwords ayuda un poco más ({n(svm_bi_sw)}). La regularización
elegida es débil (C alto, alpha pequeño): conviene dejar que el modelo se ajuste mucho a train. Nuestra interpretación
es que el dataset es sintético y repite plantillas de redacción. Ponderar las clases mejora la regresión logística
({n(lr_nobal)} → {n(lr_bal)}).</p>
<h2>2.4 Features no textuales</h2>
<p>Para combinar distintos tipos de features calculamos {len(R['meta_features'])} <b>meta-features de forma</b>, que
describen cómo está escrito el ticket y no qué palabras usa: longitud en tokens y caracteres, número de oraciones,
longitud media de palabra, si tiene asunto, interrogaciones, exclamaciones, cifras, marcadores anonimizados, proporción
de mayúsculas y saltos de línea. Los conteos se pasan a escala logarítmica y se estandarizan con train. No usamos
<code>type</code>, <code>priority</code> ni los <code>tag_*</code>, porque los pone el agente después de leer el ticket.
Se concatenan al vector TF-IDF multiplicadas por un peso <code>w</code> que también se elige en validación.</p>
<h2>2.5 Resultados en test</h2>
{tabla_res("queue", [r for r in R["resultados"] if r["tarea"] == "queue" and r["familia"] in ("Shallow", "Baseline")])}
<p>La <b>SVM lineal</b> es el mejor modelo shallow (F1 macro {n(SVM['f1'])}), seguida de cerca por la regresión
logística ({n(LR['f1'])}). Naive Bayes tiene la precision más alta ({n(NB['precision'])}) pero el recall más bajo
({n(NB['recall'])}): cuando asigna una cola pequeña suele acertar, pero deja escapar muchos tickets de esas colas.</p>
<p>Las <b>meta-features no aportan</b>: solas apenas superan a la clase mayoritaria (F1 macro {n(META_SOLA['f1'])}) y
añadidas al TF-IDF dejan el F1 igual ({n(META_MAS['f1'])}), con el peso más pequeño probado. La longitud o la
puntuación de un ticket no dicen a qué departamento va.</p>""")

# ----------------------------------------------------------------------------- 3. deep
std_cnn = max(C[x]["f1_test_std"] for x in EMBS)
cnn200 = [x["f1_val"] for x in BQ["CNN"] if x["n_filtros"] == 200]
ep_max = max((x for arq in ("CNN", "RNN") for x in BQ[arq]), key=lambda x: x["mejor_epoca"])
S.append(h1("3. Deep learning: CNN y RNN con atención",
            "Segunda familia: redes que leen el ticket como una secuencia de embeddings. Comparamos cinco formas de "
            "inicializar los embeddings en las dos arquitecturas vistas en la Unidad 3."))
S.append(f"""<h2>3.1 Vocabulario y secuencias</h2>
<p>Usamos la clase <code>Vocabulary</code> de E2 con dos tokens reservados: <code>&lt;PAD&gt;</code> (relleno, índice 0)
y <code>&lt;UNK&gt;</code> (palabras desconocidas). Se construye solo con train y descarta las palabras que aparecen
menos de dos veces, lo que deja {miles(R['vocabulario'])} tokens. Cada ticket pasa a ser una lista de índices recortada o
rellenada hasta 200 tokens (el 99 % tiene menos de 180).</p>
<h2>3.2 Embeddings</h2>
<p>El enunciado pide comparar tipos de embeddings y el efecto del fine-tuning. Probamos cinco variantes de 300 dimensiones:</p>
{tabla(["Variante", "Inicialización", "¿Se entrena?"],
       [["Aleatorio", "Vectores al azar, uniformes en [−0,25; 0,25]", "Sí, desde cero"],
        ["Google News congelado", "Word2Vec preentrenado de E2", "No"],
        ["Google News fine-tuning", "Word2Vec preentrenado de E2", "Sí"],
        ["Word2Vec propio congelado", "Skip-gram con tickets y respuestas de train (como <code>w2v_all</code> de E2)", "No"],
        ["Word2Vec propio fine-tuning", "Skip-gram con tickets y respuestas de train", "Sí"]])}
<p>Para Google News mantenemos los <i>fallbacks</i> de E2 (mayúscula inicial, todo en mayúsculas y sin el posesivo
<code>'s</code>); las palabras sin vector reciben uno aleatorio. Cobertura sobre el nuevo vocabulario:</p>
{tabla(["Embedding", "Palabras del vocabulario con vector", "Tokens de test con vector"],
       [[e(k), pct(v["tipos del vocabulario con vector"]), pct(v["tokens de test con vector"])] for k, v in COB.items()],
       num=(1, 2))}
<p>Google News cubre el {pct(COB['Google News']['tipos del vocabulario con vector'])} del vocabulario, en línea con E2.
Por tokens baja al {pct(COB['Google News']['tokens de test con vector'])} porque aquí contamos también los signos de
puntuación, que Google News no tiene y que las redes sí reciben.</p>
<h2>3.3 Arquitecturas</h2>
<p><b>CNN para texto (Kim, 2014).</b> Filtros de convolución de anchura 3, 4 y 5 recorren el ticket y cada uno se activa
con un patrón de 3, 4 o 5 palabras seguidas. De cada filtro se toma su activación máxima (<i>max-over-time
pooling</i>), las activaciones se concatenan y pasan por una capa densa con dropout y softmax. En el artículo original
nuestras variantes se llaman <i>CNN-rand</i>, <i>CNN-static</i> y <i>CNN-non-static</i>.</p>
<p><b>RNN bidireccional (LSTM o GRU).</b> Lee el ticket palabra a palabra en los dos sentidos. Las puertas de LSTM y GRU
evitan el desvanecimiento del gradiente; el tipo de celda es un hiperparámetro más. El relleno se gestiona con
<i>packed sequences</i>.</p>
<p><b>Atención.</b> Sin atención, todo el ticket se resume en el último estado oculto. Con atención la red vuelve a
mirar todos los estados. Seguimos el cálculo de la Unidad 3; como al clasificar no hay decoder, el «estado actual»
<i>q</i> es el estado final de la RNN:</p>
<ol>
<li>Puntuación de cada estado oculto con el estado actual (producto escalar): s<sub>t</sub> = h<sub>t</sub> · q.</li>
<li>Softmax de las puntuaciones para obtener los pesos de atención α<sub>t</sub>.</li>
<li>Vector de contexto: c = Σ α<sub>t</sub> h<sub>t</sub>.</li>
<li>Se concatena [c ; q] y entra en la capa de salida.</li>
</ol>
<p><b>Modelo no visto en clase: atención «general» (Luong, Pham y Manning, 2015).</b> Es la misma atención con una matriz
aprendida entre los dos vectores, s<sub>t</sub> = h<sub>t</sub> · W q. La incluimos porque el producto escalar solo
compara h<sub>t</sub> y q tal cual, mientras que W deja que la red aprenda qué buscar en cada estado. La comparamos en
el apartado 3.7.</p>
<p>Las dos redes admiten además las meta-features del apartado 2.4, que pasan por una capa densa pequeña y se
concatenan a la representación del texto antes de la salida.</p>
<h2>3.4 Entrenamiento</h2>
{lista(["Entropía cruzada categórica, optimizador Adam (tasa 0,001) y lotes de 64 tickets de longitud parecida.",
        "Dropout antes de la capa de salida.",
        f"<b>Early stopping</b>: tras cada época se mide el F1 macro en validación; si pasan {R['paciencia']} épocas sin "
        f"mejorar se para y se recupera el mejor modelo. Máximo {R['max_epocas']} épocas.",
        f"Cada configuración final se entrena con <b>{len(R['semillas'])} semillas</b> y damos media ± desviación típica. "
        "Una diferencia menor que esa desviación no es significativa."])}
<h2>3.5 Búsqueda de hiperparámetros</h2>
<p>Rejilla evaluada en validación con una sola variante (Google News congelado) y una semilla; los valores elegidos se
mantienen fijos en el resto de experimentos. La RNN tarda bastante más en entrenar, así que su rejilla es más pequeña y
hereda de la CNN el dropout y los pesos por clase. La tasa de aprendizaje, el tamaño de lote y las anchuras de filtro
se dejan en los valores habituales.</p>
<div class="bloque">{tabla(["Filtros", "Dropout", "Pesos por clase", "F1 validación", "Mejor época"],
       [[b["n_filtros"], n(b["dropout"], 1), "sí" if b["pesos_clase"] else "no", n(b["f1_val"]), f"{b['mejor_epoca']:.0f}"]
        for b in BQ["CNN"]], num=(0, 1, 3, 4), negrita=(0,))}</div>
<div class="bloque">{tabla(["Celda", "Estado oculto", "F1 validación", "Mejor época", "Tiempo (s)"],
       [[b["celda"], b["oculto"], n(b["f1_val"]), f"{b['mejor_epoca']:.0f}", f"{b['segundos']:.0f}"] for b in BQ["RNN"]],
       num=(1, 2, 3, 4), negrita=(0,))}</div>
<p><b>Decisión.</b> En la CNN pesa sobre todo el número de filtros: las cuatro configuraciones con 200 filtros superan a
las cuatro con 100, y entre ellas el F1 varía menos de dos puntos ({n(min(cnn200))} a {n(max(cnn200))}). Elegimos 200
filtros, dropout {n(HP['CNN']['dropout'], 1)} y sin pesos por clase. En la RNN gana una <b>{RNN}</b> de
{HP['RNN']['oculto']} unidades, muy por debajo de la CNN. En las dos redes la mejor capacidad es la mayor que probamos,
así que podrían mejorar con más; no lo ampliamos por tiempo de cómputo. Además, una configuración llegó a su mejor
resultado en la época {ep_max['mejor_epoca']:.0f}, por eso el máximo de épocas es {R['max_epocas']} y no 40.</p>
<h2>3.6 Comparación de embeddings</h2>
<p>Con los hiperparámetros fijados, cada arquitectura se entrena con las cinco variantes y tres semillas:</p>
{tabla(["Embedding", "CNN: F1 val.", "CNN: F1 test", f"{RNN}: F1 val.", f"{RNN}: F1 test"],
       [[e(emb), n(C[emb]["f1_val"]), f'{n(C[emb]["f1_test"])} <span class="std">± {n(C[emb]["f1_test_std"])}</span>',
         n(Rn[emb]["f1_val"]), f'{n(Rn[emb]["f1_test"])} <span class="std">± {n(Rn[emb]["f1_test_std"])}</span>']
        for emb in EMBS], num=(1, 2, 3, 4))}
{figura("03_comparacion_embeddings", "Figura 1. F1 macro en test de cada arquitectura y embedding (media de 3 semillas). La línea vertical es la mejor SVM.", "82%")}
<p><b>Embeddings preentrenados.</b> En la CNN, partir de vectores preentrenados mejora unos dos puntos al aleatorio (de
{n(C['Aleatorio']['f1_test'])} a {n(C['Google News congelado']['f1_test'])}–{n(C['Google News fine-tuning']['f1_test'])}),
más que la variación entre semillas (como mucho ± {n(std_cnn)}). Con {miles(P['train'])} tickets de entrenamiento la red no
tiene datos suficientes para aprender desde cero lo que Word2Vec ya sabe.</p>
<p><b>Congelado o fine-tuning.</b> Con Google News da igual ({n(C['Google News congelado']['f1_test'])} y
{n(C['Google News fine-tuning']['f1_test'])}): sus vectores ya son buenos. Con nuestro Word2Vec, entrenado con solo
2,2 millones de tokens, el fine-tuning sí ayuda ({n(C['Word2Vec propio congelado']['f1_test'])} →
{n(C['Word2Vec propio fine-tuning']['f1_test'])}). El fine-tuning además converge antes (mejor época
{C['Google News fine-tuning']['mejor_epoca']:.0f} frente a {C['Google News congelado']['mejor_epoca']:.0f}).</p>
<p><b>La RNN.</b> La {RNN} queda entre doce y catorce puntos por debajo de la CNN con cada embedding. Los embeddings
congelados no le aportan nada frente a los aleatorios y lo que más ayuda es el fine-tuning del Word2Vec propio
({n(Rn['Word2Vec propio fine-tuning']['f1_test'])}). Nuestra interpretación es que la RNN tiene que resumir tickets de
unos 70 tokens en un único vector, mientras que a la CNN le basta con detectar unos pocos n-gramas característicos.</p>
<h2>3.7 Atención</h2>
<p>Entrenamos la misma {RNN}, con su mejor embedding ({e(ME['RNN'])}), sin atención, con la atención de clase y con la
variante general. Además del F1 medimos dónde cae la atención:</p>
{tabla(["Variante", "Accuracy", "F1 macro", "Peso en el primer y último token", "Peso del token más atendido"],
       [[e(a["variante"]), n(a["accuracy"]), f'{n(a["f1"])} <span class="std">± {n(a["f1_std"])}</span>',
         pct(a.get("atención en primer y último token")), pct(a.get("peso del token más atendido"))] for a in R["atencion_tabla"]],
       num=(1, 2, 3, 4))}
<p><b>En el F1 no hay diferencias</b>: las tres variantes quedan dentro de la variación entre semillas. Nuestros tickets
son cortos y lo que limita a la RNN no parece ser el cuello de botella del estado final.</p>
<p><b>Donde sí se nota es en qué mira la red.</b> La atención es muy selectiva: el token más atendido se lleva de media
el {pct(AT_DOT['peso del token más atendido'], 0)} del peso con producto escalar y el
{pct(AT_GEN['peso del token más atendido'], 0)} con la variante general, es decir, la red decide casi siempre por una
palabra (en la figura 2, <i>payment</i>). Además, la atención por producto escalar dedica el
{pct(AT_DOT['atención en primer y último token'])} del peso al primer y al último token, frente al
{pct(AT_GEN['atención en primer y último token'])} de la general. Es un efecto de usar el estado final como consulta: el
estado del primer token contiene el final de la lectura hacia atrás y el del último, el de la lectura hacia delante, así
que su producto escalar con la consulta sale alto aunque esas palabras no digan nada. La matriz W lo evita.</p>
{figura("05_atencion_ejemplo", "Figura 2. Los diez tokens con más peso de atención en un ticket de test, con cada variante.", "92%")}
<p>Agregando sobre todo test, en <i>Billing and Payments</i> la atención va a <i>billing</i>, <i>invoice</i> y
<i>payment</i>, y en <i>Service Outages and Maintenance</i> a <i>outage</i>, <i>maintenance</i> y <i>disruption</i>. En
las dos colas más pequeñas, <i>General Inquiry</i> y <i>Human Resources</i>, atiende sobre todo a palabras vacías: no ha
encontrado palabras que las identifiquen.</p>""")

cm_, cs_ = deep(f"{NA['CNN']} + meta-features · {ME['CNN']}"), deep(f"{NA['CNN']} · {ME['CNN']}")
rm_, rs_ = deep(f"{NA['RNN']} + meta-features · {ME['RNN']}"), deep(f"{NA['RNN']} · {ME['RNN']}")
S.append(f"""<h2>3.8 Red híbrida: texto + meta-features</h2>
<p>La misma idea del apartado 2.4, dentro de la red y con la mejor configuración de cada arquitectura:</p>
{tabla(["Modelo", "F1 validación", "F1 test"],
       [[e(x["modelo"]), n(x["f1_val"]), f'{n(x["f1_test"])} <span class="std">± {n(x["f1_test_std"])}</span>'] for x in (cs_, cm_, rs_, rm_)],
       num=(1, 2))}
<p>En la CNN, las meta-features suben el F1 de test ({n(cs_['f1_test'])} → {n(cm_['f1_test'])}) pero lo bajan en
validación ({n(cs_['f1_val'])} → {n(cm_['f1_val'])}). Como elegimos los modelos por validación, no lo damos por una
mejora real. En la {RNN} no cambian nada. Es coherente con el apartado 2.5.</p>""")


# ----------------------------------------------------------------------------- 4. prioridad y tipo
def top2(t):
    filas = sorted((r for r in R["resultados"] if r["tarea"] == t and r["familia"] != "Baseline"), key=lambda r: -r["f1"])
    return filas[0], next(r for r in filas if r["familia"] != filas[0]["familia"])


pr1, pr2 = top2("priority")
ty1, ty2 = top2("type")
pr_cnn, ty_cnn = deep(f"{NA['CNN']} · {ME['CNN']}", "priority"), deep(f"{NA['CNN']} · {ME['CNN']}", "type")
S.append(h1("4. Prioridad y tipo",
            "Repetimos el proceso para las otras dos etiquetas de E1. Los modelos clásicos repiten su búsqueda en rejilla; "
            "las redes reutilizan los hiperparámetros y el mejor embedding de la cola."))
S.append(f"""<h2>4.1 Prioridad (3 clases)</h2>
{tabla_res("priority", high=True)}
<p>La prioridad es bastante más difícil que el tipo. La CNN ({n(pr1['f1'])}) y la SVM ({n(pr2['f1'])}) quedan
prácticamente empatadas (la diferencia es del orden de la desviación entre semillas, ± {n(pr_cnn['f1_test_std'])}) y las
dos recuperan en torno al {pct(pr1['recall_high'], 0)} de los tickets urgentes. Es decir, más de uno de cada cuatro tickets
de prioridad alta se clasifica como media o baja. La urgencia depende de cosas como cuánta gente está afectada, más que de
palabras concretas.</p>
<h2>4.2 Tipo (4 clases)</h2>
{tabla_res("type")}
<p>El tipo es la tarea más fácil. Aquí la CNN ({n(ty1['f1'])}) sí supera a la SVM ({n(ty2['f1'])}) por un punto, unas tres
veces su desviación entre semillas (± {n(ty_cnn['f1_test_std'])}). Nuestra interpretación es que el tipo (incidencia,
petición, problema o cambio) se expresa con fórmulas muy repetidas que la CNN detecta bien como n-gramas.</p>""")

# ----------------------------------------------------------------------------- 5. comparativa
S.append(h1("5. Tabla comparativa final",
            f"Todos los modelos de la tarea principal (cola), ordenados por F1 macro en test. En los modelos deep, cada "
            f"cifra es la media de {len(R['semillas'])} semillas y en gris va la desviación típica."))
S.append(f"""{tabla_res("queue")}
<p>En la cola, los dos mejores modelos son lineales sobre TF-IDF: la SVM ({n(SVM['f1'])}) y la regresión logística
({n(LR['f1'])}). La mejor CNN queda a unos cuatro puntos ({n(CNN_Q['f1'])}) y la mejor {RNN}, a más de quince
({n(RNN_Q['f1'])}). Todos los modelos que usan el texto tienen más precision que recall: son conservadores con las colas
pequeñas. Que las redes no ganen en la cola tiene, a nuestro juicio, tres causas:</p>
{lista([f"{miles(P['train'])} tickets de entrenamiento son pocos para que una red aprenda una representación mejor que los bigramas que TF-IDF ya cuenta.",
        "La cola depende sobre todo de palabras y expresiones clave (la atención se concentra en una sola palabra), y eso lo captura directamente un modelo lineal sobre bigramas.",
        "Las redes están menos ajustadas: su rejilla es más pequeña y su mejor capacidad cae en el extremo de lo probado."])}
<p>En prioridad y tipo, en cambio, la CNN iguala o supera a la SVM, así que no hay una familia que gane en todas las tareas.</p>""")

# ----------------------------------------------------------------------------- 6. errores
orden = sorted(COLAS, key=lambda c: -CL[c]["f1-score"])
gi = CL["General Inquiry"]
soporte = {"Technical Support", "Product Support", "IT Support", "Customer Service"}
n_sop = sum(1 for c in R["confusiones"] if c["real"] in soporte and c["predicha"] in soporte)
S.append(h1("6. Análisis de errores",
            f"Analizamos el mejor modelo de la cola según validación, {e(R['mejor_modelo_cola'])}. En la práctica es la SVM "
            f"sola: las meta-features entran con el peso mínimo y sus métricas de test coinciden."))
S.append(f"""{figura("07_matriz_confusion", "Figura 3. Matriz de confusión normalizada por filas: a qué colas van los tickets de cada cola real.", "68%")}
{tabla(["Cola", "Precision", "Recall", "F1", "Tickets en test"],
       [[e(c), n(CL[c]["precision"]), n(CL[c]["recall"]), n(CL[c]["f1-score"]), miles(CL[c]["support"])] for c in orden],
       num=(1, 2, 3, 4))}
<div class="bloque"><p>Las confusiones más frecuentes:</p>
{tabla(["Cola real", "Cola predicha", "Tickets"], [[e(c["real"]), e(c["predicha"]), miles(c["tickets"])] for c in R["confusiones"]], num=(2,))}</div>
<p><b>Qué vemos.</b> Las colas mejor resueltas son las que tienen un vocabulario propio: <b>{orden[0]}</b> (F1
{n(CL[orden[0]]['f1-score'])}) y <b>{orden[1]}</b> ({n(CL[orden[1]]['f1-score'])}). Los errores se concentran en cuatro colas
que se solapan, <b>Technical Support, Product Support, IT Support y Customer Service</b>: {n_sop} de las
{len(R['confusiones'])} confusiones más frecuentes son entre ellas. Su contenido se parece mucho; por ejemplo, de los
tickets que mencionan <i>data breach</i>, el 38 % va a Technical Support, el 20 % a Product Support y el 12 % a
Customer Service. Si el mismo tipo de problema se reparte entre colas, ningún modelo basado solo en el texto puede
acertarlas todas.</p>
<p><b>General Inquiry</b>, la cola más pequeña ({miles(gi['support'])} tickets en test), tiene precision alta
({n(gi['precision'])}) y recall bajo ({n(gi['recall'])}): el modelo casi nunca la predice y, cuando lo hace, acierta.</p>""")

# ----------------------------------------------------------------------------- 7. conclusiones
S.append(h1("7. Conclusiones y siguientes pasos",
            "Resumen de lo que hemos aprendido en esta primera iteración y de lo que haremos en la siguiente."))
S.append(f"""<ol>
<li><b>Datos.</b> Quitar los duplicados era imprescindible: con la partición de E2, uno de cada cuatro tickets de test
estaba también en train. Todos los resultados están medidos sin esa fuga y con validación separada de test.</li>
<li><b>Shallow learning.</b> La SVM lineal sobre TF-IDF con unigramas y bigramas sin stopwords es el mejor modelo para la
cola (F1 macro {n(SVM['f1'])}) y un baseline muy difícil de batir. Las meta-features de forma no aportan.</li>
<li><b>Deep learning.</b> La CNN funciona claramente mejor que la RNN ({n(CNN_Q['f1'], 2)} frente a {n(RNN_Q['f1'], 2)} en la
cola). Los embeddings preentrenados aportan unos dos puntos y el fine-tuning solo ayuda cuando los vectores de partida
son pobres.</li>
<li><b>Atención.</b> No mejora el F1, pero muestra que la red decide casi siempre por una sola palabra, y nos ha permitido
detectar el sesgo de la atención por producto escalar hacia los extremos del ticket.</li>
<li><b>Prioridad y tipo.</b> El tipo se resuelve bien (F1 {n(ty1['f1'])}); la prioridad no tanto (F1 {n(pr1['f1'])}, con un
{pct(pr1['recall_high'], 0)} de recall en la clase urgente).</li>
<li><b>Límite del problema.</b> Gran parte del error de la cola está en cuatro departamentos cuyos tickets se parecen
mucho; probablemente una parte no se pueda resolver solo con el texto.</li>
</ol>
<h2>Siguientes pasos</h2>
{lista(["Más capacidad en las redes y ajustar sus hiperparámetros para prioridad y tipo.",
        "Fine-tuning de un Transformer tipo BERT o DistilBERT (Unidad 4).",
        "Pre-entrenar con datasets parecidos, por ejemplo las conversaciones de soporte de Twitter de TweetSumm.",
        "Mejorar el recall de la prioridad alta ajustando el umbral de decisión de esa clase.",
        "Clasificar las cuatro colas de soporte de forma jerárquica, como Zangari et al. (2023).",
        "Segunda tarea del proyecto: resumen de conversaciones de TweetSumm con un seq2seq con atención, evaluado con ROUGE."])}
<h2>Referencias</h2>
<div class="refs">{lista(["Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. EMNLP.",
        "Luong, M.-T., Pham, H. y Manning, C. D. (2015). Effective Approaches to Attention-based Neural Machine Translation. EMNLP.",
        "Bahdanau, D., Cho, K. y Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. ICLR.",
        "Mikolov, T. et al. (2013). Distributed Representations of Words and Phrases and their Compositionality. NeurIPS.",
        "Rennie, J. et al. (2003). Tackling the Poor Assumptions of Naive Bayes Text Classifiers. ICML.",
        "Zangari, A. et al. (2023). Ticket automation: An insight into current research with applications to multi-level classification scenarios. Expert Systems with Applications.",
        "Dataset: Tobi-Bueck/customer-support-tickets (Hugging Face). Librerías: scikit-learn, PyTorch y gensim."])}</div>""")

documento = (f'<!doctype html><html lang="es"><head><meta charset="utf-8"><title>Gestión de Tickets · E3</title>'
             f'<style>{CSS}</style></head><body>{"".join(S)}</body></html>')
SALIDA_HTML.write_text(documento, encoding="utf-8")
print("HTML:", SALIDA_HTML)
with tempfile.TemporaryDirectory() as perfil:
    subprocess.run([str(EDGE), "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={perfil}",
                    "--virtual-time-budget=20000", f"--print-to-pdf={SALIDA_PDF}", SALIDA_HTML.as_uri()],
                   check=True, timeout=180)
print("PDF:", SALIDA_PDF)
