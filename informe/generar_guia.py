"""Guía de estudio de la E3: qué se ha hecho, por qué y cómo funciona cada pieza.

No es el informe de la entrega (ese es generar_informe.py). Es un documento para entender el trabajo
y preparar su defensa. Las cifras salen de artifacts/resultados.json.

Uso (desde la raíz del proyecto):
    python informe/generar_guia.py
"""
import html
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INF = ROOT / "informe"
FIG = ROOT / "artifacts" / "figuras"
R = json.loads((ROOT / "artifacts" / "resultados.json").read_text(encoding="utf-8"))
SALIDA_HTML = INF / "Guia_E3.html"
SALIDA_PDF = INF / "Guia_E3_para_entenderla.pdf"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")


def n(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def pct(x, d=1):
    return f"{100 * x:.{d}f}".replace(".", ",") + " %"


def miles(x):
    return f"{int(x):,}".replace(",", ".")


def e(s):
    return html.escape(str(s))


def res(tarea, modelo):
    return next(r for r in R["resultados"] if r["tarea"] == tarea and r["modelo"] == modelo)


def mejor(tarea, cond):
    return max((r for r in R["resultados"] if r["tarea"] == tarea and cond(r)), key=lambda r: r["f1"])


def figura(nombre, pie, ancho="100%"):
    return (f'<figure><img src="{(FIG / f"{nombre}.png").as_uri()}" style="width:{ancho}">'
            f'<figcaption>{pie}</figcaption></figure>')


def tabla(cab, filas, num=(), clase=""):
    th = "".join(f'<th class="{"n" if i in num else ""}">{c}</th>' for i, c in enumerate(cab))
    tb = "".join("<tr>" + "".join(f'<td class="{"n" if i in num else ""}">{c}</td>' for i, c in enumerate(f)) + "</tr>"
                 for f in filas)
    return f'<table class="{clase}"><thead><tr>{th}</tr></thead><tbody>{tb}</tbody></table>'


def codigo(texto):
    """Recuadro «En el código»: dónde mirar en el notebook."""
    return f'<div class="cod"><div class="et">En el código</div>{texto}</div>'


def idea(texto):
    return f'<div class="idea"><div class="et">La idea</div>{texto}</div>'


# ----------------------------------------------------------------------------- cifras
P, F = R["particiones"], R["fuga"]
SVM = res("queue", "TF-IDF + SVM lineal")
LR = res("queue", "TF-IDF + Regresión logística")
NB = res("queue", "TF-IDF + Naive Bayes")
BASE = mejor("queue", lambda r: r["familia"] == "Baseline")
NA, ME, HP = R["nombre_arq"], R["mejor_embedding"], R["hp_deep"]
CNN_Q = mejor("queue", lambda r: r["modelo"].startswith("CNN") and "meta" not in r["modelo"])
RNN_Q = mejor("queue", lambda r: r["modelo"].startswith("Bi") and "meta" not in r["modelo"])
CNN_AL = res("queue", "CNN · Aleatorio")
COB = {c["embedding"]: c for c in R["cobertura"]}
CL = {c["clase"]: c for c in R["informe_clases"]}
AT = {a["variante"]: a for a in R["atencion_tabla"]}
AT_DOT = next(a for k, a in AT.items() if k.endswith("+ atención"))
AT_GEN = next(a for k, a in AT.items() if "general" in k)
AT_NO = next(a for k, a in AT.items() if "sin atención" in k)
TAREAS = {"queue": "Cola (10 clases)", "priority": "Prioridad (3 clases)", "type": "Tipo (4 clases)"}


def fila_tarea(t):
    sh = mejor(t, lambda r: r["familia"] == "Shallow" and "meta" not in r["modelo"])
    cn = mejor(t, lambda r: r["modelo"].startswith("CNN") and "meta" not in r["modelo"])
    rn = mejor(t, lambda r: r["modelo"].startswith("Bi") and "meta" not in r["modelo"])
    ba = mejor(t, lambda r: r["familia"] == "Baseline")
    top = max((sh, cn, rn), key=lambda r: r["f1"])
    fmt = lambda r: f"<b>{n(r['f1'])}</b>" if r is top else n(r["f1"])  # noqa: E731
    return [TAREAS[t], fmt(sh), fmt(cn), fmt(rn), n(ba["f1"])]


# ----------------------------------------------------------------------------- diagramas (SVG)
AZ, AZC, GR, GRC, TX = "#2347b5", "#eef2fb", "#8a8a8f", "#f3f3f5", "#1d1d1f"


def caja(x, y, w, h, l1, l2="", fondo=AZC, borde=AZ, negrita=True):
    t2 = f'<text x="{x + w / 2}" y="{y + h / 2 + 13}" class="s2">{e(l2)}</text>' if l2 else ""
    dy = -4 if l2 else 5
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{fondo}" stroke="{borde}" stroke-width="1.3"/>'
            f'<text x="{x + w / 2}" y="{y + h / 2 + dy}" class="{"s1" if negrita else "s3"}">{e(l1)}</text>{t2}')


def flecha(x1, y1, x2, y2):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{GR}" stroke-width="1.4" marker-end="url(#pt)"/>'


DEFS = (f'<defs><marker id="pt" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{GR}"/></marker></defs>'
        '<style>.s1{font:600 12.5px Inter,"Segoe UI",sans-serif;fill:#1d1d1f;text-anchor:middle}'
        '.s2{font:11px Inter,"Segoe UI",sans-serif;fill:#4a4a4f;text-anchor:middle}'
        '.s3{font:12px Inter,"Segoe UI",sans-serif;fill:#1d1d1f;text-anchor:middle}'
        '.tl{font:600 11px Inter,"Segoe UI",sans-serif;fill:#2347b5;text-anchor:middle;letter-spacing:.3px}'
        '.tk{font:12px Inter,"Segoe UI",sans-serif;fill:#1d1d1f;text-anchor:end}'
        '.nm{font:11px Inter,"Segoe UI",sans-serif;fill:#1d1d1f;text-anchor:middle}</style>')


def svg_pipeline():
    w, g, x0, y0 = 150, 20, 10, 18
    xs = [x0 + i * (w + g) for i in range(4)]
    top = [("Dataset bruto", f"{miles(61765)} tickets"), ("Filtros de E2", f"{miles(R['tickets_e2'])} en inglés"),
           ("Sin duplicados", f"{miles(R['tickets_unicos'])} tickets únicos"), ("Partición", "70 / 15 / 15 estratificada")]
    s = [caja(x, y0, w, 58, a, b) for x, (a, b) in zip(xs, top)]
    s += [flecha(xs[i] + w, y0 + 29, xs[i + 1] - 2, y0 + 29) for i in range(3)]
    cx = xs[3] + w / 2
    ax, bx, bw = 40, 370, 270
    s.append(f'<path d="M{cx},{y0 + 58} L{cx},104 L{ax + bw / 2},104 L{ax + bw / 2},128" fill="none" stroke="{GR}" stroke-width="1.4" marker-end="url(#pt)"/>')
    s.append(f'<path d="M{cx},104 L{bx + bw / 2},104 L{bx + bw / 2},128" fill="none" stroke="{GR}" stroke-width="1.4" marker-end="url(#pt)"/>')
    s.append(f'<text x="{ax + bw / 2}" y="124" class="tl">SHALLOW · REPRESENTACIÓN DISPERSA</text>'.replace('y="124"', 'y="121"'))
    s.append(f'<text x="{bx + bw / 2}" y="121" class="tl">DEEP · REPRESENTACIÓN DENSA</text>')
    s += [caja(ax, 130, bw, 50, "TF-IDF", "unigramas + bigramas, sin stopwords"),
          caja(bx, 130, bw, 50, "Secuencias + embeddings", "200 tokens × 300 dimensiones")]
    s += [flecha(ax + bw / 2, 180, ax + bw / 2, 204), flecha(bx + bw / 2, 180, bx + bw / 2, 204)]
    s += [caja(ax, 206, bw, 50, "Naive Bayes · Reg. logística · SVM", "búsqueda en rejilla sobre validación"),
          caja(bx, 206, bw, 50, "CNN · BiGRU con atención", "5 variantes de embedding × 3 semillas")]
    s.append(f'<path d="M{ax + bw / 2},256 L{ax + bw / 2},282 L{bx + bw / 2},282 L{bx + bw / 2},256" fill="none" stroke="{GR}" stroke-width="1.4"/>')
    s.append(flecha(340, 282, 340, 300))
    s.append(caja(190, 302, 300, 48, "Evaluación en test", "accuracy, precision, recall y F1 macro", fondo=GRC, borde=GR))
    return f'<svg viewBox="0 0 680 360" class="dg">{DEFS}{"".join(s)}</svg>'


def svg_cnn():
    toks = ["i", "was", "charged", "twice", "for", "my", "invoice"]
    s = ['<text x="105" y="14" class="tl">1 · EMBEDDINGS</text>']
    for i, t in enumerate(toks):
        y = 26 + i * 25
        s.append(f'<text x="62" y="{y + 15}" class="tk">{t}</text>')
        s.append(f'<rect x="70" y="{y}" width="120" height="19" rx="3" fill="{GRC}" stroke="#c8c8cc"/>')
        s += [f'<rect x="{74 + k * 12}" y="{y + 5}" width="8" height="9" rx="1.5" fill="#c9d4f2"/>' for k in range(9)]
    s.append(f'<rect x="66" y="{26 + 25 - 4}" width="128" height="{3 * 25 + 2}" rx="5" fill="none" stroke="{AZ}" stroke-width="2.2"/>')
    s.append(f'<text x="105" y="214" class="nm">ventana de 3 palabras</text>')
    s.append('<text x="300" y="14" class="tl">2 · UN FILTRO</text>')
    vals = [0.1, 0.9, 0.3, 0.1, 0.0]
    for i, v in enumerate(vals):
        y = 38 + i * 30
        hl = i == 1
        s.append(f'<rect x="270" y="{y}" width="60" height="22" rx="4" fill="{AZ if hl else AZC}" stroke="{AZ}"/>')
        s.append(f'<text x="300" y="{y + 15}" class="nm" style="fill:{"#fff" if hl else TX}">{n(v, 1)}</text>')
    s.append(flecha(196, 85, 266, 60))
    s.append('<text x="300" y="214" class="nm">5 posiciones posibles</text>')
    s.append('<text x="430" y="14" class="tl">3 · MAX-POOLING</text>')
    s.append(caja(370, 92, 120, 44, "máx = 0,9", "¿aparece el patrón?", fondo=AZC))
    s.append(flecha(334, 114, 367, 114))
    s.append('<text x="596" y="14" class="tl">4 · CLASIFICACIÓN</text>')
    s.append(caja(518, 60, 156, 52, "600 valores", "200 filtros × 3 anchuras", fondo=GRC, borde=GR))
    s.append(flecha(492, 114, 515, 92))
    s.append(caja(518, 140, 156, 52, "Capa densa + softmax", "probabilidad de cada cola", fondo=GRC, borde=GR))
    s.append(flecha(596, 112, 596, 137))
    return f'<svg viewBox="0 0 680 222" class="dg">{DEFS}{"".join(s)}</svg>'


def svg_atencion():
    toks = ["my", "invoice", "payment", "failed", "again", "."]
    pesos = [0.02, 0.06, 0.78, 0.08, 0.03, 0.03]
    s = []
    x0, dx = 40, 82
    s.append(f'<text x="{x0 + 2.5 * dx + 30}" y="14" class="tl">PESOS DE ATENCIÓN α (SUMAN 1)</text>')
    for i, (t, a) in enumerate(zip(toks, pesos)):
        x = x0 + i * dx
        h = 6 + a * 90
        s.append(f'<rect x="{x + 15}" y="{118 - h}" width="30" height="{h}" rx="3" fill="{AZ if a > 0.5 else "#9fb3e6"}"/>')
        s.append(f'<text x="{x + 30}" y="{112 - h}" class="nm">{n(a, 2)}</text>')
        s.append(f'<rect x="{x + 5}" y="128" width="50" height="34" rx="6" fill="{AZC}" stroke="{AZ}"/>')
        s.append(f'<text x="{x + 30}" y="150" class="nm">h<tspan dy="3" font-size="9">{i + 1}</tspan></text>')
        s.append(f'<text x="{x + 30}" y="186" class="s3">{t}</text>')
    s.append(f'<text x="18" y="150" class="tk" style="text-anchor:middle;font-size:10px;fill:#4a4a4f">RNN</text>')
    s.append(caja(545, 30, 125, 46, "q = estado final", "la «pregunta»", fondo=GRC, borde=GR))
    s.append(caja(545, 96, 125, 46, "c = Σ α·h", "resumen ponderado", fondo=AZC))
    s.append(caja(545, 162, 125, 46, "[c ; q] → softmax", "cola predicha", fondo=GRC, borde=GR))
    s.append(flecha(607, 76, 607, 93))
    s.append(flecha(607, 142, 607, 159))
    s.append(f'<path d="M545,53 C500,53 470,60 455,70" fill="none" stroke="{GR}" stroke-width="1.2" stroke-dasharray="3 3"/>')
    s.append(f'<text x="470" y="46" class="nm" style="font-size:10px;fill:#4a4a4f">compara q con cada h</text>')
    return f'<svg viewBox="0 0 680 212" class="dg">{DEFS}{"".join(s)}</svg>'


def svg_flujo():
    s = []
    cajas = [("Scripts .py", "e3_parte1, 2 y 3"), ("Unirlos", "build_notebook.py"), ("Notebook", "Entrega_3_NLP.ipynb"),
             ("Resultados", "artifacts/"), ("Informe PDF", "generar_informe.py")]
    w, g = 126, 12
    for i, (a, b) in enumerate(cajas):
        x = 1 + i * (w + g)
        s.append(caja(x, 10, w, 52, a, b, fondo=AZC if i in (2, 4) else GRC, borde=AZ if i in (2, 4) else GR))
        if i:
            s.append(flecha(x - g, 36, x - 3, 36))
    return f'<svg viewBox="0 0 680 72" class="dg">{DEFS}{"".join(s)}</svg>'


# ----------------------------------------------------------------------------- CSS
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Montserrat:wght@600;700&display=block');
:root { --azul: #2347b5; --tinta: #1d1d1f; --tinta2: #4a4a4f; --linea: #c8c8cc; --cab: #d9d9d9; }
@page { size: A4; margin: 19mm 18mm 19mm 18mm;
        @bottom-right { content: counter(page); font: 9pt 'Inter', 'Segoe UI', sans-serif; color: #77777c; } }
@page portada { margin: 0; @bottom-right { content: none; } }
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: 'Inter', 'Segoe UI', sans-serif; font-size: 10pt; line-height: 1.5; color: var(--tinta); margin: 0; }
.portada { page: portada; height: 297mm; width: 210mm; position: relative; break-after: page; }
.portada .bloque { position: absolute; top: 92mm; left: 22mm; right: 22mm; }
.portada .kicker { font-family: 'Montserrat', 'Segoe UI', sans-serif; color: var(--azul); font-size: 11pt; letter-spacing: .5pt; }
.portada h1 { font-family: 'Montserrat', 'Segoe UI', sans-serif; font-weight: 700; font-size: 34pt; line-height: 1.1; margin: 4mm 0 6mm; color: #13182a; }
.portada .sub { font-size: 12.5pt; color: var(--tinta2); max-width: 140mm; }
.portada .nota { position: absolute; left: 22mm; right: 22mm; bottom: 24mm; font-size: 9.5pt; color: #77777c;
                 border-top: 0.6pt solid var(--linea); padding-top: 4mm; }
h2 { font-family: 'Montserrat', 'Segoe UI', sans-serif; color: var(--azul); font-size: 17pt; margin: 0 0 3mm; line-height: 1.25; break-after: avoid; }
h2.nueva { break-before: page; }
h3 { font-weight: 600; color: var(--azul); font-size: 12pt; margin: 5.5mm 0 2mm; break-after: avoid; }
p { margin: 0 0 2.5mm; }
ul, ol { margin: 0 0 2.5mm; padding-left: 6mm; }
li { margin-bottom: 1mm; }
code { font-family: Consolas, monospace; font-size: 8.8pt; background: #f1f1f3; padding: 0 1mm; border-radius: 1mm; }
table { border-collapse: collapse; width: 100%; margin: 2mm 0 4mm; font-size: 8.8pt; break-inside: avoid; }
th, td { border: 0.6pt solid #8a8a8f; padding: 1.3mm 2mm; vertical-align: top; text-align: left; }
th { background: var(--cab); font-weight: 600; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
figure { margin: 3mm 0 4mm; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; }
figcaption { font-size: 8.6pt; color: var(--tinta2); margin-top: 1.5mm; }
svg.dg { width: 100%; height: auto; margin: 2mm 0 1mm; break-inside: avoid; }
.pie { font-size: 8.6pt; color: var(--tinta2); text-align: center; margin: 0 0 4mm; }
.cod, .idea { border-radius: 2mm; padding: 2.6mm 3.6mm; margin: 3mm 0 4mm; break-inside: avoid; font-size: 9.4pt; }
.cod { background: #f5f5f7; border: 0.6pt solid var(--linea); }
.idea { background: #f3f5fc; border-left: 1.2mm solid var(--azul); }
.et { font-family: 'Montserrat', 'Segoe UI', sans-serif; font-size: 8pt; font-weight: 700; letter-spacing: .6pt;
      text-transform: uppercase; color: var(--azul); margin-bottom: 1mm; }
.cod p:last-child, .idea p:last-child, .cod ul:last-child { margin-bottom: 0; }
.ticket { border: 0.6pt solid var(--linea); border-radius: 2mm; padding: 3mm 4mm; margin: 2mm 0 4mm; font-size: 9.2pt; break-inside: avoid; }
.ticket .meta { margin-top: 2mm; display: flex; gap: 3mm; flex-wrap: wrap; }
.chip { border-radius: 10mm; padding: 0.6mm 3mm; font-size: 8.4pt; background: #eef2fb; color: var(--azul); font-weight: 600; }
.qa { break-inside: avoid; margin-bottom: 3.2mm; }
.qa .q { font-weight: 600; margin-bottom: 0.8mm; }
.qa .a { color: #2c2c30; padding-left: 4mm; border-left: 0.8mm solid #d5dcf2; }
.dos { display: grid; grid-template-columns: 1fr 1fr; gap: 5mm; }
.indice li { margin-bottom: 1.6mm; }
"""

# ----------------------------------------------------------------------------- contenido
S = []

S.append("""
<section class="portada"><div class="bloque">
  <div class="kicker">GESTIÓN DE TICKETS · ENTREGA 3</div>
  <h1>Cómo funciona<br>nuestra E3</h1>
  <div class="sub">Guía para entender el trabajo: qué hemos hecho, por qué lo hemos hecho así, cómo funciona cada
  modelo y cómo leer los resultados. Incluye las preguntas que pueden hacernos en la defensa.</div>
</div>
<div class="nota">Documento de estudio. No es el informe de la entrega (ese es <i>E3_Gestion_de_Tickets.pdf</i>).
Todas las cifras salen de la ejecución completa del notebook guardada en <i>artifacts/resultados.json</i>.</div>
</section>""")

S.append(f"""
<h2>Índice</h2>
<ol class="indice">
<li><b>La idea en una página</b>: qué problema resolvemos y el recorrido completo, de los datos a los resultados.</li>
<li><b>Los datos</b>: qué entra en los modelos, la fuga por duplicados y por qué hay tres particiones.</li>
<li><b>Cómo medimos</b>: accuracy, precision, recall y F1, y por qué usamos el F1 macro.</li>
<li><b>Shallow learning</b>: TF-IDF y los tres clasificadores clásicos.</li>
<li><b>Deep learning</b>: embeddings, la CNN, la RNN, la atención y cómo se entrenan las redes.</li>
<li><b>Qué ha salido y por qué</b>: los resultados y su interpretación.</li>
<li><b>El código</b>: cómo está organizado, cómo se ejecuta y dónde está cada requisito del enunciado.</li>
<li><b>Preguntas para la defensa</b>, con respuesta corta.</li>
<li><b>Lo que conviene reconocer</b>: las limitaciones del trabajo.</li>
</ol>

<h2 class="nueva">1. La idea en una página</h2>
<p>Un equipo de soporte recibe muchos tickets y tiene que leerlos, decidir a qué departamento van, cómo de urgentes son
y de qué tipo son. Queremos que un modelo lo haga a partir del texto del ticket. Un ejemplo real del dataset:</p>
<div class="ticket"><b>Asunto:</b> Account Disruption<br>
<b>Cuerpo:</b> <i>Dear Customer Support Team, I am writing to report a significant problem with the centralized account
management portal, which currently appears to be offline. This outage is blocking access to account settings…</i>
<div class="meta"><span class="chip">cola: Technical Support</span><span class="chip">prioridad: high</span>
<span class="chip">tipo: Incident</span></div></div>
<p>Son tres problemas de <b>clasificación multiclase</b>: la cola (10 departamentos), la prioridad (3 niveles) y el
tipo (4 clases). Este es el recorrido completo del trabajo:</p>
<div style="break-inside:avoid">{svg_pipeline()}
<div class="pie">El camino de los datos: se limpian, se quitan duplicados, se reparten en tres conjuntos y se
representan de dos formas, una para los modelos clásicos y otra para las redes.</div></div>
{idea(f"<p>Probamos dos familias de modelos sobre exactamente los mismos tickets y las comparamos con las mismas "
      f"métricas. Resultado principal: en la cola gana un modelo clásico, la <b>SVM lineal sobre TF-IDF</b> (F1 macro "
      f"{n(SVM['f1'])}); la mejor red, una <b>CNN</b>, saca {n(CNN_Q['f1'])} en la cola pero gana en prioridad y tipo.</p>")}
""")

S.append(f"""
<h2 class="nueva">2. Los datos</h2>
<h3>Qué entra en los modelos</h3>
<p>Usamos el dataset de Hugging Face <code>Tobi-Bueck/customer-support-tickets</code>, el mismo de E1 y E2. Viene en
tres CSV con {miles(61765)} tickets. Aplicamos los filtros de E2: quitamos los tickets en alemán (nuestros embeddings
y stopwords son de inglés) y los de cuerpo vacío. Quedan {miles(R['tickets_e2'])}.</p>
<p>La entrada de los modelos es el <b>asunto + el cuerpo</b>. El dataset trae también la respuesta del agente
(<code>answer</code>), pero <b>no la usamos para clasificar</b>: cuando llega un ticket nuevo todavía no hay respuesta,
así que usarla sería hacer trampa. Solo la aprovechamos como texto extra para entrenar nuestro Word2Vec (que no ve
las etiquetas).</p>
<h3>La trampa de los duplicados</h3>
<p>Dos de los tres CSV son versiones distintas del mismo dataset y comparten {miles(R['tickets_e2'] - R['tickets_unicos'])}
tickets idénticos, con las mismas etiquetas. En E2 no lo vimos. ¿Por qué importa? Si un ticket está en train y su
gemelo en test, el modelo acierta el de test porque ya lo ha visto, no porque haya aprendido. Es como si en el examen
cayeran preguntas idénticas a las de los apuntes: la nota sube, pero no mide lo que sabes. A esto se le llama
<b>fuga de información</b> (<i>data leakage</i>).</p>
{tabla(["Partición", "Test con gemelo en train", "Accuracy", "F1 macro"],
       [[e(f['partición']), e(f['test con gemelo en train']).replace('.', ','), n(f['accuracy']), n(f['f1 macro'])] for f in F],
       num=(1, 2, 3))}
<p>Con el mismo modelo, los duplicados inflaban la accuracy unos {n(100 * (F[0]['accuracy'] - F[1]['accuracy']), 0)}
puntos. Por eso trabajamos con los {miles(R['tickets_unicos'])} tickets únicos y recalculamos todo (TF-IDF, vocabulario,
Word2Vec) desde cero.</p>
<h3>Tres conjuntos: train, validación y test</h3>
<div class="dos">
<div><p><b>Train ({miles(P['train'])}, 70 %)</b>: con lo que aprende el modelo. Son los apuntes.</p>
<p><b>Validación ({miles(P['val'])}, 15 %)</b>: con lo que decidimos cosas (qué hiperparámetros, cuándo parar de
entrenar). Son los simulacros de examen.</p>
<p><b>Test ({miles(P['test'])}, 15 %)</b>: se mira una sola vez, al final. Es el examen de verdad.</p></div>
<div>{idea("<p>Si eligiéramos los hiperparámetros mirando test, acabaríamos ajustando el modelo a esos tickets "
          "concretos y la nota de test dejaría de ser honesta. Por eso hace falta un tercer conjunto.</p>"
          "<p>La partición es <b>estratificada</b>: cada conjunto mantiene la proporción de cada cola, que es muy "
          "desigual.</p>")}</div>
</div>
{codigo("<p>Apartados 1 y 2 del notebook. <code>clave_texto()</code> normaliza el texto para detectar gemelos, "
        "<code>medir_fuga()</code> calcula la tabla de arriba, <code>procesar()</code> parte en oraciones y tokens, y la "
        "partición usa <code>train_test_split(..., stratify=df['queue'], random_state=42)</code> dos veces (70/30 y luego "
        "el 30 a la mitad).</p>")}
""")

gi = CL["General Inquiry"]
S.append(f"""
<h2>3. Cómo medimos</h2>
<p>Para cada cola nos preguntamos dos cosas:</p>
<ul>
<li><b>Precision</b>: de los tickets que el modelo manda a esa cola, ¿cuántos eran de verdad de esa cola?
(«cuando digo X, ¿acierto?»).</li>
<li><b>Recall</b>: de los tickets que eran de esa cola, ¿cuántos encuentra el modelo? («de todos los X, ¿cuántos
pillo?»).</li>
<li><b>F1</b>: la media armónica de las dos. Solo es alta si las dos lo son.</li>
<li><b>Accuracy</b>: el porcentaje total de aciertos, sin mirar clases.</li>
</ul>
{idea(f"<p>Ejemplo real con el mejor modelo: <i>General Inquiry</i> tiene precision {n(gi['precision'])} y recall "
      f"{n(gi['recall'])}. Cuando el modelo dice «General Inquiry» casi siempre acierta, pero de los "
      f"{miles(gi['support'])} tickets de esa cola en test solo encuentra la mitad: el resto los manda a otras colas.</p>")}
<h3>Por qué el F1 macro</h3>
<p>Precision, recall y F1 se calculan por clase y luego hay que promediarlas. El promedio <b>macro</b> da el mismo
peso a cada cola; el <b>ponderado</b> pesa más las colas grandes. Como <i>Technical Support</i> tiene unas veinte veces
más tickets que <i>General Inquiry</i>, la accuracy y el F1 ponderado se pueden subir acertando solo las colas grandes.
Lo vemos con un modelo que responde siempre la cola más frecuente: tiene accuracy {n(BASE['accuracy'])} pero F1 macro
{n(BASE['f1'])}, porque falla todas las demás colas. El F1 macro castiga eso, y por eso es la métrica que usamos para
decidir qué modelo es mejor.</p>
<p>En la prioridad miramos además el <b>recall de la clase <i>high</i></b>: un ticket urgente que se queda en la cola
de los no urgentes es el error más caro (lo dijimos en E1).</p>
{codigo("<p>Apartado 3. <code>metricas()</code> calcula todo con scikit-learn; <code>registrar()</code> guarda una fila "
        "por modelo (con media y desviación si hay varias semillas) y <code>tabla()</code> la muestra ordenada por F1.</p>")}
""")

tf = R["tfidf_features"]
ms = R["mejor_shallow"]["queue"]
S.append(f"""
<h2 class="nueva">4. Shallow learning: TF-IDF y modelos clásicos</h2>
<h3>TF-IDF: el ticket como una bolsa de palabras con pesos</h3>
<p>Cada ticket se convierte en un vector con una posición por palabra (o par de palabras) del vocabulario. El valor de
cada posición es el peso <b>TF-IDF</b>:</p>
<ul>
<li><b>TF</b> (frecuencia del término): cuántas veces sale la palabra en el ticket. Más veces, más peso.</li>
<li><b>IDF</b> (frecuencia inversa en documentos): baja el peso de las palabras que salen en casi todos los tickets.
<i>«support»</i> sale en casi todos y no ayuda a distinguir; <i>«invoice»</i> sale en pocos y sí.</li>
</ul>
<p>Con unigramas y bigramas el vocabulario tiene {miles(tf['uni+bigramas, sin stopwords'])} entradas, pero cada ticket
usa unas pocas decenas: casi todo el vector son ceros. Por eso se llama representación <b>dispersa</b>. No guarda el
orden de las palabras, salvo el que capturan los <b>bigramas</b> (pares de palabras seguidas), y eso resulta clave:
todos los tickets con <i>«billing discrepancy»</i> son de <i>Billing and Payments</i>.</p>
<h3>Los tres clasificadores</h3>
<ul>
<li><b>Naive Bayes</b>: calcula, para cada cola, la probabilidad de ver esas palabras, suponiendo que son
independientes entre sí (por eso «naive», ingenuo). Muy rápido. Su hiperparámetro <code>alpha</code> es el suavizado:
evita que una palabra que nunca salió en una cola anule su probabilidad.</li>
<li><b>Regresión logística</b>: aprende un peso por palabra y cola, suma los pesos de las palabras del ticket y lo
convierte en probabilidades con softmax.</li>
<li><b>SVM lineal</b>: busca la frontera entre colas que deja el mayor margen posible. En espacios de muchas
dimensiones y dispersos suele ser el mejor modelo lineal, y aquí lo es.</li>
</ul>
<h3>Hiperparámetros y búsqueda en rejilla</h3>
<p>Un <b>hiperparámetro</b> es una decisión que no aprende el modelo sino que fijamos nosotros, como <code>C</code>,
que controla la regularización: con C pequeño el modelo es prudente y simple; con C grande se ajusta mucho a
train. <code>class_weight="balanced"</code> hace que los errores en colas pequeñas cuenten más.</p>
<p>La <b>búsqueda en rejilla</b> prueba todas las combinaciones de una lista de valores, entrena cada una con train,
mide su F1 macro en validación y se queda con la mejor. Si el mejor valor queda en el extremo de la lista, no sabemos
si seguiría mejorando más allá, así que ampliamos los rangos dos veces hasta que los óptimos quedaron dentro.</p>
{tabla(["Modelo", "Mejor combinación encontrada", "F1 validación"],
       [[e(k), e(v['tfidf']) + "; " + ", ".join(f"{a} = {str(b).replace('.', ',')}" for a, b in v['params'].items()), n(v['f1_val'])]
        for k, v in ms.items()], num=(2,))}
<p>Las tres búsquedas eligen bigramas sin stopwords y poca regularización. Nuestra interpretación: el dataset es
sintético y repite formas de redactar, así que «memorizar» expresiones concretas también funciona en tickets nuevos.</p>
<h3>Meta-features: probar algo que no sea el texto</h3>
<p>El enunciado valora combinar features distintas. Calculamos 12 sobre la forma del ticket (longitud, número de
oraciones, interrogaciones, cifras, mayúsculas…) y las añadimos al TF-IDF. No ayudan: solas casi no superan a la clase
mayoritaria y junto al TF-IDF no cambian nada. Tiene lógica: lo largo que es un ticket no dice a qué departamento va.</p>
{codigo("<p>Apartado 4. <code>texto_tfidf()</code> prepara el texto, <code>VEC_CONFIGS</code> define las 4 formas de "
        "TF-IDF, <code>GRIDS</code> los valores a probar, <code>busqueda_shallow()</code> entrena todas las combinaciones en "
        "paralelo (8 procesos con joblib, unos 8 minutos) y <code>evaluar_shallow()</code> reentrena la mejor de cada modelo "
        "y la mide en test. <code>meta_features()</code> y <code>con_meta()</code> son el experimento de 4.2.</p>")}
""")

S.append(f"""
<h2 class="nueva">5. Deep learning</h2>
<h3>De palabras a números: vocabulario y secuencias</h3>
<p>Las redes no leen texto. Primero se crea un <b>vocabulario</b> con las palabras de train que salen al menos dos
veces ({miles(R['vocabulario'])}), cada una con un número. Cada ticket pasa a ser una lista de números, recortada o
rellenada hasta 200. El relleno es <code>&lt;PAD&gt;</code> (0) y las palabras desconocidas son <code>&lt;UNK&gt;</code>
(1). A diferencia de TF-IDF, aquí <b>el orden se conserva</b>.</p>
<h3>Embeddings: cada palabra, un vector de 300 números</h3>
<p>Cada número del ticket se sustituye por un vector denso (<b>embedding</b>). La idea de Word2Vec, de la Unidad 2:
palabras que aparecen en contextos parecidos acaban con vectores parecidos (<i>invoice</i> cerca de <i>billing</i>).
Probamos tres orígenes y dos formas de usarlos:</p>
{tabla(["Variante", "De dónde salen los vectores", "¿Cambian al entrenar?"],
       [["Aleatorio", "Al azar; la red los aprende desde cero", "Sí"],
        ["Google News congelado", "Word2Vec preentrenado con ~100.000 millones de palabras de noticias", "No"],
        ["Google News fine-tuning", "El mismo", "Sí"],
        ["Word2Vec propio congelado", f"Entrenado por nosotros con tickets y respuestas de train ({n(2.217107, 1)} M de tokens)", "No"],
        ["Word2Vec propio fine-tuning", "El mismo", "Sí"]])}
{idea("<p><b>Congelado</b> es como usar un diccionario impreso: los significados no cambian y la red solo aprende a "
      "usarlos. <b>Fine-tuning</b> es un diccionario en el que puedes escribir notas: los vectores se ajustan a nuestra "
      "tarea, pero hay más cosas que aprender y más riesgo de memorizar.</p>")}
<p>Google News tiene vector para el {pct(COB['Google News']['tipos del vocabulario con vector'])} de nuestro vocabulario
(le faltan sobre todo nombres de productos, como en E2).</p>
<h3>La CNN (Kim, 2014)</h3>
<div style="break-inside:avoid">{svg_cnn()}
<div class="pie">Un filtro de anchura 3 se desliza por el ticket. Cada posición da un número: alto si esas tres palabras
se parecen al patrón que el filtro ha aprendido. Del recorrido solo se guarda el máximo.</div></div>
<p>Cada filtro es un detector de un patrón de 3, 4 o 5 palabras seguidas: un «n-grama aprendido». Por ejemplo, un
filtro puede aprender a activarse con <i>«charged twice for»</i>. Con el <b>max-pooling</b> solo importa si el patrón
aparece en algún sitio, no dónde. Con 200 filtros de cada anchura salen 600 números por ticket, que una capa densa
convierte en probabilidades de cada cola. Es rápida y encaja muy bien con este problema, que depende de expresiones
clave.</p>
<h3>La RNN bidireccional (LSTM o GRU)</h3>
<p>Una <b>RNN</b> lee el ticket palabra a palabra y mantiene un estado oculto, una especie de memoria de lo leído.
La RNN básica olvida pronto (desvanecimiento del gradiente); <b>LSTM</b> y <b>GRU</b> lo arreglan con
<b>puertas</b> que deciden qué guardar y qué olvidar. La GRU tiene menos puertas y menos pesos; en la búsqueda ganó a la
LSTM por poco, así que usamos una <b>{NA['RNN'].split()[0]}</b> de {HP['RNN']['oculto']} unidades. Es
<b>bidireccional</b>: una lectura va de izquierda a derecha y otra al revés, y cada palabra queda representada con
lo que hay antes y después. Con <i>packed sequences</i> la red no pierde tiempo leyendo el relleno.</p>
<h3>La atención</h3>
<div style="break-inside:avoid">{svg_atencion()}
<div class="pie">La red compara su estado final (q) con el estado de cada palabra, convierte las puntuaciones en
pesos que suman 1 y hace un resumen ponderado. Los pesos son de ejemplo, pero el patrón (casi todo en una palabra) es
el que vimos en los datos.</div></div>
<p>Sin atención, todo el ticket tiene que caber en el estado final de la RNN: es un <b>cuello de botella</b>. Con
atención la red vuelve a mirar todas las palabras y decide cuánto peso darle a cada una. Seguimos los pasos de la
Unidad 3 (diapositiva 71):</p>
<ol>
<li>Puntuación de cada estado h<sub>t</sub> con el estado actual q: producto escalar h<sub>t</sub>·q.</li>
<li>Softmax de las puntuaciones: pesos α<sub>t</sub> que suman 1.</li>
<li>Vector de contexto c = Σ α<sub>t</sub> h<sub>t</sub>.</li>
<li>Se concatena [c ; q] y se clasifica.</li>
</ol>
<p>Lo comparamos con la variante <b>general</b> de Luong (2015), que mete una matriz aprendida W entre los dos
vectores (h<sub>t</sub>·W·q) para que la red decida qué buscar.</p>
<h3>Cómo se entrena una red</h3>
<ul>
<li><b>Lote</b> (batch): se le pasan 64 tickets, calcula su error y ajusta los pesos. Una <b>época</b> es una pasada
por todo train.</li>
<li><b>Función de pérdida</b>: entropía cruzada, que mide lo lejos que están las probabilidades predichas de la cola
real. <b>Adam</b> es el optimizador que decide cuánto mover cada peso.</li>
<li><b>Dropout</b>: durante el entrenamiento apaga al azar parte de las neuronas para que la red no dependa de unas
pocas y memorice menos.</li>
<li><b><i>Early stopping</i></b>: tras cada época se mide el F1 en validación. Si en 5 épocas no mejora, se para y se
recupera la mejor versión. Así se evita el sobreajuste: en las curvas se ve que la pérdida de train llega casi a 0
mientras validación se estanca.</li>
<li><b>Semillas</b>: una red da resultados algo distintos cada vez que se entrena (empieza con pesos al azar). Por eso
cada configuración final se entrena 3 veces y damos media ± desviación típica. Una diferencia menor que esa desviación
no es una diferencia de verdad.</li>
</ul>
{codigo("<p>Apartado 5. <code>Vocabulary</code> y <code>a_matriz()</code> (5.1); <code>matriz_desde()</code> monta la matriz "
        "de embeddings y <code>EMBEDDINGS</code> define las 5 variantes (5.2); <code>TextCNN</code> y <code>BiRNN</code> son "
        "las redes, con la atención dentro de <code>BiRNN.forward()</code> (5.3); <code>entrenar()</code> es el bucle con "
        "<i>early stopping</i> y <code>experimento()</code> lo repite con 3 semillas y registra el resultado (5.4); "
        "<code>buscar()</code> es la rejilla de las redes (5.5).</p>")}
""")

filas_t = [fila_tarea(t) for t in TAREAS]
conf = R["confusiones"]
PS_REC = CL["Product Support"]["recall"]
PS_TS = next(c["tickets"] for c in conf if c["real"] == "Product Support" and c["predicha"] == "Technical Support") \
    / CL["Product Support"]["support"]
S.append(f"""
<h2 class="nueva">6. Qué ha salido y por qué</h2>
<p>F1 macro en test (en las redes, media de 3 semillas). En negrita, el mejor de cada tarea.</p>
{tabla(["Tarea", "Mejor clásico (SVM)", "Mejor CNN", "Mejor BiGRU", "Clase mayoritaria"], filas_t, num=(1, 2, 3, 4))}
{figura('03_comparacion_embeddings', 'F1 macro de cada red con cada embedding. La línea vertical es la SVM.', '88%')}
<h3>Las seis ideas que hay que saber explicar</h3>
<ol>
<li><b>En la cola gana la SVM ({n(SVM['f1'])}).</b> La cola depende de palabras y expresiones clave, y eso lo captura
directamente un modelo lineal sobre bigramas. Además, 16.620 tickets son pocos para que una red aprenda algo mejor.</li>
<li><b>La CNN va mucho mejor que la RNN</b> ({n(CNN_Q['f1'])} frente a {n(RNN_Q['f1'])}). A la CNN le basta con
detectar unos pocos n-gramas; la RNN tiene que resumir todo el ticket.</li>
<li><b>Los embeddings preentrenados ayudan</b>: unos dos puntos en la CNN ({n(CNN_AL['f1'])} con aleatorios). Con
pocos datos, partir de lo que Word2Vec ya sabe es mejor que aprender desde cero. El fine-tuning solo ayuda cuando los
vectores de partida son pobres (nuestro Word2Vec).</li>
<li><b>La atención no mejora el F1</b> (sin atención {n(AT_NO['f1'])}, con atención {n(AT_DOT['f1'])}, general
{n(AT_GEN['f1'])}: todo dentro del ruido entre semillas), pero nos deja ver qué hace la red: el token más atendido se
lleva de media el {pct(AT_DOT['peso del token más atendido'], 0)} del peso. Decide casi siempre por una palabra.</li>
<li><b>La atención por producto escalar mira demasiado los extremos</b> ({pct(AT_DOT['atención en primer y último token'])}
del peso en el primer y último token, frente al {pct(AT_GEN['atención en primer y último token'])} de la general). La
razón: el estado del primer token contiene el final de la lectura hacia atrás y el del último, el de la lectura hacia
delante. Cada uno contiene media «pregunta», así que su producto escalar con ella sale alto sin que la palabra
importe. La matriz W de la variante general lo evita.</li>
<li><b>Los errores se concentran en cuatro colas de soporte</b> (<i>Technical, Product, IT, Customer Service</i>): las
{len(conf)} confusiones más frecuentes son entre ellas. Sus tickets se parecen mucho; por ejemplo, los que hablan de
<i>«data breach»</i> se reparten entre Technical, Product y Customer Service. Parte del error es ambigüedad de las
propias etiquetas.</li>
</ol>
{figura('07_matriz_confusion', 'Matriz de confusión del mejor modelo: cada fila es una cola real y dice a qué colas van sus tickets. La diagonal son los aciertos.', '68%')}
<p><b>Cómo leer la matriz:</b> la fila <i>Product Support</i> tiene {n(PS_REC, 2)} en la diagonal (acierta el
{pct(PS_REC, 0)} de sus tickets) y {n(PS_TS, 2)} en la columna <i>Technical Support</i>: casi uno de cada cinco
tickets de Product Support acaba en Technical Support.</p>
""")

S.append(f"""
<h2 class="nueva">7. El código</h2>
<h3>Cómo se construye todo</h3>
<div style="break-inside:avoid">{svg_flujo()}
<div class="pie">El notebook no se edita a mano: se escribe en tres scripts y se genera. Al ejecutarlo deja los
resultados en <i>artifacts/</i>, de donde sale el informe.</div></div>
{tabla(["Fichero", "Qué contiene"],
       [["<code>src/e3_parte1_datos_shallow.py</code>", "Apartados 0-4: configuración, datos, duplicados, partición, métricas, TF-IDF, modelos clásicos y meta-features"],
        ["<code>src/e3_parte2_embeddings.py</code>", "Apartados 5.1-5.2: vocabulario, secuencias y las matrices de embeddings"],
        ["<code>src/e3_parte3_deep_evaluacion.py</code>", "Apartados 5.3-8: redes, entrenamiento, búsqueda, comparación, atención, prioridad y tipo, tabla final y conclusiones"],
        ["<code>src/build_notebook.py</code>", "Junta los tres scripts en <code>src/Entrega_3_NLP.ipynb</code>"],
        ["<code>artifacts/</code>", "<code>resultados.json</code> y <code>.csv</code> con todas las métricas, las 7 figuras y la caché de Google News"],
        ["<code>informe/generar_informe.py</code>", "Construye el informe PDF a partir de <code>artifacts/</code>; la redacción está en <code>textos_informe.py</code>"]])}
<h3>Cómo se ejecuta</h3>
<ul>
<li>Hace falta PyTorch con GPU. En este PC está en el entorno <code>C:\\Users\\Usuario\\.venvs\\nlp-e3</code>
(el <code>env/</code> del repo, el de E2, no tiene PyTorch).</li>
<li>Ejecución completa: abrir el notebook con ese entorno y ejecutar todo. Tarda unos <b>96 minutos</b> en la RTX 3050.
La primera vez descarga Google News (1,6 GB) si no está en la caché de gensim.</li>
<li>Prueba rápida: con la variable de entorno <code>E3_FAST=1</code> usa el 20 % de los datos, 1 semilla y 2 épocas
(unos 2 minutos). Ojo: sobrescribe <code>artifacts/</code>.</li>
<li>Si se cambia algo de los <code>.py</code>: <code>python src/build_notebook.py</code> y volver a ejecutar.</li>
<li>El informe: <code>python informe/generar_informe.py</code> (y esta guía, <code>informe/generar_guia.py</code>).</li>
</ul>
<h3>Dónde está cada requisito del enunciado</h3>
{tabla(["El enunciado pide", "Dónde lo cumplimos"],
       [["Mínimo: 2 técnicas de shallow learning con representación dispersa", "Naive Bayes, Regresión logística y SVM lineal sobre TF-IDF (apartado 4). Son tres."],
        ["Mínimo: 2 técnicas de deep learning comparando embeddings y fine-tuning", "CNN y BiGRU con atención, cada una con 5 variantes: aleatorio, Google News y Word2Vec propio, congelados y con fine-tuning (5.6)"],
        ["Tabla comparativa con accuracy, recall, precision y F1", "Apartado 7 del notebook y apartado 5 del informe, para las tres tareas"],
        ["Justificar la optimización de hiperparámetros", "Búsqueda en rejilla sobre validación (4.1 y 5.5), con los rangos ampliados hasta que el óptimo quedó dentro"],
        ["Justificar modelos no vistos en clase", "La atención general de Luong (2015), explicada y referenciada; el resto es de la Unidad 3"],
        ["Se valora: atención", "Atención en la RNN y análisis de en qué palabras se fija (5.7)"],
        ["Se valora: combinar features no textuales", "12 meta-features de forma, en los modelos lineales (4.2) y en las redes (5.8)"],
        ["Se valora: pre-entrenar con datasets parecidos", "Solo en parte: Google News es preentrenado, pero no con datos de soporte. Queda como siguiente paso"]])}
""")

S.append(f"""
<h2 class="nueva">8. Preguntas para la defensa</h2>
<div class="qa"><div class="q">¿Por qué usáis el F1 macro y no la accuracy?</div><div class="a">Porque las colas están
muy desbalanceadas. Un modelo que siempre dice «Technical Support» saca accuracy {n(BASE['accuracy'])} y F1 macro
{n(BASE['f1'])}. El F1 macro da el mismo peso a cada cola y no se deja engañar por las grandes.</div></div>
<div class="qa"><div class="q">¿Por qué quitasteis los duplicados si en E2 no lo hicisteis?</div><div class="a">Porque
al entrenar modelos supervisados vimos que el 26 % de test tenía un gemelo exacto en train. Era una fuga que inflaba la
accuracy unos 9 puntos. En E2 solo preparábamos el texto y no se notaba.</div></div>
<div class="qa"><div class="q">¿Por qué no usáis la respuesta del agente?</div><div class="a">Porque no existe cuando
llega un ticket nuevo. Solo la usamos para entrenar el Word2Vec, que no ve las etiquetas.</div></div>
<div class="qa"><div class="q">¿Para qué sirve el conjunto de validación?</div><div class="a">Para elegir
hiperparámetros y decidir cuándo parar de entrenar sin mirar test. Si usáramos test para eso, su nota dejaría de ser
una estimación honesta.</div></div>
<div class="qa"><div class="q">¿Cómo elegisteis los hiperparámetros?</div><div class="a">Con búsqueda en rejilla: todas
las combinaciones, cada una medida con F1 macro en validación. En los clásicos ampliamos los rangos hasta que el mejor
valor quedó dentro de la rejilla. En las redes la rejilla es más pequeña porque cada entrenamiento tarda minutos.</div></div>
<div class="qa"><div class="q">¿Qué es C y por qué os sale tan alto?</div><div class="a">Es la inversa de la
regularización: con C alto el modelo se ajusta más a train. Sale alto porque el dataset es sintético y repite formas de
redactar, así que memorizar expresiones también sirve en tickets nuevos.</div></div>
<div class="qa"><div class="q">¿Por qué una SVM gana a las redes neuronales?</div><div class="a">Porque la cola se decide
por palabras y expresiones clave (los bigramas lo capturan directamente), hay pocos datos para una red (16.620 tickets) y
las redes están menos ajustadas. En prioridad y tipo la CNN sí iguala o gana a la SVM.</div></div>
<div class="qa"><div class="q">¿Por qué la RNN va tan mal comparada con la CNN?</div><div class="a">La RNN tiene que
condensar todo el ticket en un vector, mientras que a la CNN le basta con detectar unos pocos patrones locales. Para
clasificar por temas, lo segundo es justo lo necesario.</div></div>
<div class="qa"><div class="q">¿Qué diferencia hay entre congelar los embeddings y hacer fine-tuning?</div><div class="a">
Congelados, los vectores no cambian y la red solo aprende a usarlos. Con fine-tuning también se ajustan a la tarea. En
nuestro caso ayuda con nuestro Word2Vec (vectores menos pulidos) y da igual con Google News.</div></div>
<div class="qa"><div class="q">Si la atención no mejora el F1, ¿para qué sirve?</div><div class="a">Para entender qué
mira el modelo: vemos que decide casi siempre por una sola palabra (como <i>«payment»</i>) y descubrimos un sesgo de la
atención por producto escalar hacia los extremos del ticket. Además, su ventaja crece con textos largos (traducción), y
nuestros tickets son cortos.</div></div>
<div class="qa"><div class="q">¿Por qué entrenáis cada red tres veces?</div><div class="a">Porque cada entrenamiento
empieza con pesos al azar y el resultado varía. Con la media y la desviación sabemos si una diferencia es real o ruido.</div></div>
<div class="qa"><div class="q">¿Por qué no está TweetSumm?</div><div class="a">Porque su fichero solo trae los IDs de
los tweets, no el texto, y E3 sigue con el dataset y las features de E2. Es la segunda tarea del proyecto (resumen) y la
dejamos para la siguiente entrega.</div></div>
<div class="qa"><div class="q">¿Qué haríais después?</div><div class="a">Dar más capacidad a las redes, probar un
Transformer tipo BERT (Unidad 4), pre-entrenar con conversaciones de soporte, mejorar el recall de la prioridad alta y
abordar el resumen con TweetSumm.</div></div>
""")

S.append(f"""
<h2>9. Lo que conviene reconocer</h2>
<p>Ningún trabajo es perfecto, y es mejor decirlo antes de que lo pregunten:</p>
<ul>
<li><b>Capacidad de las redes en el borde.</b> La mejor CNN usa 200 filtros y la mejor RNN 128 unidades, que son los
máximos probados. Podrían mejorar con más; no se amplió por tiempo de cómputo.</li>
<li><b>Prioridad y tipo con hiperparámetros heredados.</b> Las redes de esas dos tareas usan la configuración elegida
para la cola, sin optimizarla para ellas.</li>
<li><b>El «mejor modelo» de la cola lleva meta-features.</b> Por validación gana <i>TF-IDF + meta-features + SVM</i>
por una milésima, pero el peso de las meta-features es mínimo y en test da exactamente lo mismo que la SVM sola.</li>
<li><b>La mejora de la CNN con meta-features no es fiable.</b> Sube en test pero baja en validación; puede ser azar.</li>
<li><b>No hemos pre-entrenado con datasets parecidos</b>, que el enunciado valora; Google News es preentrenado, pero
con noticias.</li>
<li><b>Recall de la prioridad alta del {pct(mejor('priority', lambda r: r['familia'] != 'Baseline')['recall_high'], 0)}</b>: más de
uno de cada cuatro tickets urgentes se clasifica como no urgente.</li>
<li><b>El dataset es sintético</b>: los resultados pueden no trasladarse tal cual a tickets reales.</li>
</ul>
""")

documento = (f'<!doctype html><html lang="es"><head><meta charset="utf-8"><title>Cómo funciona nuestra E3</title>'
             f'<style>{CSS}</style></head><body>{"".join(S)}</body></html>')
SALIDA_HTML.write_text(documento, encoding="utf-8")
with tempfile.TemporaryDirectory() as perfil:
    subprocess.run([str(EDGE), "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={perfil}",
                    "--virtual-time-budget=20000", f"--print-to-pdf={SALIDA_PDF}", SALIDA_HTML.as_uri()],
                   check=True, timeout=180)
print("PDF:", SALIDA_PDF)
