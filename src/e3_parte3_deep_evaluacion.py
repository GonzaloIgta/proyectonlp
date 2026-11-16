# %% [markdown]
# ### 5.3 Las arquitecturas (las vistas en clase, Unidad 3)
#
# **CNN para texto (Kim, 2014).** Filtros de convolución de anchura 3, 4 y 5 recorren el ticket y
# cada uno se activa cuando ve un patrón concreto de 3, 4 o 5 palabras seguidas (un n-grama
# aprendido). De cada filtro nos quedamos solo con su activación máxima (*max-over-time pooling*):
# importa si el patrón aparece, no dónde. Las activaciones se concatenan y pasan por una capa
# densa con dropout y softmax. En el artículo original, las variantes que comparamos aquí se
# llaman *CNN-rand* (embeddings aleatorios), *CNN-static* (congelados) y *CNN-non-static*
# (fine-tuning).
#
# **RNN bidireccional (LSTM o GRU).** La red lee el ticket palabra a palabra manteniendo una
# memoria; al ser bidireccional, otra lo lee de derecha a izquierda, de modo que cada palabra
# queda representada con su contexto por ambos lados. LSTM y GRU usan *puertas* para decidir qué
# recordar y qué olvidar, y así evitan el desvanecimiento del gradiente de la RNN básica. La GRU
# tiene menos puertas y menos pesos; según la regla vista en clase suele ir mejor con pocos datos,
# así que el tipo de celda se trata como un hiperparámetro más (5.5). El relleno se gestiona con
# *packed sequences*, para que la red no procese los `<PAD>`.
#
# **Atención.** Sin atención, el ticket entero se resume en el último estado oculto, que hace de
# cuello de botella. Con atención la red vuelve a mirar todos los estados. Se sigue el cálculo de
# clase, con una adaptación: al clasificar no hay decoder, así que el «estado actual» $q$ es el
# estado final de la RNN.
#
# 1. Puntuación de similitud de cada estado oculto $h_t$ con el estado actual: $s_t = h_t^\top q$ (producto escalar).
# 2. Softmax sobre las puntuaciones para obtener los pesos de atención: $\alpha_t = e^{s_t} / \sum_k e^{s_k}$.
# 3. Suma de los estados ponderada por su peso, el vector de contexto: $c = \sum_t \alpha_t h_t$.
# 4. Se concatena el contexto con el estado actual, $[c\,;q]$, y eso entra en la capa de salida.
#
# En 5.7 se compara además con la variante *general* de Luong et al. (2015), $s_t = h_t^\top W q$,
# que aprende una matriz $W$ para decidir qué buscar.
#
# Las dos redes admiten una entrada opcional de meta-features, que se concatena a la
# representación del texto justo antes de la capa de salida (5.8).

# %%
def capa_embedding(matriz, congelar):
    if matriz is None:
        emb = nn.Embedding(len(vocab), EMB_DIM, padding_idx=0)
        nn.init.uniform_(emb.weight, -0.25, 0.25)
        with torch.no_grad():
            emb.weight[0].zero_()
        return emb
    return nn.Embedding.from_pretrained(torch.tensor(matriz), freeze=congelar, padding_idx=0)


class RedTexto(nn.Module):
    """Parte común: capa de embedding, rama de meta-features y capa de salida."""

    def __init__(self, matriz, congelar, dim_texto, n_clases, dropout, n_meta):
        super().__init__()
        self.emb = capa_embedding(matriz, congelar)
        self.dropout = nn.Dropout(dropout)
        self.meta = nn.Sequential(nn.Linear(n_meta, 16), nn.ReLU()) if n_meta else None
        self.salida = nn.Linear(dim_texto + (16 if n_meta else 0), n_clases)
        self.atencion = None          # aquí se guardan los últimos pesos de atención calculados

    def clasificar(self, rep, meta):
        rep = self.dropout(rep)
        if self.meta is not None:
            rep = torch.cat([rep, self.meta(meta)], dim=1)
        return self.salida(rep)


class TextCNN(RedTexto):
    def __init__(self, matriz, congelar, n_clases, n_filtros=100, tamanos=(3, 4, 5), dropout=0.5, n_meta=0):
        super().__init__(matriz, congelar, n_filtros * len(tamanos), n_clases, dropout, n_meta)
        self.convs = nn.ModuleList([nn.Conv1d(EMB_DIM, n_filtros, k) for k in tamanos])

    def forward(self, x, longitudes, meta=None):
        e = self.emb(x).transpose(1, 2)                                              # (lote, 300, tokens)
        rep = torch.cat([F.relu(conv(e)).amax(dim=2) for conv in self.convs], dim=1)   # max-over-time
        return self.clasificar(rep, meta)


class BiRNN(RedTexto):
    """LSTM o GRU bidireccional. `atencion` puede ser "no", "producto escalar" o "general"."""

    def __init__(self, matriz, congelar, n_clases, celda="LSTM", oculto=128, dropout=0.5,
                 atencion="producto escalar", n_meta=0):
        super().__init__(matriz, congelar, (2 if atencion == "no" else 4) * oculto, n_clases, dropout, n_meta)
        self.rnn = getattr(nn, celda)(EMB_DIM, oculto, batch_first=True, bidirectional=True)
        self.tipo_atencion = atencion
        self.proyeccion = nn.Linear(2 * oculto, 2 * oculto, bias=False) if atencion == "general" else None

    def forward(self, x, longitudes, meta=None):
        paquete = pack_padded_sequence(self.emb(x), longitudes, batch_first=True, enforce_sorted=False)
        salida, h = self.rnn(paquete)
        h = h[0] if isinstance(h, tuple) else h                # la LSTM devuelve (h, c); la GRU, solo h
        estado = torch.cat([h[0], h[1]], dim=1)                 # estado final de cada dirección
        if self.tipo_atencion == "no":
            return self.clasificar(estado, meta)
        H, _ = pad_packed_sequence(salida, batch_first=True, total_length=x.size(1))    # un estado por token
        consulta = estado if self.proyeccion is None else self.proyeccion(estado)
        puntos = torch.bmm(H, consulta.unsqueeze(2)).squeeze(2)                        # 1) similitud
        alfa = torch.softmax(puntos.masked_fill(x == 0, float("-inf")), dim=1)         # 2) pesos (sin el relleno)
        self.atencion = alfa.detach()
        contexto = torch.bmm(alfa.unsqueeze(1), H).squeeze(1)                          # 3) suma ponderada
        return self.clasificar(torch.cat([contexto, estado], dim=1), meta)             # 4) contexto + estado


def crear_red(arq, matriz, congelar, n_clases, hp, n_meta=0):
    if arq == "CNN":
        return TextCNN(matriz, congelar, n_clases, n_filtros=hp["n_filtros"], dropout=hp["dropout"], n_meta=n_meta)
    return BiRNN(matriz, congelar, n_clases, celda=hp["celda"], oculto=hp["oculto"], dropout=hp["dropout"],
                 atencion=hp["atencion"], n_meta=n_meta)


def nombre_arq(arq, hp):
    if arq == "CNN":
        return "CNN"
    sufijo = {"no": "sin atención", "producto escalar": "+ atención", "general": "+ atención general"}
    return f"Bi{hp['celda']} {sufijo[hp['atencion']]}"


# %% [markdown]
# ### 5.4 Entrenamiento
#
# Todas las redes se entrenan igual, para que lo único que cambie entre experimentos sea lo que
# queremos comparar:
#
# - Función de pérdida: entropía cruzada categórica. Optimizador **Adam**, lotes de 64 tickets.
# - ***Early stopping***: tras cada época se mide el F1 macro en validación; si pasan 5 épocas
#   sin mejorar se para y se recupera el mejor modelo. El máximo son 40 épocas.
# - **Dropout** antes de la capa de salida.
# - Cada lote junta tickets de longitud parecida, para rellenar menos y entrenar más rápido.
# - Cada configuración final se entrena con **3 semillas** y se da la media y la desviación
#   típica. Una red cambia de un entrenamiento a otro por la inicialización aleatoria, así que
#   una diferencia más pequeña que esa desviación no significa nada.

# %%
LR, TAM_LOTE, PACIENCIA = 1e-3, 64, 5
PESOS = {t: compute_class_weight("balanced", classes=np.arange(len(c)), y=Y[t]["train"])
         for t, c in TAREAS.items()}


def lote(split, idx, usar_meta):
    """Lote recortado a la longitud de su ticket más largo. `idx` son posiciones (en CPU)."""
    d, en_gpu = DATOS[split], idx.to(DEVICE)
    longitudes = d["L"][idx]
    ancho = max(int(longitudes.max()), 5)            # 5 = filtro más ancho de la CNN
    return d["X"][en_gpu][:, :ancho], longitudes, (d["M"][en_gpu] if usar_meta else None), en_gpu


def lotes_de_train(gen):
    """Baraja train y junta en cada lote tickets de longitud parecida."""
    L = DATOS["train"]["L"]
    orden, trozo, lotes = torch.randperm(len(L), generator=gen), TAM_LOTE * 50, []
    for i in range(0, len(orden), trozo):
        bloque = orden[i:i + trozo]
        bloque = bloque[torch.argsort(L[bloque])]
        lotes += [bloque[j:j + TAM_LOTE] for j in range(0, len(bloque), TAM_LOTE)]
    return [lotes[k] for k in torch.randperm(len(lotes), generator=gen)]


@torch.no_grad()
def predecir(red, split, usar_meta=False, con_atencion=False):
    red.eval()
    n, logits, atenciones = len(DATOS[split]["L"]), [], []
    for i in range(0, n, 512):
        x, longitudes, meta, _ = lote(split, torch.arange(i, min(i + 512, n)), usar_meta)
        logits.append(red(x, longitudes, meta))
        if con_atencion:
            atenciones.append(F.pad(red.atencion, (0, MAX_LEN - x.size(1))))
    logits = torch.cat(logits)
    return (logits, torch.cat(atenciones)) if con_atencion else logits


def entrenar(red, tarea, *, semilla, pesos_clase, usar_meta=False):
    opt = torch.optim.Adam([p for p in red.parameters() if p.requires_grad], lr=LR)
    w = torch.tensor(PESOS[tarea], dtype=torch.float32, device=DEVICE) if pesos_clase else None
    perdida = nn.CrossEntropyLoss(weight=w)
    gen = torch.Generator().manual_seed(semilla)
    n, mejor, historia = len(DATOS["train"]["L"]), {"f1": -1.0, "epoca": 0}, []
    for epoca in range(1, MAX_EPOCHS + 1):
        red.train()
        suma = 0.0
        for idx in lotes_de_train(gen):
            x, longitudes, meta, en_gpu = lote("train", idx, usar_meta)
            opt.zero_grad()
            loss = perdida(red(x, longitudes, meta), YT[tarea]["train"][en_gpu])
            loss.backward()
            nn.utils.clip_grad_norm_(red.parameters(), 5.0)
            opt.step()
            suma += loss.item() * len(idx)
        logits = predecir(red, "val", usar_meta)
        f1 = f1_score(Y[tarea]["val"], logits.argmax(1).cpu().numpy(), average="macro")
        historia.append({"epoca": epoca, "loss_train": suma / n,
                         "loss_val": perdida(logits, YT[tarea]["val"]).item(), "f1_val": f1})
        if f1 > mejor["f1"]:
            mejor = {"f1": f1, "epoca": epoca,
                     "estado": {k: v.detach().clone() for k, v in red.state_dict().items()}}
        elif epoca - mejor["epoca"] >= PACIENCIA:
            break
    red.load_state_dict(mejor["estado"])
    return mejor["f1"], mejor["epoca"], pd.DataFrame(historia)


HISTORIAS, REDES, DEEP_LOG = {}, {}, []


def experimento(tarea, arq, emb, hp, *, usar_meta=False, semillas=SEEDS, apuntar=True):
    """Entrena una configuración con varias semillas. Con `apuntar` la evalúa en test y la registra."""
    matriz, congelar = EMBEDDINGS[emb]
    nombre = f"{nombre_arq(arq, hp)}{' + meta-features' if usar_meta else ''} · {emb}"
    f1s, epocas, preds, t0 = [], [], [], time.time()
    for s in semillas:
        set_seed(s)
        red = crear_red(arq, matriz, congelar, len(TAREAS[tarea]), hp, N_META if usar_meta else 0).to(DEVICE)
        f1, epoca, historia = entrenar(red, tarea, semilla=s, pesos_clase=hp["pesos_clase"], usar_meta=usar_meta)
        f1s.append(f1)
        epocas.append(epoca)
        if apuntar:
            preds.append(predecir(red, "test", usar_meta).argmax(1).cpu().numpy())
            if s == semillas[0]:
                HISTORIAS[(tarea, nombre)], REDES[(tarea, nombre)] = historia, red.cpu()
        del red
    fila = {"tarea": tarea, "modelo": nombre, "arquitectura": arq, "embedding": emb, "meta": usar_meta, **hp,
            "f1_val": float(np.mean(f1s)), "f1_val_std": float(np.std(f1s)),
            "mejor_epoca": float(np.mean(epocas)), "segundos": time.time() - t0}
    if apuntar:
        detalle = ", ".join(f"{k}={v}" for k, v in hp.items()) + f"; mejor época ≈ {np.mean(epocas):.0f}"
        r = registrar(tarea, "Deep", nombre, preds, f1s, detalle)
        fila.update(f1_test=r["f1"], f1_test_std=r["f1_std"], acc_test=r["accuracy"])
        DEEP_LOG.append(fila)
        print(f"{nombre:55s} F1 val {fila['f1_val']:.3f} ± {fila['f1_val_std']:.3f} | "
              f"F1 test {r['f1']:.3f} ± {r['f1_std']:.3f} | época {np.mean(epocas):.0f} | {fila['segundos']:.0f} s")
    return fila


# %% [markdown]
# ### 5.5 Búsqueda de hiperparámetros de las redes
#
# Igual que con los modelos clásicos, una rejilla evaluada en validación. Para que la búsqueda no
# se mezcle con la comparación de embeddings, se hace con una única variante (Google News
# congelado) y una semilla, y los valores elegidos se mantienen fijos en el resto de experimentos.
#
# | Red | Hiperparámetro | Valores |
# |---|---|---|
# | CNN | filtros por tamaño de ventana | 100, 200 |
# | CNN | dropout | 0,3 y 0,5 |
# | CNN | pesos por clase en la función de pérdida | no / sí (`balanced`) |
# | RNN | tipo de celda | LSTM, GRU |
# | RNN | tamaño del estado oculto (por dirección) | 64, 128 |
#
# La RNN tarda bastante más en entrenar, así que su rejilla es más pequeña: hereda de la CNN el
# dropout y la decisión sobre los pesos por clase, que dependen sobre todo del desbalanceo de los
# datos y no de la arquitectura. La tasa de aprendizaje, el tamaño de lote y los tamaños de
# ventana (3, 4, 5) se dejan en valores habituales.

# %%
def buscar(arq, grid):
    """Entrena cada combinación de la rejilla (1 semilla, Google News congelado) y elige por F1 de validación."""
    filas = []
    for hp in (grid[:2] if FAST else grid):
        fila = experimento("queue", arq, "Google News congelado", hp, semillas=SEEDS[:1], apuntar=False)
        filas.append(fila)
        print(f"{arq}: {hp} -> F1 val {fila['f1_val']:.3f} (época {fila['mejor_epoca']:.0f}, {fila['segundos']:.0f} s)")
    filas.sort(key=lambda f: -f["f1_val"])
    display(pd.DataFrame(filas)[[*grid[0], "f1_val", "mejor_epoca", "segundos"]].round(3))
    return {k: filas[0][k] for k in grid[0]}, filas


t0 = time.time()
HP, BUSQUEDA_DEEP = {}, {}
HP["CNN"], BUSQUEDA_DEEP["CNN"] = buscar("CNN", [{"n_filtros": f, "dropout": d, "pesos_clase": w}
                                                 for f in (100, 200) for d in (0.3, 0.5) for w in (False, True)])
HP["RNN"], BUSQUEDA_DEEP["RNN"] = buscar("RNN", [{"celda": c, "oculto": h, "dropout": HP["CNN"]["dropout"],
                                                  "pesos_clase": HP["CNN"]["pesos_clase"], "atencion": "producto escalar"}
                                                 for c in ("LSTM", "GRU") for h in (64, 128)])
ARQS = ["CNN", "RNN"]
NOMBRE_ARQ = {arq: nombre_arq(arq, HP[arq]) for arq in ARQS}
print(f"\nBúsqueda completa en {(time.time() - t0) / 60:.1f} min")
for arq in ARQS:
    print(f"  {NOMBRE_ARQ[arq]}: {HP[arq]}")

# %% [markdown]
# ### 5.6 Comparación de embeddings
#
# Con los hiperparámetros fijados, cada arquitectura se entrena con las cinco variantes de
# embedding y tres semillas. Es el experimento central de la entrega.

# %%
for arq in ARQS:
    for emb in EMBEDDINGS:
        experimento("queue", arq, emb, HP[arq])

COMP = pd.DataFrame([r for r in DEEP_LOG if r["tarea"] == "queue" and not r["meta"]
                     and r["modelo"] == f"{NOMBRE_ARQ[r['arquitectura']]} · {r['embedding']}"])
COMP["red"] = COMP["arquitectura"].map(NOMBRE_ARQ)
display(COMP.pivot(index="embedding", columns="red", values="f1_test").loc[list(EMBEDDINGS)].round(3))

# %%
mejor_shallow_fila = max((r for r in RESULTADOS if r["tarea"] == "queue" and r["familia"] == "Shallow"),
                         key=lambda r: r["f1_val"])
fig, ax = plt.subplots(figsize=(9, 4.6))
alto, y = 0.34, np.arange(len(EMBEDDINGS))
for k, (arq, color) in enumerate(zip(ARQS, SERIE)):
    sub = COMP[COMP["arquitectura"] == arq].set_index("embedding").loc[list(EMBEDDINGS)]
    pos = y + (k - 0.5) * (alto + 0.04)
    ax.barh(pos, sub["f1_test"], height=alto, color=color, label=NOMBRE_ARQ[arq], xerr=sub["f1_test_std"],
            error_kw={"ecolor": COL["tinta2"], "elinewidth": 1, "capsize": 2})
    for p, v, e in zip(pos, sub["f1_test"], sub["f1_test_std"]):
        ax.text(v + e + 0.006, p, f"{v:.3f}", va="center", fontsize=9)
ax.axvline(mejor_shallow_fila["f1"], color=COL["tinta2"], lw=1)
ax.text(mejor_shallow_fila["f1"], -0.72, f"mejor shallow ({mejor_shallow_fila['f1']:.3f})",
        ha="center", va="bottom", fontsize=9, color=COL["tinta2"])
ax.set_yticks(y, list(EMBEDDINGS))
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.tick_params(length=0)
ax.set_xlim(0, max(COMP["f1_test"].max(), mejor_shallow_fila["f1"]) * 1.15)
ax.set_xlabel("F1 macro en test (media de las semillas; la línea fina es la desviación típica)")
ax.set_title("Cola: efecto del embedding en cada arquitectura", pad=18)
ax.legend(loc="lower right")
guardar(fig, "03_comparacion_embeddings")

# %%
# Curvas de aprendizaje de la CNN: qué pasa época a época según el embedding.
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for emb, color in zip(["Aleatorio", "Google News congelado", "Google News fine-tuning"], SERIE):
    h = HISTORIAS[("queue", f"CNN · {emb}")]
    for ax, col in zip(axes, ("loss_train", "f1_val")):
        ax.plot(h["epoca"], h[col], color=color, lw=2, label=emb)
axes[0].set_title("CNN: pérdida en train")
axes[1].set_title("CNN: F1 macro en validación")
for ax in axes:
    ax.set_xlabel("época")
axes[1].legend(loc="lower right")
fig.tight_layout()
guardar(fig, "04_curvas_aprendizaje")

# %% [markdown]
# ### 5.7 ¿Qué aporta la atención?
#
# Se entrena la misma RNN, con su mejor embedding, de tres formas: sin atención (solo el estado
# final), con la atención de producto escalar vista en clase y con la variante general. Además
# del F1 se mide **dónde cae la atención**: qué parte del peso total se llevan el primer y el
# último token del ticket.

# %%
MEJOR_EMB = {arq: COMP[COMP["arquitectura"] == arq].sort_values("f1_val", ascending=False).iloc[0]["embedding"]
             for arq in ARQS}
print("Mejor embedding de cada arquitectura (según validación):", MEJOR_EMB)

emb_rnn = MEJOR_EMB["RNN"]
VARIANTES_ATT = {att: f"{nombre_arq('RNN', {**HP['RNN'], 'atencion': att})} · {emb_rnn}"
                 for att in ("no", "producto escalar", "general")}
for att in ("no", "general"):
    experimento("queue", "RNN", emb_rnn, {**HP["RNN"], "atencion": att})


def atencion_en_test(nombre):
    """Predicción y pesos de atención sobre test de una red ya entrenada (la de la primera semilla)."""
    red = REDES[("queue", nombre)].to(DEVICE)
    logits, alfa = predecir(red, "test", con_atencion=True)
    red.cpu()
    return logits.argmax(1).cpu().numpy(), alfa.cpu().numpy()


L_test = DATOS["test"]["L"].numpy()
filas_test = np.arange(len(L_test))
ATT, filas = {}, []
for att, nombre in VARIANTES_ATT.items():
    r = next(r for r in RESULTADOS if r["tarea"] == "queue" and r["modelo"] == nombre)
    fila = {"variante": nombre_arq("RNN", {**HP["RNN"], "atencion": att}), "accuracy": r["accuracy"],
            "f1": r["f1"], "f1_std": r["f1_std"]}
    if att != "no":
        ATT[att] = atencion_en_test(nombre)
        alfa = ATT[att][1]
        extremos = alfa[filas_test, 0] + np.where(L_test > 1, alfa[filas_test, L_test - 1], 0)
        fila["atención en primer y último token"] = float(extremos.mean())
        fila["peso del token más atendido"] = float(alfa.max(axis=1).mean())
    filas.append(fila)
ATENCION_TABLA = pd.DataFrame(filas)
display(ATENCION_TABLA.round(3))

# %%
import html as _html

# Palabras que más atención reciben en los tickets asignados a cada cola, con cada variante.
# El valor que se promedia es el peso relativo: 1 = lo que tocaría con un reparto uniforme.
MIN_VECES = 3 if FAST else 15


def palabras_atendidas(pred, alfa, k=6):
    suma, veces = {}, {}
    for i, toks in enumerate(TOKENS["test"]):
        n = min(len(toks), MAX_LEN)
        for j in range(n):
            if toks[j].isalpha():
                clave = (pred[i], toks[j])
                suma[clave] = suma.get(clave, 0.0) + alfa[i, j] * n
                veces[clave] = veces.get(clave, 0) + 1
    salida = []
    for c in range(len(TAREAS["queue"])):
        medias = [(w, suma[(cc, w)] / v) for (cc, w), v in veces.items() if cc == c and v >= MIN_VECES]
        salida.append(", ".join(w for w, _ in sorted(medias, key=lambda t: -t[1])[:k]))
    return salida


ATENCION_TOP = pd.DataFrame({"cola": TAREAS["queue"],
                             **{f"atención {att}": palabras_atendidas(*ATT[att]) for att in ATT}})
display(ATENCION_TOP)


# %%
def resaltar(i, att):
    """Ticket de test con cada token sombreado según su peso de atención."""
    pred, alfa = ATT[att]
    toks = TOKENS["test"][i][:MAX_LEN]
    a = alfa[i, :len(toks)] / alfa[i, :len(toks)].max()
    spans = " ".join(f'<span style="background: rgba(42,120,214,{0.6 * x:.2f}); border-radius: 3px; '
                     f'padding: 1px 2px">{_html.escape(t)}</span>' for t, x in zip(toks, a))
    return (f'<p style="line-height: 1.9; color: #0b0b0b; background: #fcfcfb; padding: 8px">'
            f'<b>Atención {att}</b> · real: {TAREAS["queue"][Y["queue"]["test"][i]]} · '
            f'predicho: {TAREAS["queue"][pred[i]]}<br>{spans}</p>')


# Tickets cortos que las dos variantes clasifican bien.
aciertan = np.all([ATT[att][0] == Y["queue"]["test"] for att in ATT], axis=0)
EJEMPLOS_ATT = [int(i) for i in np.flatnonzero(aciertan & (L_test > 20) & (L_test < 60))[:2]]
if not EJEMPLOS_ATT:
    EJEMPLOS_ATT = [int(i) for i in np.flatnonzero(aciertan)[:1]] or [0]
display(HTML("".join(resaltar(i, att) for i in EJEMPLOS_ATT for att in ATT)))

# La misma idea como gráfica: los tokens con más atención de un ticket, con cada variante.
i = EJEMPLOS_ATT[0]
toks = TOKENS["test"][i][:MAX_LEN]
fig, axes = plt.subplots(1, len(ATT), figsize=(11, 3.8))
for ax, att in zip(np.atleast_1d(axes), ATT):
    alfa = ATT[att][1]
    orden = np.argsort(-alfa[i, :len(toks)])[:10]
    barras_h(ax, [f"{toks[j]}  ({j + 1})" for j in orden], [float(alfa[i, j]) for j in orden], fmt="{:.2f}")
    ax.set_title(f"Atención {att}")
    ax.set_xlabel("peso de atención (entre paréntesis, la posición del token)")
fig.suptitle(f"Tokens con más atención en un ticket de «{TAREAS['queue'][Y['queue']['test'][i]]}» "
             f"de {len(toks)} tokens", x=0.01, ha="left", fontsize=11, fontweight="bold")
fig.tight_layout()
guardar(fig, "05_atencion_ejemplo")
ATENCION_EJEMPLO = {"texto": " ".join(toks), "cola": TAREAS["queue"][Y["queue"]["test"][i]], "tokens": len(toks)}

# %% [markdown]
# ### 5.8 Red híbrida: texto + meta-features
#
# La misma idea que en 4.2, ahora dentro de la red: las 12 meta-features pasan por una capa
# pequeña y se concatenan a la representación del texto antes de la capa de salida. Lo probamos
# con la mejor configuración de cada arquitectura.

# %%
for arq in ARQS:
    experimento("queue", arq, MEJOR_EMB[arq], HP[arq], usar_meta=True)
display(tabla("queue").round(3))

# %% [markdown]
# ## 6. Las otras dos salidas: prioridad y tipo
#
# Repetimos el proceso para las otras dos etiquetas de E1. Los modelos clásicos repiten su
# búsqueda en rejilla completa (es barata). Las redes reutilizan los hiperparámetros y el mejor
# embedding encontrados para la cola; optimizarlos para cada tarea queda para la siguiente
# iteración.

# %%
for tarea in ("priority", "type"):
    t0 = time.time()
    GRID_SHALLOW[tarea] = busqueda_shallow(tarea)
    MEJOR_SHALLOW[tarea] = evaluar_shallow(tarea, GRID_SHALLOW[tarea])
    print(f"\n== {NOMBRE_TAREA[tarea]}: grid shallow en {time.time() - t0:.0f} s")
    for arq in ARQS:
        experimento(tarea, arq, MEJOR_EMB[arq], HP[arq])

print("\nPrioridad")
display(tabla("priority", extra=["recall_high"]).round(3))
print("Tipo")
display(tabla("type").round(3))

# %% [markdown]
# ## 7. Tabla comparativa final y análisis de errores

# %%
FINAL = pd.DataFrame(RESULTADOS)
for tarea in TAREAS:
    print(f"\n=== {NOMBRE_TAREA[tarea]} · métricas en test ===")
    display(tabla(tarea, extra=["recall_high"] if tarea == "priority" else []).round(3))

# %%
# Ranking de todos los modelos de la cola. El color indica la familia.
t = FINAL[FINAL["tarea"] == "queue"].sort_values("f1", ascending=False)
color_familia = {"Shallow": COL["azul"], "Deep": COL["naranja"], "Baseline": COL["gris"]}
fig, ax = plt.subplots(figsize=(9.5, 0.34 * len(t) + 1.2))
barras_h(ax, t["modelo"].tolist(), t["f1"].tolist(), color=[color_familia[f] for f in t["familia"]],
         err=t["f1_std"].tolist())
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=f) for f, c in color_familia.items()],
          loc="lower right")
ax.set_xlabel("F1 macro en test")
ax.set_title("Cola: todos los modelos")
guardar(fig, "06_ranking_cola")

# %%
# Matriz de confusión del mejor modelo de la cola (elegido por validación, evaluado en test).
mejor_fila = max((r for r in RESULTADOS if r["tarea"] == "queue"), key=lambda r: r["f1_val"])
MEJOR_MODELO = mejor_fila["modelo"]
y_real, y_pred = Y["queue"]["test"], PREDICCIONES[("queue", MEJOR_MODELO)]
clases = TAREAS["queue"]
cm = confusion_matrix(y_real, y_pred, labels=range(len(clases)), normalize="true")
fig, ax = plt.subplots(figsize=(8.2, 6.6))
im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(clases)), clases, rotation=40, ha="right")
ax.set_yticks(range(len(clases)), clases)
ax.grid(False)
ax.tick_params(length=0)
for a in range(len(clases)):
    for b in range(len(clases)):
        if cm[a, b] >= 0.005:
            ax.text(b, a, f"{cm[a, b]:.2f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if cm[a, b] > 0.55 else COL["tinta"])
ax.set_xlabel("cola predicha")
ax.set_ylabel("cola real")
ax.set_title(f"Matriz de confusión (cada fila suma 1) · {MEJOR_MODELO}")
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).outline.set_visible(False)
guardar(fig, "07_matriz_confusion")

INFORME_CLASES = pd.DataFrame(classification_report(y_real, y_pred, labels=range(len(clases)), target_names=clases,
                                                    output_dict=True, zero_division=0)).T
display(INFORME_CLASES.round(3))

# Las confusiones más frecuentes: qué cola real se confunde con cuál.
errores = pd.DataFrame(confusion_matrix(y_real, y_pred, labels=range(len(clases))), index=clases, columns=clases)
CONFUSIONES = pd.DataFrame([{"real": a, "predicha": b, "tickets": int(errores.loc[a, b])}
                            for a in clases for b in clases if a != b])
CONFUSIONES = CONFUSIONES.sort_values("tickets", ascending=False).head(8).reset_index(drop=True)
display(CONFUSIONES)

# %%
# Se guarda todo lo necesario para el informe.
def limpio(o):
    if isinstance(o, dict):
        return {str(k): limpio(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [limpio(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        o = o.item()
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


resumen = {
    "particiones": {s: int(len(i)) for s, i in IDX.items()},
    "tickets_e2": int(len(df_e2)), "tickets_unicos": int(df_e2["clave"].nunique()),
    "fuga": FUGA.to_dict("records"),
    "clases": {t: df[t].value_counts().to_dict() for t in TAREAS},
    "vocabulario": len(vocab), "tfidf_features": {k: int(v["train"].shape[1]) for k, v in TFIDF.items()},
    "mejor_shallow": {t: {n: {"tfidf": m["tfidf"], "params": m["params"], "f1_val": m["f1_val"]}
                          for n, m in d.items()} for t, d in MEJOR_SHALLOW.items()},
    "grid_shallow_queue": GRID_SHALLOW["queue"].to_dict("records"),
    "cobertura": COBERTURA.to_dict("records"),
    "busqueda_deep": BUSQUEDA_DEEP, "hp_deep": HP, "nombre_arq": NOMBRE_ARQ, "mejor_embedding": MEJOR_EMB,
    "deep_log": DEEP_LOG, "resultados": RESULTADOS, "mejor_modelo_cola": MEJOR_MODELO,
    "informe_clases": INFORME_CLASES.reset_index(names="clase").to_dict("records"),
    "confusiones": CONFUSIONES.to_dict("records"),
    "atencion_tabla": ATENCION_TABLA.to_dict("records"), "atencion_top": ATENCION_TOP.to_dict("records"),
    "atencion_ejemplo": ATENCION_EJEMPLO,
    "meta_features": list(META.columns), "semillas": SEEDS, "max_epocas": MAX_EPOCHS, "paciencia": PACIENCIA,
    "lr": LR, "tam_lote": TAM_LOTE, "fast": FAST,
}
(ART_DIR / "resultados.json").write_text(json.dumps(limpio(resumen), ensure_ascii=False, indent=1), encoding="utf-8")
FINAL.to_csv(ART_DIR / "resultados.csv", index=False)
print("Guardado en", ART_DIR / "resultados.json")
