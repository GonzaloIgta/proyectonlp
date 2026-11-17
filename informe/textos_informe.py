"""Textos interpretativos del informe de E3.

Se redactaron a la vista de los resultados de la ejecución completa. Las cifras se leen del JSON
(`R`), así que si se reejecuta el notebook los números se actualizan solos; si cambiara el sentido de
algún resultado, habría que revisar la redacción.
"""


def TEXTOS(R, n, pct, miles):
    p = lambda s: f"<p>{s}</p>"  # noqa: E731
    res = lambda tarea, modelo: next(r for r in R["resultados"] if r["tarea"] == tarea and r["modelo"] == modelo)  # noqa: E731
    deep = lambda tarea, modelo: next((d for d in R["deep_log"] if d["tarea"] == tarea and d["modelo"] == modelo), None)  # noqa: E731
    por_familia = lambda tarea, fam: sorted((r for r in R["resultados"] if r["tarea"] == tarea and r["familia"] == fam),  # noqa: E731
                                            key=lambda r: -r["f1"])
    NA, ME, HP = R["nombre_arq"], R["mejor_embedding"], R["hp_deep"]
    T = {}

    # ------------------------------------------------------------------ shallow
    g = R["grid_shallow_queue"]

    def mejor_grid(**filtro):
        filas = [x for x in g if all(x.get(k) == v for k, v in filtro.items())]
        return max(x["f1_val"] for x in filas)

    svm_uni = mejor_grid(modelo="SVM lineal", tfidf="unigramas")
    svm_bi = mejor_grid(modelo="SVM lineal", tfidf="uni+bigramas")
    svm_bi_sw = mejor_grid(modelo="SVM lineal", tfidf="uni+bigramas, sin stopwords")
    lr_bal = mejor_grid(modelo="Regresión logística", class_weight="balanced")
    lr_nobal = max(x["f1_val"] for x in g if x["modelo"] == "Regresión logística" and x["class_weight"] is None)
    MS = R["mejor_shallow"]["queue"]
    T["grid_shallow"] = (
        p(f"Las tres búsquedas eligen la misma representación, <b>unigramas + bigramas sin stopwords</b>. Los bigramas son "
          f"lo que más aporta: en la SVM, el mejor F1 de validación pasa de {n(svm_uni)} con solo unigramas a {n(svm_bi)} "
          f"con bigramas, porque hay expresiones que señalan el departamento mucho mejor que sus palabras por separado: "
          f"todos los tickets con <i>«billing discrepancy»</i> son de <i>Billing and Payments</i> y el 89 % de los que "
          f"dicen <i>«service outage»</i> van a <i>Service Outages and Maintenance</i>. Quitar stopwords ayuda un poco "
          f"más ({n(svm_bi_sw)}), ya que con "
          f"bigramas evita pares vacíos como <i>«of the»</i>.")
        + p(f"La regularización elegida es débil: C = {n(MS['Regresión logística']['params']['C'], 0)} en la regresión "
            f"logística, C = {n(MS['SVM lineal']['params']['C'], 0)} en la SVM y alpha = "
            f"{n(MS['Naive Bayes']['params']['alpha'])} en Naive Bayes. Es decir, conviene dejar que el modelo se ajuste "
            f"mucho a train. Nuestra interpretación es que el dataset es sintético y repite plantillas de redacción, así "
            f"que memorizar expresiones concretas también sirve en validación. Ponderar las clases (<code>balanced</code>) "
            f"mejora la regresión logística ({n(lr_nobal)} → {n(lr_bal)}), como cabía esperar con la cola tan desbalanceada.")
    )

    svm, lr, nb = res("queue", "TF-IDF + SVM lineal"), res("queue", "TF-IDF + Regresión logística"), res("queue", "TF-IDF + Naive Bayes")
    meta_sola = res("queue", "Solo meta-features + Regresión logística")
    base = next(r for r in R["resultados"] if r["tarea"] == "queue" and r["familia"] == "Baseline")
    meta_mas = next(r for r in R["resultados"] if r["tarea"] == "queue" and r["modelo"].startswith("TF-IDF + meta-features"))
    T["shallow"] = (
        p(f"La <b>SVM lineal</b> es el mejor modelo shallow (F1 macro {n(svm['f1'])}), seguida de cerca por la regresión "
          f"logística ({n(lr['f1'])}). Naive Bayes queda por detrás ({n(nb['f1'])}) con un perfil distinto: tiene la "
          f"precision más alta ({n(nb['precision'])}) pero el recall más bajo ({n(nb['recall'])}). Cuando asigna una cola "
          f"pequeña suele acertar, pero deja escapar muchos tickets de esas colas hacia las grandes.")
        + p(f"Las meta-features no aportan a la cola. Solas apenas superan a la clase mayoritaria (F1 macro "
            f"{n(meta_sola['f1'])} frente a {n(base['f1'])}; su accuracy es más baja porque, con las clases ponderadas, "
            f"reparten predicciones entre las colas pequeñas), y añadidas al TF-IDF dejan el F1 igual "
            f"({n(meta_mas['f1'])}). La búsqueda elige además el peso más pequeño probado para ellas, que equivale a casi no "
            f"usarlas. Tiene sentido: la longitud o la puntuación de un ticket no dicen a qué departamento va.")
    )

    # ------------------------------------------------------------------ deep
    cob = {c["embedding"]: c for c in R["cobertura"]}
    T["cobertura"] = p(
        f"Google News tiene vector para el {pct(cob['Google News']['tipos del vocabulario con vector'])} de las palabras "
        f"del vocabulario, en línea con lo que medimos en E2. Por tokens la cobertura baja al "
        f"{pct(cob['Google News']['tokens de test con vector'])} porque aquí contamos también los signos de puntuación, "
        f"que Google News no incluye y que las redes sí reciben. El Word2Vec propio, entrenado con el mismo texto, tiene "
        f"vector para el {pct(cob['Word2Vec propio']['tipos del vocabulario con vector'])} del vocabulario; el "
        f"{pct(1 - cob['Word2Vec propio']['tokens de test con vector'])} de tokens de test que se queda sin vector son "
        f"palabras que no aparecen en train (<code>&lt;UNK&gt;</code>).")

    bq = R["busqueda_deep"]
    cnn_mejor, cnn_peor = bq["CNN"][0], bq["CNN"][-1]
    cnn200 = [x["f1_val"] for x in bq["CNN"] if x["n_filtros"] == 200]
    T["busqueda_deep"] = (
        p(f"En la CNN, el factor que más pesa es el número de filtros: las cuatro configuraciones con 200 filtros superan a "
          f"las cuatro con 100. Con 200 filtros, el dropout y los pesos por clase mueven el F1 menos de dos puntos (entre "
          f"{n(min(cnn200))} y {n(max(cnn200))}); gana la combinación con dropout {n(HP['CNN']['dropout'], 1)} y sin pesos "
          f"por clase (F1 de validación {n(cnn_mejor['f1_val'])}, frente a {n(cnn_peor['f1_val'])} de la peor).")
        + p(f"En la RNN, la GRU y la LSTM quedan muy cerca y gana la de estado oculto más grande; el mejor resultado es una "
            f"<b>{NA['RNN']}</b> de {HP['RNN']['oculto']} unidades. Su F1 de validación ({n(bq['RNN'][0]['f1_val'])}) "
            f"queda muy por debajo de la CNN con el mismo embedding.")
        + p("En las dos redes el mejor valor de capacidad (200 filtros, 128 unidades) es el mayor que probamos, así que "
            "podrían mejorar algo con más capacidad. No lo ampliamos en esta iteración por tiempo de cómputo; queda "
            "anotado como primer paso de la siguiente.")
    )

    d = lambda arq, emb, tarea="queue": deep(tarea, f"{NA[arq]} · {emb}")  # noqa: E731
    c_al, c_gc, c_gf = d("CNN", "Aleatorio"), d("CNN", "Google News congelado"), d("CNN", "Google News fine-tuning")
    c_wc, c_wf = d("CNN", "Word2Vec propio congelado"), d("CNN", "Word2Vec propio fine-tuning")
    r_al, r_gc, r_gf = d("RNN", "Aleatorio"), d("RNN", "Google News congelado"), d("RNN", "Google News fine-tuning")
    r_wc, r_wf = d("RNN", "Word2Vec propio congelado"), d("RNN", "Word2Vec propio fine-tuning")
    std_max_cnn = max(x["f1_test_std"] for x in (c_al, c_gc, c_gf, c_wc, c_wf))
    T["embeddings"] = (
        p(f"<b>Embeddings preentrenados.</b> En la CNN, partir de vectores preentrenados mejora unos dos puntos al "
          f"embedding aleatorio (de {n(c_al['f1_test'])} a {n(c_gc['f1_test'])}–{n(c_gf['f1_test'])}), una diferencia "
          f"claramente mayor que la variación entre semillas (desviación típica de {n(std_max_cnn)} como mucho). Con 16.620 "
          f"tickets de entrenamiento, la red no tiene datos suficientes para aprender desde cero lo que Word2Vec ya sabe "
          f"sobre qué palabras se parecen. Entre Google News y nuestro Word2Vec las diferencias son menores que el "
          f"ruido entre semillas, salvo el Word2Vec propio congelado, que queda algo por debajo ({n(c_wc['f1_test'])}).")
        + p(f"<b>Congelar o hacer fine-tuning.</b> Con Google News da igual ({n(c_gc['f1_test'])} congelado y "
            f"{n(c_gf['f1_test'])} con fine-tuning): sus vectores, entrenados con unos 100.000 millones de palabras, ya "
            f"son buenos. Con nuestro Word2Vec, entrenado con solo 2,2 millones de tokens, el fine-tuning sí ayuda "
            f"({n(c_wc['f1_test'])} → {n(c_wf['f1_test'])}) porque corrige vectores menos pulidos. El fine-tuning también "
            f"converge antes: la CNN con Google News alcanza su mejor época hacia la {c_gf['mejor_epoca']:.0f} con "
            f"fine-tuning y hacia la {c_gc['mejor_epoca']:.0f} congelada.")
        + p(f"<b>La RNN.</b> La {NA['RNN']} queda entre doce y catorce puntos por debajo de la CNN con cada embedding (entre "
            f"{n(min(x['f1_test'] for x in (r_al, r_gc, r_gf, r_wc, r_wf)))} y "
            f"{n(max(x['f1_test'] for x in (r_al, r_gc, r_gf, r_wc, r_wf)))}). En ella los embeddings congelados no "
            f"aportan nada frente a los aleatorios ({n(r_gc['f1_test'])} y {n(r_wc['f1_test'])} frente a "
            f"{n(r_al['f1_test'])}) y lo que más ayuda es el fine-tuning, sobre todo del Word2Vec propio "
            f"({n(r_wf['f1_test'])}, su mejor resultado). Nuestra interpretación es que la RNN tiene que resumir tickets de "
            f"unos 70 tokens de media en un único vector, mientras que a la CNN le basta con detectar unos pocos n-gramas "
            f"característicos, que es justo lo que distingue un departamento de otro.")
        + p("Ninguna red supera a la SVM lineal sobre TF-IDF (línea vertical de la figura 3). Lo comentamos en el apartado 5.")
    )
    ep_max = max((x for arq in ("CNN", "RNN") for x in bq[arq]), key=lambda x: x["mejor_epoca"])
    T["curvas"] = p(
        f"Las curvas explican por qué hace falta el <i>early stopping</i>. Con embeddings aleatorios o con fine-tuning, "
        f"la pérdida de train cae casi a cero en unas doce épocas mientras el F1 de validación se estanca hacia 0,62: a "
        f"partir de ahí la red memoriza train sin generalizar mejor. Con Google News congelado hay menos parámetros que "
        f"ajustar, la pérdida baja más despacio y el F1 de validación sigue subiendo durante más épocas. Por eso subimos "
        f"el máximo a {R['max_epocas']} épocas: en la búsqueda de hiperparámetros, una de las configuraciones con "
        f"embeddings congelados llegó a su mejor resultado en la época {ep_max['mejor_epoca']:.0f}, que con el límite "
        f"inicial de 40 no habría alcanzado.")

    at = {a["variante"]: a for a in R["atencion_tabla"]}
    sin = next(a for k, a in at.items() if "sin atención" in k)
    dot = next(a for k, a in at.items() if k.endswith("+ atención"))
    gen = next(a for k, a in at.items() if "general" in k)
    stds = [a["f1_std"] for a in (sin, dot, gen)]
    f1s = [a["f1"] for a in (sin, dot, gen)]
    T["atencion"] = (
        p(f"<b>En el F1 no hay diferencias.</b> Las tres variantes quedan entre {n(min(f1s))} y {n(max(f1s))}, dentro de "
          f"la variación entre semillas (desviación típica de {n(min(stds))} a {n(max(stds))}). En esta tarea la atención "
          f"no mejora la clasificación. La figura de la Unidad 3 sobre traducción muestra que su ventaja crece con la "
          f"longitud de la frase; nuestros tickets son cortos (unos 70 tokens) y, como se vio en 3.6, lo que limita a la "
          f"RNN no parece ser el cuello de botella del estado final.")
        + p(f"<b>Donde sí se nota es en qué mira la red.</b> Las dos atenciones son muy selectivas: de media, el token más "
            f"atendido se lleva el {pct(dot['peso del token más atendido'], 0)} del peso con producto escalar y el "
            f"{pct(gen['peso del token más atendido'], 0)} con la variante general. La red decide casi siempre por una sola "
            f"palabra, como <i>«payment»</i> en el ticket de la figura 5.")
        + p(f"Además, la atención por producto escalar dedica el {pct(dot['atención en primer y último token'])} del peso "
            f"al primer y al último token, frente al {pct(gen['atención en primer y último token'])} de la general (en un "
            f"ticket de longitud media, un reparto uniforme les daría un 3 %). Es un efecto de usar el estado final como "
            f"consulta: el estado del primer token contiene el estado final de la lectura hacia atrás, y el del último "
            f"token, el de la lectura hacia delante, es decir, cada uno contiene la mitad de la propia consulta. Su "
            f"producto escalar con ella sale alto aunque esas palabras no digan nada del ticket. La matriz <i>W</i> de la "
            f"variante general permite a la red evitarlo.")
    )
    T["atencion_palabras"] = p(
        "En varias colas, la atención se fija en las palabras que cualquiera asociaría al departamento: "
        "<i>billing</i>, <i>invoice</i> y <i>payment</i> en <i>Billing and Payments</i>; <i>outage</i>, "
        "<i>maintenance</i> y <i>disruption</i> en <i>Service Outages and Maintenance</i>; <i>connectivity</i>, "
        "<i>downtime</i> y <i>crash</i> en <i>IT Support</i>. En <i>Returns and Exchanges</i> solo la variante general "
        "encuentra las palabras obvias (<i>returns</i>, <i>return</i>). En las dos colas más pequeñas, <i>General "
        "Inquiry</i> y <i>Human Resources</i>, la red atiende sobre todo a palabras vacías (<i>in</i>, <i>on</i>, "
        "<i>the</i>, <i>this</i>): no ha encontrado ninguna palabra que las identifique.")

    cm_ = deep("queue", f"{NA['CNN']} + meta-features · {ME['CNN']}")
    cs_ = deep("queue", f"{NA['CNN']} · {ME['CNN']}")
    rm_ = deep("queue", f"{NA['RNN']} + meta-features · {ME['RNN']}")
    rs_ = deep("queue", f"{NA['RNN']} · {ME['RNN']}")
    T["meta_deep"] = p(
        f"En la CNN, añadir las meta-features sube el F1 de test de {n(cs_['f1_test'])} a {n(cm_['f1_test'])}, pero en "
        f"validación lo baja ({n(cs_['f1_val'])} → {n(cm_['f1_val'])}). Como los modelos se eligen por validación y ahí no "
        f"mejora, no lo damos por una mejora real: con {miles(R['particiones']['test'])} tickets por conjunto, una "
        f"diferencia de un punto puede deberse a qué tickets han caído en cada uno. En la {NA['RNN']} no cambia nada "
        f"({n(rs_['f1_test'])} → {n(rm_['f1_test'])}). Es coherente con lo visto en 2.2: la forma del ticket apenas dice a "
        f"qué departamento va.")

    # ------------------------------------------------------------------ prioridad y tipo
    def top2(tarea):
        filas = sorted((r for r in R["resultados"] if r["tarea"] == tarea and r["familia"] != "Baseline"), key=lambda r: -r["f1"])
        return filas[0], next(r for r in filas if r["familia"] != filas[0]["familia"])

    pr1, pr2 = top2("priority")
    ty1, ty2 = top2("type")
    pr_cnn = deep("priority", f"{NA['CNN']} · {ME['CNN']}")
    ty_cnn = deep("type", f"{NA['CNN']} · {ME['CNN']}")
    pr_rnn = deep("priority", f"{NA['RNN']} · {ME['RNN']}")
    ty_rnn = deep("type", f"{NA['RNN']} · {ME['RNN']}")
    T["prioridad"] = (
        p(f"La prioridad es bastante más difícil que el tipo. El mejor F1 macro es {n(pr1['f1'])} ({pr1['modelo']}), con "
          f"{pr2['modelo']} prácticamente empatado ({n(pr2['f1'])}; la diferencia es del orden de la desviación típica "
          f"entre semillas de la CNN, {n(pr_cnn['f1_test_std'])}). Los dos recuperan en torno al "
          f"{pct(pr1['recall_high'], 0)} de los tickets de prioridad alta. Dicho al revés, más de uno de cada cuatro "
          f"tickets urgentes acaba clasificado como de prioridad media o baja, que es justo el error más caro según E1.")
        + p("La urgencia depende de cosas como cuánta gente está afectada o qué parte del servicio ha caído, más que de "
            "palabras concretas, y eso la hace más difícil para todos los modelos. Mejorar el recall de <i>high</i> es "
            "uno de los objetivos de la siguiente iteración (por ejemplo, desplazando el umbral de decisión de esa clase).")
    )
    T["tipo"] = p(
        f"El tipo es la tarea más fácil: F1 macro {n(ty1['f1'])} con {ty1['modelo']} y {n(ty2['f1'])} con "
        f"{ty2['modelo']}. Aquí la CNN sí supera a la SVM por un punto, unas tres veces su desviación entre semillas "
        f"({n(ty_cnn['f1_test_std'])}). Nuestra interpretación es que el tipo (incidencia, petición, problema o cambio) "
        f"se expresa con fórmulas muy repetidas en la redacción del ticket, que la CNN detecta bien como n-gramas. La "
        f"{NA['RNN']} vuelve a quedar última entre los modelos entrenados ({n(ty_rnn['f1_test'])} en tipo y "
        f"{n(pr_rnn['f1_test'])} en prioridad).")

    # ------------------------------------------------------------------ comparativa y errores
    q = sorted((r for r in R["resultados"] if r["tarea"] == "queue"), key=lambda r: -r["f1"])
    q_cnn = max((r for r in q if r["modelo"].startswith("CNN") and "meta" not in r["modelo"]), key=lambda r: r["f1"])
    q_rnn = max((r for r in q if r["modelo"].startswith(NA["RNN"].split()[0]) and "meta" not in r["modelo"]), key=lambda r: r["f1"])
    T["comparativa"] = (
        p(f"En la cola, los dos mejores modelos son lineales sobre TF-IDF: la SVM ({n(svm['f1'])}) y la regresión "
          f"logística ({n(lr['f1'])}). La mejor CNN queda a unos cuatro puntos ({n(q_cnn['f1'])}) y la mejor "
          f"{NA['RNN'].split()[0]} a más de quince ({n(q_rnn['f1'])}). Todos los modelos que usan el texto tienen más "
          f"precision que recall en promedio macro (la SVM, {n(svm['precision'])} frente a {n(svm['recall'])}): son "
          f"conservadores con las colas pequeñas, que predicen poco pero acertando.")
        + p("Que las redes no ganen en la cola tiene, a nuestro juicio, tres causas. Primero, 16.620 tickets de "
            "entrenamiento son pocos para que una red aprenda una representación mejor que los bigramas que TF-IDF ya "
            "cuenta. Segundo, la cola depende sobre todo de palabras y expresiones clave, como muestra que la atención se "
            "concentre en una sola palabra por ticket, y eso lo captura directamente un modelo lineal sobre bigramas. "
            "Tercero, las redes están menos ajustadas: su rejilla es más pequeña y su mejor capacidad cae en el extremo "
            "de lo probado. En prioridad y tipo, en cambio, la CNN iguala o supera a la SVM, así que no hay una familia "
            "que gane en todas las tareas.")
    )

    best = next(r for r in R["resultados"] if r["tarea"] == "queue" and r["modelo"] == R["mejor_modelo_cola"])
    cl = {c["clase"]: c for c in R["informe_clases"]}
    colas = [c for c in cl if c not in ("accuracy", "macro avg", "weighted avg")]
    orden = sorted(colas, key=lambda c: -cl[c]["f1-score"])
    soporte = {"Technical Support", "Product Support", "IT Support", "Customer Service"}
    conf_soporte = sum(1 for c in R["confusiones"] if c["real"] in soporte and c["predicha"] in soporte)
    gi = cl["General Inquiry"]
    T["errores"] = (
        p(f"El modelo elegido por validación es <i>{best['modelo']}</i>. En la práctica es la SVM sola: la búsqueda dio a "
          f"las meta-features el peso más pequeño y sus métricas de test coinciden con las de la SVM sin ellas.")
        + p(f"Las colas mejor resueltas son las que tienen un vocabulario propio: <i>{orden[0]}</i> (F1 "
            f"{n(cl[orden[0]]['f1-score'])}) y <i>{orden[1]}</i> ({n(cl[orden[1]]['f1-score'])}). Los errores se concentran "
            f"en cuatro colas que se solapan, <i>Technical Support</i>, <i>Product Support</i>, <i>IT Support</i> y "
            f"<i>Customer Service</i>: {conf_soporte} de las {len(R['confusiones'])} confusiones más frecuentes son entre "
            f"ellas. Su contenido se parece mucho; por ejemplo, de los tickets del dataset que mencionan "
            f"<i>«data breach»</i>, el 38 % va a <i>Technical Support</i>, el 20 % a <i>Product Support</i> y el 12 % a "
            f"<i>Customer Service</i>. Si el mismo tipo de problema se reparte entre colas, ningún modelo basado solo en el "
            f"texto puede acertarlas todas: parte del error viene de la ambigüedad de las propias etiquetas.")
        + p(f"<i>General Inquiry</i>, la cola más pequeña ({miles(gi['support'])} tickets en test), tiene precision alta "
            f"({n(gi['precision'])}) pero recall bajo ({n(gi['recall'])}): el modelo casi nunca la predice y, cuando lo "
            f"hace, acierta, pero más de la mitad de sus tickets acaban en otras colas.")
    )

    # ------------------------------------------------------------------ conclusiones
    f0 = R["fuga"]
    T["resumen"] = p(
        f"El mejor modelo para la cola es la <b>SVM lineal sobre TF-IDF</b> con unigramas y bigramas (F1 macro "
        f"{n(svm['f1'])} en test). La mejor red, una CNN con embeddings de Google News, se queda en {n(q_cnn['f1'])} en la "
        f"cola, aunque iguala a la SVM en prioridad ({n(pr_cnn['f1_test'])}) y la supera en tipo ({n(ty_cnn['f1_test'])}). "
        f"Por el camino corregimos una fuga de información de E2: casi un tercio de los tickets estaba duplicado entre "
        f"dos ficheros del dataset, lo que inflaba la accuracy unos {n(100 * (f0[0]['accuracy'] - f0[1]['accuracy']), 0)} "
        f"puntos.")
    T["conclusiones"] = (
        "<ol>"
        f"<li><b>Datos.</b> La deduplicación era imprescindible: con los duplicados de E2, uno de cada cuatro tickets de "
        f"test estaba también en train. Todos los resultados de esta entrega están medidos sin esa fuga y con una "
        f"partición de validación separada de test.</li>"
        f"<li><b>Shallow learning.</b> La SVM lineal sobre TF-IDF con unigramas y bigramas sin stopwords es el mejor "
        f"modelo para la cola (F1 macro {n(svm['f1'])}) y un baseline muy difícil de batir. Los bigramas son la decisión de "
        f"representación que más aporta, y la regularización óptima es débil.</li>"
        f"<li><b>Deep learning.</b> La CNN funciona claramente mejor que la RNN en este problema (unos {n(q_cnn['f1'], 2)} "
        f"frente a {n(q_rnn['f1'], 2)} en la cola). Los embeddings preentrenados aportan unos dos puntos frente a los "
        f"aleatorios y el fine-tuning solo ayuda cuando los vectores de partida son pobres (nuestro Word2Vec, entrenado "
        f"con poco texto).</li>"
        f"<li><b>Atención.</b> No mejora el F1, pero permite ver qué decide la clasificación: la red se fija casi siempre "
        f"en una sola palabra clave. Además, detectamos que la atención por producto escalar con el estado final como "
        f"consulta se desvía hacia el primer y el último token, y que la variante general lo corrige.</li>"
        f"<li><b>Features no textuales.</b> Las meta-features de forma no aportan a la cola, ni en los modelos lineales ni "
        f"en las redes.</li>"
        f"<li><b>Prioridad y tipo.</b> El tipo se resuelve bien (F1 {n(ty1['f1'])}); la prioridad no tanto (F1 "
        f"{n(pr1['f1'])}, con un {pct(pr1['recall_high'], 0)} de recall en la clase urgente).</li>"
        f"<li><b>Límite del problema.</b> Gran parte del error de la cola está en cuatro departamentos de soporte cuyos "
        f"tickets se parecen mucho entre sí; probablemente una parte no se pueda resolver solo con el texto.</li>"
        "</ol>")
    T["siguientes"] = (
        "<ul>"
        "<li>Dar más capacidad a las redes (más filtros y estado oculto más grande, porque el óptimo cayó en el extremo de "
        "la rejilla) y ajustar sus hiperparámetros para prioridad y tipo, que ahora heredan los de la cola.</li>"
        "<li>Probar <b>Transformers</b> (Unidad 4): hacer fine-tuning de un modelo preentrenado tipo BERT o DistilBERT, "
        "como los proyectos sobre este mismo dataset que citamos en E1.</li>"
        "<li><b>Pre-entrenar con datasets parecidos</b>, que el enunciado valora y no hemos hecho en esta iteración: por "
        "ejemplo, entrenar el Word2Vec o la propia CNN con las conversaciones de soporte de Twitter que usaremos para "
        "TweetSumm, y medir si mejora la clasificación.</li>"
        "<li>Mejorar el recall de la prioridad alta ajustando el umbral de decisión o el peso de esa clase.</li>"
        "<li>Tratar las cuatro colas de soporte de forma jerárquica (primero «soporte» frente al resto y después cuál), "
        "en la línea de Zangari et al. (2023), que revisamos en E1.</li>"
        "<li><b>Segunda tarea del proyecto: resumen.</b> Abordar TweetSumm como resumen de conversaciones con un modelo "
        "sequence-to-sequence con atención (Unidad 3), evaluado con ROUGE. Requiere cruzar sus anotaciones con el texto "
        "de los tweets del dataset <i>Customer Support on Twitter</i>.</li>"
        "</ul>")

    PENDIENTE = p("<i>[Pendiente de redactar con los resultados definitivos.]</i>")
    for k in ("resumen", "embeddings", "curvas", "atencion", "atencion_palabras", "meta_deep", "prioridad", "tipo",
              "comparativa", "errores", "conclusiones", "siguientes"):
        T.setdefault(k, PENDIENTE)
    return T
