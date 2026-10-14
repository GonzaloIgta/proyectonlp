import os

import pandas as pd
from datasets import load_dataset, load_from_disk

PATH = os.path.join("data", "tickets")
CSV_PATH = os.path.join(PATH, "raw/dataraw.csv")
PROCESSED_PATH = os.path.join(PATH, "dataprocessed/dataprocessed.csv")


def limpieza_datos(df: pd.DataFrame, output_path: str = PROCESSED_PATH) -> pd.DataFrame:
    """Elimina las filas con 'body' vacío (NaN) y guarda el resultado en CSV."""
    n_antes = len(df)
    df_limpio = df.dropna(subset=["body"]).reset_index(drop=True)
    print(f"Filas eliminadas: {n_antes - len(df_limpio)} (quedan {len(df_limpio)})")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_limpio.to_csv(output_path, index=False)
    return df_limpio


if not os.path.exists(os.path.join(PATH, "dataset_dict.json")):
    load_dataset("Tobi-Bueck/customer-support-tickets").save_to_disk(PATH)

if not os.path.exists(CSV_PATH):
    ds = load_from_disk(PATH)
    # SOLO TIENE TRAIN, POR LO QUE TENEMOS QUE HACER NOSOTROS EL SPLIT
    df = ds["train"].to_pandas()
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
else:
    df = pd.read_csv(CSV_PATH)

df = limpieza_datos(df)
print(df.head())


"""
    PREGUNTAR ARITZBI QUE HAY EN ALEMAN Y INGLES
    NOS QUEDAMOS SOLO CON INGLES?????
    
    TENEMOS TAMBIEN LAS RESPUESTAS, ENTIENDO QUE PARA LA PRIMERA TAREA (CLASIFICAR TICKET ESA COLUMNA LA ELIMINAMOS <- NO VA A EXISTIR RESPUESTA EN EL MOMENTO DE CLASIFICACION)
    
    
    LO IDEAL SERIA PREDECIR QUEQUE (A QUIEN VA PREDECIDO), PRIORITY (IMPORTANCIA), TAGS
"""