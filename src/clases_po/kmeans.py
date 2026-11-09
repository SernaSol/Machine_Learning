from datetime import datetime
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def generar_reporte_kmeans(
    df: pd.DataFrame | None = None,
    k_clusters: int = 3,
    output_filename: str | Path = "resumen_ejecucion.txt",
) -> str:
    """
    Ejecuta el pipeline completo de K-Means utilizando ColumnTransformer
    y genera un informe en archivo de texto (.txt) con las métricas y perfilamiento.
    """
    fecha_ejecucion = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Carga del Dataset
    dataset_id = "shijiabraham/insurance-data"
    if df is None:
        try:
            dataset = load_dataset(dataset_id, split="train")
            df_raw = dataset.to_pandas()
        except Exception:
            url = f"https://huggingface.co/datasets/{dataset_id}/raw/main/insurance.csv"
            df_raw = pd.read_csv(url)
    elif isinstance(df, pd.DataFrame):
        df_raw = df.copy()
    else:
        raise TypeError("df debe ser un pandas.DataFrame o None.")

    shape_inicial = df_raw.shape
    df_clean = df_raw.copy()

    # 2. Preprocesamiento con ColumnTransformer (Coincide con Jupyter Notebook)
    num_cols = ['age', 'bmi', 'children', 'charges']
    cat_cols = ['sex', 'smoker', 'region']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first'), cat_cols)
        ]
    )

    X_prep = preprocessor.fit_transform(df_clean)

    # 3. Entrenamiento del Modelo K-Means
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    df_clean['Cluster'] = kmeans.fit_predict(X_prep)

    # 4. Cálculo de Métricas Evaluativas
    inercia = float(kmeans.inertia_)
    sil_score = float(silhouette_score(X_prep, df_clean['Cluster']))
    db_index = float(davies_bouldin_score(X_prep, df_clean['Cluster']))
    ch_score = float(calinski_harabasz_score(X_prep, df_clean['Cluster']))

    # 5. Generación de Salidas Exactas (Formato Jupyter Consola)
    promedios_df = df_clean.groupby('Cluster')[['age', 'bmi', 'children', 'charges']].mean()
    crosstab_smoker = pd.crosstab(df_clean['Cluster'], df_clean['smoker'], normalize='index') * 100
    counts_series = df_clean['Cluster'].value_counts().sort_index()
    counts_series.name = "count"

    # 6. Construcción de la Estructura del Reporte
    reporte_txt = f"""================================================================================
          RESUMEN AUTOMÁTICO DE EJECUCIÓN - EXPERIMENTO K-MEANS
================================================================================
Fecha y Hora de Ejecución: {fecha_ejecucion}

1. PREGUNTA DE NEGOCIO Y OBJETIVO
--------------------------------------------------------------------------------
Pregunta: ¿Cuáles son los patrones biológicos y de comportamiento subyacentes
          en la cartera de asegurados y cómo se traducen en niveles de gasto médico
          para diseñar coberturas personalizadas y programas preventivos?
Beneficiarios: Gerencia de Salud Preventiva, Suscripción (Underwriting) y Producto.
Decisión: Focalización de programas preventivos y estratificación tarifaria.

2. IDENTIFICACIÓN Y DATOS DEL DATASET
--------------------------------------------------------------------------------
Identificador Hugging Face: {dataset_id}
Enlace: https://huggingface.co/datasets/{dataset_id}
Split Utilizado: train
Registros Totales Analizados: {shape_inicial[0]} filas | {shape_inicial[1]} columnas

3. DESCRIPCIÓN DEL PREPROCESAMIENTO APLICADO
--------------------------------------------------------------------------------
- Transformador General: ColumnTransformer de Scikit-Learn.
- Variables Numéricas ({num_cols}): StandardScaler (Media = 0, Desv. Est. = 1).
- Variables Categóricas ({cat_cols}): OneHotEncoder (drop='first').

4. CONFIGURACIÓN DEL EXPERIMENTO Y PARÁMETROS DEL MODELO
--------------------------------------------------------------------------------
Tipo de Aprendizaje: No Supervisado
Algoritmo: K-Means Clustering
Número de Clústeres (K): {k_clusters}
Semilla Aleatoria (random_state): 42
Inicialización de Centroides (n_init): 10 (k-means++)

5. MÉTRICAS CALCULADAS DE CALIDAD DEL AGRUPAMIENTO
--------------------------------------------------------------------------------
- Inercia Intraclúster (WCSS):         {inercia:.2f}
- Coeficiente de Silueta Promedio:     {sil_score:.3f}
- Índice de Davies-Bouldin:            {db_index:.3f}
- Índice de Calinski-Harabasz:         {ch_score:.2f}

6. CARACTERIZACIÓN FINANCIERA Y BIOLÓGICA POR CLÚSTER (SALIDA JUPYTER)
--------------------------------------------------------------------------------
=== PERFIL PROMEDIO POR CLÚSTER ===
{promedios_df.to_string()}

=== DISTRIBUCIÓN DE FUMADORES POR CLÚSTER (%) ===
{crosstab_smoker.to_string()}

=== CONTEO DE OBSERVACIONES POR CLÚSTER ===
{counts_series.to_string()}

================================================================================
Fin del Reporte Automático
================================================================================
"""

    # 7. Escritura en disco
    ruta_salida = Path(output_filename)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    ruta_salida.write_text(reporte_txt, encoding="utf-8")
    print(f"Reporte escrito correctamente en '{ruta_salida.resolve()}'.")
    
    return reporte_txt


if __name__ == "__main__":
    generar_reporte_kmeans(k_clusters=3)