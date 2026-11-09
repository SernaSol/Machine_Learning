# Machine_Learning

## Participantes Grupo *AE_Grupo 2*

- Sol Elena Serna Vargas
- Juan Esteban Ospina Lopez


# Actividad 3 – Modelo de Aprendizaje No Supervisado K-Means

## 0. Pregunta de negocio

¿Cuáles son los arquetipos naturales de asegurados presentes en la cartera en función de sus características demográficas, biológicas y de estilo de vida (edad, IMC, número de dependientes y hábito de fumar), y cómo varían sus costos médicos registrados para diseñar programas de salud preventiva focalizados y apoyar el diseño de pólizas?

El modelo desarrollado pertenece al **aprendizaje no supervisado**, utilizando el algoritmo de agrupamiento **K-Means**.

El problema no busca predecir una categoría previamente definida, como «cliente de alto riesgo» o «cliente de bajo riesgo», ni estimar directamente un valor futuro. Su objetivo es descubrir patrones y agrupaciones en una población de asegurados sin utilizar etiquetas de clasificación.

El algoritmo evalúa la similitud de las observaciones mediante distancias en un espacio de características transformadas y estandarizadas, permitiendo identificar clústeres a partir de la combinación de variables demográficas, biológicas, de comportamiento y financieras.

### Variables utilizadas para el modelo no supervisado

**Variables numéricas:**
- `age`: edad.
- `bmi`: índice de masa corporal.
- `children`: número de hijos o dependientes.
- `charges`: gastos médicos registrados.

**Variables categóricas:**
- `sex`: sexo.
- `smoker`: hábito de fumar.
- `region`: región geográfica.

### ¿Quiénes se benefician?

Los principales beneficiarios potenciales son la Gerencia de Salud Preventiva, el Equipo de Suscripción y la Dirección de Producto de una compañía aseguradora.

Los resultados pueden servir como insumo exploratorio para identificar perfiles de asegurados, orientar programas preventivos y analizar diferencias en los gastos médicos.

### Justificación de la pregunta planteada

El conjunto de datos permite abordar la pregunta de negocio porque reúne características personales, biológicas, conductuales y financieras de los asegurados.

Las variables `age`, `bmi`, `children` y `smoker` permiten caracterizar diferencias entre las personas. Adicionalmente, `sex` y `region` aportan información para la segmentación.

La variable `charges` representa los gastos médicos registrados y permite comparar los niveles de costo entre los grupos identificados.

Es importante aclarar que, en la implementación actual, **`charges` también participa en el entrenamiento de K-Means**. Por tanto, los gastos médicos influyen en la construcción de los clústeres y no constituyen una variable de validación financiera completamente independiente.

### Selección del número de clústeres

El algoritmo K-Means agrupa observaciones según su similitud, asignándolas al centroide más cercano y actualizando iterativamente los centroides.

Para evaluar el número de grupos se utilizaron el método del codo y el coeficiente de silueta, considerando valores de K entre 2 y 10.

En el análisis exploratorio, las visualizaciones favorecieron **K=4**. Sin embargo, se decidió implementar el modelo final con **K=3**, buscando una segmentación más sencilla de interpretar y evitando una posible fragmentación excesiva de los grupos.

Esta decisión representa un compromiso entre interpretabilidad y calidad de agrupamiento, pero no demuestra por sí sola que K=3 sea estadísticamente superior.

Una posible limitación es que, al utilizar menos grupos, personas con características diferentes podrían quedar agrupadas en un mismo clúster, reduciendo el nivel de detalle disponible para el análisis.

### Interpretación de los grupos

El modelo final genera tres clústeres, identificados como 0, 1 y 2.

Se construyen visualizaciones mediante PCA para representar los grupos en dos dimensiones, así como gráficos comparativos de edad, IMC, gastos médicos y hábito de fumar.

La interpretación de cada clúster debe realizarse utilizando los promedios, frecuencias y distribuciones obtenidos en la ejecución del modelo.

Los colores de las gráficas sirven para diferenciar visualmente los grupos y no representan, por sí mismos, niveles de riesgo.

### Perfilamiento estadístico

Finalmente, se realiza un perfilamiento estadístico que permite identificar:

- Número de asegurados asignados a cada clúster.
- Porcentaje de fumadores y no fumadores por grupo.
- Edad promedio de los asegurados.
- IMC promedio.
- Número promedio de dependientes.
- Gastos médicos promedio.

Esta caracterización permite comparar los perfiles encontrados y explorar posibles aplicaciones en programas preventivos y análisis de costos.

## 2. Dataset utilizado para el modelo K-Means

**Nombre:** Medical Insurance

**Identificador:** `shijiabraham/insurance-data`

**Fuente:** https://huggingface.co/datasets/shijiabraham/insurance-data

**Split:** `train`

**Registros originales:** 1.338

**Columnas originales:** 7

El dataset contiene información sobre características personales y gastos médicos registrados.

| Columna | Descripción |
|---|---|
| `age` | Edad de la persona |
| `sex` | Sexo |
| `bmi` | Índice de masa corporal |
| `children` | Número de hijos o dependientes |
| `smoker` | Indica si la persona fuma |
| `region` | Región de residencia |
| `charges` | Gastos médicos registrados |

Los datos se obtienen automáticamente desde Hugging Face mediante la clase `HFDatasetLoader`, sin necesidad de realizar una descarga manual previa.

## 3. Preprocesamiento

Inicialmente se analizaron las dimensiones del dataset, los tipos de datos, los valores faltantes y los registros duplicados.

| Indicador | Antes | Después |
|---|---:|---:|
| Registros | 1.338 | 1.337 |
| Duplicados exactos | 1 | 0 |
| Valores faltantes | 0 | 0 |

Se identificó y eliminó un registro duplicado exacto. No fue necesario realizar imputación de valores faltantes.

Posteriormente, se aplicaron las siguientes transformaciones:

**StandardScaler:** estandarización de las variables numéricas `age`, `bmi`, `children` y `charges`.

**OneHotEncoder:** transformación de las variables categóricas `sex`, `smoker` y `region`, utilizando `drop='first'`.

**ColumnTransformer:** integración de las transformaciones numéricas y categóricas para construir la matriz de características utilizada por K-Means.

La estandarización es importante porque K-Means utiliza distancias euclidianas y las diferencias de escala podrían hacer que algunas variables numéricas dominaran el agrupamiento.

## 4. Entrenamiento y evaluación del modelo

Se implementó K-Means con la siguiente configuración:

| Parámetro | Valor |
|---|---|
| Algoritmo | K-Means |
| Tipo de aprendizaje | No supervisado |
| Número de clústeres | 3 |
| `random_state` | 42 |
| `n_init` | 10 |
| Valores de K evaluados | 2 a 10 |

Se utilizaron las siguientes métricas para evaluar el agrupamiento:

- **Inercia (WCSS):** mide la suma de las distancias cuadráticas de las observaciones respecto al centroide asignado.
- **Coeficiente de silueta:** evalúa la cohesión y separación de los grupos.
- **Davies-Bouldin:** permite comparar la separación y dispersión entre clústeres.
- **Calinski-Harabasz:** compara la dispersión entre grupos con la dispersión interna.

Los valores calculados se generan automáticamente durante la ejecución del modelo y se incorporan al reporte de resultados.

## 5. Visualización de resultados

El notebook `EA3_Modelo_No_Supervisor.ipynb` genera visualizaciones para analizar y comparar los tres clústeres.

Se incluyen:

- Método del codo y coeficiente de silueta.
- Representación bidimensional mediante PCA.
- Distribución de gastos médicos por clúster.
- Distribución de edad por clúster.
- Distribución del índice de masa corporal por clúster.
- Gastos médicos promedio según el hábito de fumar.

Las gráficas se almacenan en:

`src/carga_datos/datos/graficas/`

Archivos generados:

- `elbow_silhouette.png`
- `pca_clusters.png`
- `cluster_feature_distributions.png`

Las visualizaciones permiten identificar diferencias entre los grupos, aunque no establecen por sí solas relaciones causales entre las características de los asegurados y los gastos médicos.

## 6. Generación automática del reporte K-Means

El proyecto incluye el módulo `src/clases_po/kmeans.py`, que contiene la función:

`generar_reporte_kmeans()`

Esta función realiza el procesamiento y entrenamiento del modelo y genera un reporte en formato TXT.

El documento incluye información del dataset, configuración del experimento, métricas de evaluación y perfilamiento estadístico de los grupos.

**Archivo generado:**

`src/carga_datos/reportes/reporte_kmeans.txt`

Para garantizar la compatibilidad entre el entorno local y GitHub Actions, se utilizan rutas construidas a partir de la raíz del repositorio, evitando rutas absolutas específicas de un equipo.

## 7. Automatización con GitHub Actions

El proyecto incorpora un workflow de GitHub Actions para automatizar la ejecución del notebook y la actualización de los resultados.

El workflow se activa cuando se realiza un `push` sobre la rama configurada.

El proceso contempla:

1. Descargar el repositorio.
2. Configurar Python 3.12.
3. Crear el entorno virtual.
4. Instalar las dependencias.
5. Preparar y limpiar las carpetas de resultados.
6. Ejecutar `EA3_Modelo_No_Supervisor.ipynb`.
7. Generar las gráficas y el reporte TXT.
8. Confirmar y publicar los cambios generados, cuando existan y los permisos del repositorio lo permitan.

La ejecución se considera satisfactoria cuando todos los pasos del workflow finalizan correctamente.

## 8. Ejecución del proyecto

Desde la raíz del repositorio, instalar las dependencias:

```bash
python -m pip install -r requirements.txt
python -m pip install jupyter nbconvert nbclient ipykernel
```

Ingresar a la carpeta del notebook:

```bash
cd src/carga_datos
```

Ejecutar:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 EA3_Modelo_No_Supervisor.ipynb
```

Al finalizar correctamente, el proceso permite actualizar los resultados del notebook y generar los archivos de salida.

## 9. Conclusiones

El algoritmo K-Means permitió desarrollar una segmentación exploratoria de los asegurados en tres grupos utilizando sus características demográficas, biológicas, conductuales y financieras.

El análisis de los clústeres permite comparar diferencias relacionadas con la edad, el índice de masa corporal, el hábito de fumar y los gastos médicos.

El perfilamiento estadístico proporciona información para reconocer grupos con características similares y explorar estrategias de prevención y seguimiento.

Los resultados pueden apoyar el análisis de perfiles de gasto médico y aportar información preliminar para la toma de decisiones en una compañía aseguradora. Sin embargo, no deben interpretarse directamente como una clasificación actuarial validada del riesgo individual.

Como trabajo futuro, se propone evaluar modelos supervisados de regresión, como Random Forest o XGBoost, para estimar gastos médicos.

Para estos modelos deberá garantizarse una separación adecuada entre entrenamiento y prueba, evitando utilizar información de la variable objetivo durante la construcción de características predictoras.

## 10. Consideraciones finales

El proyecto integra procesamiento de datos, aprendizaje no supervisado, evaluación de modelos, visualización y automatización.

La utilización de GitHub Actions facilita la reproducibilidad técnica del proceso y la actualización de los resultados generados.

La selección de K=3 responde a una decisión de interpretación del modelo, mientras que los resultados exploratorios sugieren analizar también K=4 en futuras comparaciones.

El modelo representa una herramienta exploratoria de segmentación y no sustituye evaluaciones médicas ni actuariales especializadas.

---------------------------------------------------------------------------

Actividad 2  Modelos de predicción y clasificación

## Dataset
https://huggingface.co/datasets/shijiabraham/insurance-data



## 1. Descripción del proyecto

El objetivo de esta actividad es implementar un flujo de Machine Learning que permita cargar automáticamente un dataset desde Hugging Face, realizar el preprocesamiento de los datos, entrenar y evaluar modelos de regresión lineal y clasificación, y generar un resumen automático de los resultados.

El proyecto está desarrollado en Python, utilizando Pandas, scikit-learn y Matplotlib, con una estructura de clases reutilizables.

## 2. Dataset utilizado

- **Nombre:** Medical Insurance
- **Identificador:** `shijiabraham/insurance-data`
- **Fuente:** https://huggingface.co/datasets/shijiabraham/insurance-data
- **Split:** `train`
- **Registros originales:** 1.338
- **Columnas:** 7

El dataset contiene información de características personales y costos de seguros médicos.

| Columna | Descripción |
|---|---|
| `age` | Edad de la persona |
| `sex` | Sexo |
| `bmi` | Índice de masa corporal |
| `children` | Número de hijos |
| `smoker` | Indica si la persona fuma |
| `region` | Región de residencia |
| `charges` | Costos médicos registrados |

Los datos se obtienen automáticamente desde Hugging Face mediante la clase `HFDatasetLoader`, sin necesidad de descargarlos manualmente.

## 3. Preprocesamiento

Inicialmente se analizaron las dimensiones, los tipos de datos, los valores faltantes y los registros duplicados.

| Indicador | Antes | Después |
|---|---:|---:|
| Registros | 1.338 | 1.337 |
| Duplicados exactos | 1 | 0 |
| Valores faltantes | 0 | 0 |

Se eliminó un registro duplicado exacto y no fue necesaria la imputación de valores faltantes.

Para la regresión se seleccionaron variables numéricas. Para la clasificación se utilizaron las siguientes transformaciones:

- `StandardScaler` para edad, IMC y número de hijos.
- `OneHotEncoder` para sexo y región.
- `Pipeline` y `ColumnTransformer` para integrar el preprocesamiento con el clasificador.

Las transformaciones que requieren ajuste se entrenan únicamente con el conjunto de entrenamiento para evitar fugas de información hacia el conjunto de prueba.

Los datos se dividieron en un 80 % para entrenamiento y un 20 % para prueba, utilizando `random_state=42`.

## 4. Modelo de regresión lineal

**Pregunta:** ¿Es posible estimar los costos médicos de una persona a partir de sus características?

Se implementaron dos modelos utilizando `LinearRegression` de scikit-learn, con sus parámetros predeterminados.

### Regresión lineal simple

- Variable predictora: `bmi`
- Variable objetivo: `charges`
- Algoritmo: regresión lineal
- División: 80 % entrenamiento, 20 % prueba
- Semilla: 42

Resultados aproximados:

| Métrica | Resultado |
|---|---:|
| MAE | 9,891.12 |
| RMSE | 13,200.44 |
| R² | 0.05 |

El modelo presenta una capacidad predictiva limitada. El IMC, utilizado individualmente, explica una proporción reducida de la variación de los costos médicos.

### Regresión lineal múltiple

- Variables predictoras: `age`, `bmi`
- Variable objetivo: `charges`
- Algoritmo: regresión lineal
- División: 80 % entrenamiento, 20 % prueba
- Semilla: 42

Resultados aproximados:

| Métrica | Resultado |
|---|---:|
| MAE | 9,575.76 |
| RMSE | 12,559.72 |
| R² | 0.14 |

La regresión múltiple presenta un mejor rendimiento que la regresión simple, al reducir los errores y aumentar el coeficiente de determinación.

Los coeficientes estimados indican una asociación positiva de la edad y el IMC con los costos médicos, manteniendo constante la otra variable. El intercepto no tiene una interpretación práctica directa cuando ambas variables son cero.

Aunque agregar la edad mejora los resultados, ambos modelos presentan limitaciones debido a que los costos médicos dependen también de otras características.

## 5. Modelo de clasificación

**Pregunta:** ¿Es posible clasificar a una persona como fumadora o no fumadora a partir de sus características personales?

Se implementó un modelo de clasificación binaria mediante regresión logística.

- Variable objetivo: `smoker`
- Clases: `yes` (fumador) y `no` (no fumador)
- Variables predictoras: `age`, `bmi`, `children`, `sex`, `region`
- Algoritmo: `LogisticRegression`
- `max_iter=1000`
- `random_state=42`
- División: 80 % entrenamiento, 20 % prueba
- División estratificada según `smoker`

No se utilizó `charges` como variable predictora, con el fin de evaluar la clasificación a partir de características personales.

### Clasificación inicial

El modelo inicial obtuvo una exactitud del 79.48 %. Sin embargo, clasificó todas las observaciones como no fumadoras, por lo que su recall para fumadores fue de 0 %.

### Clasificación con balanceo de clases

Para mejorar la identificación de fumadores se probó el parámetro `class_weight="balanced"`.

| Métrica | Resultado |
|---|---:|
| Accuracy | 60.07 % |
| Precision (fumadores) | 26.79 % |
| Recall (fumadores) | 54.55 % |
| F1-score (fumadores) | 35.93 % |

Matriz de confusión:

| Clase real | Predice no fumador | Predice fumador |
|---|---:|---:|
| No fumador | 131 | 82 |
| Fumador | 25 | 30 |

El balanceo permitió identificar correctamente a 30 de los 55 fumadores del conjunto de prueba, aunque también generó 82 falsos positivos.

El modelo balanceado fue seleccionado como configuración final para priorizar la identificación de la clase fumadora, reconociendo la disminución de la exactitud y las limitaciones de sus predicciones.

## 6. Organización del proyecto

Archivos principales de la Actividad 2:

```text
Machine_Learning/
├── .github/
│   └── workflows/
├── src/
│   ├── clases_po/
│   │   ├── __init__.py
│   │   ├── hf_loader.py
│   │   ├── regresion_lineal.py
│   │   └── clasificacion.py
│   └── carga_datos/
│       ├── EA2_Modelo_Predicción_Clasificación.ipynb
│       ├── entrenamiento/
│       └── resumen_modelos_actividad2.txt
├── requirements.txt
└── README.md
```

La clase `RegresionLineal` permite preparar datos numéricos, entrenar, predecir, evaluar y guardar modelos de regresión.

La clase `Clasificacion` integra el preprocesamiento categórico y numérico, el entrenamiento, la evaluación y el guardado del clasificador.

Los archivos `.pkl` contienen modelos entrenados que pueden reutilizarse para realizar predicciones.

## 7. Instrucciones de ejecución local

1. Clonar el repositorio y acceder a su directorio.

2. Crear y activar un entorno virtual de Python.

3. Instalar las dependencias:

```bash
pip install -r requirements.txt
```

4. Abrir el notebook:

```text
src/carga_datos/EA2_Modelo_Predicción_Clasificación.ipynb
```

5. Seleccionar el entorno virtual como kernel de Jupyter.

6. Ejecutar las celdas de manera secuencial desde el inicio.

El notebook carga los datos, realiza el preprocesamiento, entrena y evalúa los modelos y genera el resumen.

## 8. Resumen automático de resultados

El archivo:

```text
src/carga_datos/resumen_modelos_actividad2.txt
```

se genera automáticamente mediante código Python.

Contiene la identificación del dataset, el preprocesamiento, la configuración de los modelos, las métricas, los coeficientes de regresión, la matriz de confusión, las interpretaciones, las limitaciones y la fecha de ejecución.

Los resultados provienen de las variables calculadas durante la ejecución.

## 9. Automatización con GitHub Actions

La automatización debe permitir ejecutar el flujo completo al publicar cambios en el repositorio.

El workflow instalará las dependencias, cargará el dataset desde Hugging Face, realizará el preprocesamiento, entrenará y evaluará los modelos y generará el resumen `.txt`.

El reporte deberá quedar disponible como artefacto descargable de la ejecución.

**Estado:** pendiente de configurar y verificar la ejecución automática completa.

## 10. Conclusiones

La actividad permitió implementar modelos de regresión y clasificación utilizando un mismo dataset.

En regresión, incorporar la edad junto con el IMC mejoró el rendimiento respecto al modelo simple. Sin embargo, la capacidad predictiva continuó siendo limitada.

En clasificación, se evidenció que una exactitud aparentemente alta puede ocultar un desempeño deficiente sobre la clase minoritaria. La utilización de pesos balanceados mejoró la identificación de fumadores, aunque aumentaron los falsos positivos.

El proyecto utiliza clases reutilizables y genera automáticamente un reporte de resultados, lo que facilita su integración posterior con GitHub Actions.

## 11. Integrantes

Agregar aquí los nombres de los integrantes del grupo.

## 12. Evidencias de ejecución

- **Repositorio:** https://github.com/SernaSol/Machine_Learning
- **Notebook:** `src/carga_datos/EA2_Modelo_Predicción_Clasificación.ipynb`
- **Resumen:** `src/carga_datos/resumen_modelos_actividad2.txt`
- **GitHub Actions:** agregar el enlace a la ejecución exitosa cuando esté disponible.




Actividad 1
Proyecto de aprendizaje automático y análisis de datos con Python. Incluye un notebook para descargar y explorar un dataset de insuficiencia cardíaca desde Hugging Face, transformar sus datos y generar reportes descriptivos.

## Dataset
https://huggingface.co/datasets/mstz/heart_failure


## Funcionalidades

- Descarga datasets de Hugging Face y los convierte en `pandas.DataFrame`.
- Permite elegir la configuración y división del dataset, limitar la cantidad de filas y guardar los datos en CSV, JSON, Parquet o Excel.
- Genera reportes con dimensiones del DataFrame, duplicados, tipos de datos, valores nulos y estadísticas descriptivas de las columnas numéricas.
- Incluye un flujo de análisis que traduce y transforma columnas, calcula años de seguimiento y resume indicadores clínicos por estado vital.

## Estructura del proyecto

```text
.
├── requirements.txt
├── src/
│   ├── clases_po/
│   │   ├── hf_loader.py            # Carga y exportación de datasets
│   │   └── reportes.py             # Generación de reportes
│   └── carga_datos/
│       ├── datos/                  # Archivos CSV procesados generados
│       ├── reportes/               # Reportes de texto generados
│       └── EA1_Carga_Datos.ipynb   # Notebook de carga y análisis
└── .github/
    └── workflows/
        └── ejecutar.yml            # Ejecución automática del notebook

## Uso

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

En Bash, el entorno virtual debe activarse con `source` (o con un punto):

```bash
source .venv/bin/activate
# equivalente:
. .venv/bin/activate
```
## Autenticación en Hugging Face

Para descargar datasets públicos normalmente no se necesita un token. Si el
dataset es privado o requiere aceptación de condiciones de uso, configura un
token de Hugging Face siguiendo estos pasos.

### 1. Crear un token

1. Inicia sesión en [Hugging Face](https://huggingface.co/).
2. Abre [Settings > Access Tokens](https://huggingface.co/settings/tokens).
3. Selecciona **Create new token**.
4. Asigna un nombre descriptivo, por ejemplo `RP_AD2026Q4`.
5. Selecciona el permiso mínimo necesario: **Read**.
6. Presiona **Create token** y copia el token. Normalmente comienza por
   `hf_`.

Guarda el token de forma segura y no lo incluyas en el código fuente.

### 2. Instalar la CLI e iniciar sesión

Con el entorno virtual activado, instala o actualiza la librería de Hugging
Face:

```bash
python -m pip install --upgrade huggingface_hub
```

Inicia sesión desde la terminal:

```bash
hf auth login
```

Cuando la terminal solicite el token, pégalo y presiona **Enter**. Es normal
que no se muestren caracteres mientras lo escribes o pegas.

La sesión queda guardada localmente para que `HFDatasetLoader` pueda usarla
automáticamente. Comprueba que la autenticación funcionó:

```bash
hf auth whoami
```

## Carga de un dataset como DataFrame

`HFDatasetLoader` recibe el identificador del dataset, descarga la división
indicada y devuelve un `pandas.DataFrame`. `filename` y `file_type` son
opcionales; si se proporciona `filename`, el DataFrame también se guarda en
`csv`, `json`, `parquet` o `xlsx`.

```python
from rp_utils import HFDatasetLoader

loader = HFDatasetLoader(
    "mstz/heart_failure",
    filename="datos/heart_failurecsv",
)
df = loader.load()
```