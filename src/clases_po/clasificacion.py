"""Flujo reutilizable para clasificación binaria con regresión logística."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class Clasificacion:
    """Entrena, evalúa y guarda un pipeline para clasificar fumadores."""

    def __init__(self, target_column="smoker", numeric_columns=None,
                 categorical_columns=None, test_size=0.2, random_state=42,
                 class_weight="balanced", max_iter=1000):
        self.target_column = target_column
        self.numeric_columns = list(numeric_columns if numeric_columns is not None else ["age", "bmi", "children"])
        self.categorical_columns = list(categorical_columns if categorical_columns is not None else ["sex", "region"])
        self.test_size = test_size
        self.random_state = random_state
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.modelo = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.predicciones = None
        self.metricas = None
        self.ruta_modelo = None

    def preparar_datos(self, df):
        columnas = self.numeric_columns + self.categorical_columns + [self.target_column]
        faltantes = set(columnas) - set(df.columns)
        if faltantes:
            raise KeyError(f"Faltan columnas: {sorted(faltantes)}")
        datos = df[columnas].copy()
        # La limpieza de duplicados exactos se documenta fuera de la clase.
        # Si hay nulos, se informa un error para tomar una decisión explícita.
        if datos.isna().any().any():
            raise ValueError("Hay valores nulos: revise el preprocesamiento antes de entrenar")
        if not datos[self.target_column].isin(["yes", "no"]).all():
            raise ValueError("La variable objetivo debe contener únicamente 'yes' y 'no'")
        X = datos[self.numeric_columns + self.categorical_columns]
        y = datos[self.target_column]
        return X, y

    def dividir_datos(self, X, y):
        return train_test_split(X, y, test_size=self.test_size,
                                random_state=self.random_state, stratify=y)

    def entrenar(self, X_train, y_train, directorio_entrenamiento="entrenamiento"):
        preprocesador = ColumnTransformer([
            ("numericas", StandardScaler(), self.numeric_columns),
            ("categoricas", OneHotEncoder(handle_unknown="ignore"), self.categorical_columns),
        ])
        self.modelo = Pipeline([
            ("preprocesamiento", preprocesador),
            ("clasificador", LogisticRegression(max_iter=self.max_iter,
                                                random_state=self.random_state,
                                                class_weight=self.class_weight)),
        ])
        self.modelo.fit(X_train, y_train)
        carpeta = Path(directorio_entrenamiento)
        carpeta.mkdir(parents=True, exist_ok=True)
        self.ruta_modelo = carpeta / f"{datetime.now():%Y%m%d_%H%M%S_%f}_clasificacion.pkl"
        with self.ruta_modelo.open("wb") as archivo:
            pickle.dump(self.modelo, archivo)
        return self.ruta_modelo

    def evaluar(self, X_test, y_test):
        if self.modelo is None:
            raise ValueError("Primero debe entrenar el modelo")
        self.predicciones = self.modelo.predict(X_test)
        self.metricas = {
            "accuracy": float(accuracy_score(y_test, self.predicciones)),
            "precision": float(precision_score(y_test, self.predicciones, pos_label="yes", zero_division=0)),
            "recall": float(recall_score(y_test, self.predicciones, pos_label="yes", zero_division=0)),
            "f1": float(f1_score(y_test, self.predicciones, pos_label="yes", zero_division=0)),
        }
        matriz = confusion_matrix(y_test, self.predicciones, labels=["no", "yes"])
        reporte = classification_report(y_test, self.predicciones, zero_division=0)
        return self.metricas, matriz, reporte

    def ejecutar(self, df, directorio_entrenamiento="entrenamiento"):
        X, y = self.preparar_datos(df)
        self.X_train, self.X_test, self.y_train, self.y_test = self.dividir_datos(X, y)
        ruta = self.entrenar(self.X_train, self.y_train, directorio_entrenamiento)
        metricas, matriz, reporte = self.evaluar(self.X_test, self.y_test)
        print(f"[clasificacion] Entrenamiento: {len(self.X_train)}; prueba: {len(self.X_test)}")
        print(f"[clasificacion] Modelo guardado en: {ruta}")
        print(f"[clasificacion] Métricas: {metricas}")
        print("[clasificacion] Matriz (no, yes):\n", matriz)
        print(reporte)
        return {"modelo": self.modelo, "ruta_modelo": ruta,
                "X_train": self.X_train, "X_test": self.X_test,
                "y_train": self.y_train, "y_test": self.y_test,
                "predicciones": self.predicciones,
                "metricas": metricas, "matriz_confusion": matriz,
                "reporte": reporte}
