import pandas as pd
import numpy as np
from category_encoders import TargetEncoder, LeaveOneOutEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import f_oneway
import streamlit as st

# ===================  Preprocesamiento ===================
def detectar_columnas_categoricas(df):
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

def codificar_columnas_categoricas(df, columnas_cat, log):
    df_encoded = df.copy()
    encoder = TargetEncoder()

    for col in columnas_cat:
        otras_columnas = df_encoded.drop(columns=[col])
        numericas = otras_columnas.select_dtypes(include=[np.number])

        if not numericas.empty:
            f_scores = []
            for num_col in numericas.columns:
                grupos = [df_encoded[df_encoded[col] == val][num_col].dropna() for val in df_encoded[col].unique()]
                if len(grupos) > 1:
                    try:
                        _, p_valor = f_oneway(*grupos)
                        f_scores.append(p_valor)
                    except:
                        continue

            if f_scores and min(f_scores) < 0.05:
                df_encoded[col] = encoder.fit_transform(df_encoded[col], df_encoded[numericas.columns[0]])
                log.append(f"✔️ Codificación Target aplicada a '{col}' por su relación estadística con variables numéricas.")
            else:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                log.append(f"✔️ Codificación Label aplicada a '{col}' por falta de correlación.")
        else:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            log.append(f"✔️ Codificación Label aplicada a '{col}' por ausencia de numéricas.")

    return df_encoded, log

def ejecutar_preprocesamiento_completo(df_original):
    df = df_original.copy()
    log_conversiones = []
    columnas_removidas = []

    # 1. Detección del tipo de dataset
    columnas_categoricas = detectar_columnas_categoricas(df)
    es_dataset_numerico = len(columnas_categoricas) == 0

    # 2. Limpieza de columnas duplicadas y filas duplicadas
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.drop_duplicates()

    # 3. Eliminación de columnas con demasiados nulos
    umbral_nulos = 0.8
    cols_nulas = df.columns[df.isnull().mean() > umbral_nulos].tolist()
    if cols_nulas:
        df.drop(columns=cols_nulas, inplace=True)
        columnas_removidas.extend(cols_nulas)
        log_conversiones.append(f"✔️ Columnas eliminadas por alto porcentaje de nulos: {cols_nulas}")

    # 4. Tratamiento de nulos restantes
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].median(), inplace=True)
                log_conversiones.append(f"✔️ Nulos en '{col}' rellenados con la mediana.")
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
                log_conversiones.append(f"✔️ Nulos en '{col}' rellenados con la moda.")

    # 5. Outliers por IQR
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower, lower, np.where(df[col] > upper, upper, df[col]))
        log_conversiones.append(f"✔️ Outliers tratados en '{col}' con método IQR.")

    # 6. Eliminación de columnas de varianza baja
    selector = VarianceThreshold(threshold=1e-6)
    try:
        selector.fit(df.select_dtypes(include=[np.number]))
        columnas_filtradas = df.select_dtypes(include=[np.number]).columns[~selector.get_support()].tolist()
        if columnas_filtradas:
            df.drop(columns=columnas_filtradas, inplace=True)
            columnas_removidas.extend(columnas_filtradas)
            log_conversiones.append(f"✔️ Columnas eliminadas por baja varianza: {columnas_filtradas}")
    except Exception:
        pass

    # 7. Codificación de columnas categóricas (solo si hay)
    if not es_dataset_numerico and columnas_categoricas:
        df, log_conversiones = codificar_columnas_categoricas(df, columnas_categoricas, log_conversiones)

    # Resultado final
    columnas_finales = df.columns.tolist()
    log = {
        "estado": "éxito",
        "columnas_finales": columnas_finales,
        "columnas_removidas": columnas_removidas,
        "conversiones": log_conversiones
    }

    return df, log

