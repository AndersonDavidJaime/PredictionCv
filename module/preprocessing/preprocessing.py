import pandas as pd
import numpy as np
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import f_oneway

class Preprocessing:
    @staticmethod
    def detectar_columnas_categoricas(df):
        return df.select_dtypes(include=['object', 'category']).columns.tolist()

    @staticmethod
    def codificar_columnas_categoricas(df, columnas_cat, log):
        df_encoded = df.copy()
        encoder = TargetEncoder()
        log["codificacion"] = []

        for col in columnas_cat:
            otras_columnas = df_encoded.drop(columns=[col])
            numericas = otras_columnas.select_dtypes(include=[np.number])

            if not numericas.empty:
                f_scores = []
                for num_col in numericas.columns:
                    grupos = [df_encoded[df_encoded[col] == val][num_col].dropna()
                              for val in df_encoded[col].unique()]
                    if len(grupos) > 1:
                        try:
                            _, p_valor = f_oneway(*grupos)
                            f_scores.append(p_valor)
                        except:
                            continue

                if f_scores and min(f_scores) < 0.05:
                    df_encoded[col] = encoder.fit_transform(df_encoded[col], df_encoded[numericas.columns[0]])
                    log["codificacion"].append(
                        f"'{col}' codificada con TargetEncoder por relación estadística con variables numéricas."
                    )
                else:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    log["codificacion"].append(
                        f"'{col}' codificada con LabelEncoder por falta de correlación significativa."
                    )
            else:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                log["codificacion"].append(
                    f"'{col}' codificada con LabelEncoder por ausencia de variables numéricas."
                )

        return df_encoded, log

    @classmethod
    def ejecutar_preprocesamiento_completo(cls, df_original):
        df = df_original.copy()
        log = {}

        # Fase 1: Detección de columnas categóricas
        columnas_categoricas = cls.detectar_columnas_categoricas(df)
        es_dataset_numerico = len(columnas_categoricas) == 0
        log["deteccion"] = {
            "columnas_categoricas": columnas_categoricas
        }

        # Fase 2: Eliminación de duplicados
        df = df.loc[:, ~df.columns.duplicated()]
        df = df.drop_duplicates()
        log["duplicados"] = "Duplicados eliminados: filas y columnas duplicadas."

        # Fase 3: Columnas con alto porcentaje de nulos
        umbral_nulos = 0.8
        cols_nulas = df.columns[df.isnull().mean() > umbral_nulos].tolist()
        if cols_nulas:
            df.drop(columns=cols_nulas, inplace=True)
        log["eliminacion_nulos"] = {
            "columnas_eliminadas": cols_nulas
        }

        # Fase 4: Relleno de nulos restantes
        log["relleno_nulos"] = {}
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['int64', 'float64']:
                    df[col].fillna(df[col].median(), inplace=True)
                    log["relleno_nulos"][col] = "rellenado con mediana"
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
                    log["relleno_nulos"][col] = "rellenado con moda"

        # Fase 5: Tratamiento de outliers
        log["outliers"] = []
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df[col] = np.where(df[col] < lower, lower, np.where(df[col] > upper, upper, df[col]))
            log["outliers"].append(col)

        # Fase 6: Eliminación de baja varianza
        selector = VarianceThreshold(threshold=1e-6)
        try:
            selector.fit(df.select_dtypes(include=[np.number]))
            columnas_baja_var = df.select_dtypes(include=[np.number]).columns[~selector.get_support()].tolist()
            if columnas_baja_var:
                df.drop(columns=columnas_baja_var, inplace=True)
            log["baja_varianza"] = columnas_baja_var
        except Exception:
            log["baja_varianza"] = []

        # Fase 7: Codificación de columnas categóricas
        if not es_dataset_numerico and columnas_categoricas:
            df, log = cls.codificar_columnas_categoricas(df, columnas_categoricas, log)

        # Fase 8: Transformación datetime
        log["transformacion_datetime"] = []
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    log["transformacion_datetime"].append(col)
                except:
                    continue

        # Columnas finales totales
        log["columnas_finales"] = df.columns.tolist()

        return df, log
