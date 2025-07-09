import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

class FaseSeleccion:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df_preparado = None
        self.metricas = {}  # Inicializa metricas como un diccionario vacío

    def _preparar_datos(self):
        """Prepara los datos normalizando y codificando variables"""
        df = self.df.copy()
        
        # 1. Codificación de categóricas
        categoricas = df.select_dtypes(include=['object', 'category']).columns
        for col in categoricas:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        
        # 2. Normalización numérica (evitando columna objetivo)
        if hasattr(self, 'target_column'):
            num_cols = [col for col in df.columns 
                        if col != self.target_column and 
                        df[col].dtype in ['int64', 'float64']]
        else:
            num_cols = df.select_dtypes(include=['int64', 'float64']).columns
            
        if len(num_cols) > 0:
            df[num_cols] = StandardScaler().fit_transform(df[num_cols])
            
        self.df_preparado = df  # Asegúrate de que esto esté presente

    def determinar_variables_dependientes(self, cantidad=None):
        """Determina las variables dependientes más importantes."""
        self._preparar_datos()
        
        df = self.df_preparado
        scores = {}
        
        for col in df.columns:
            X = df.drop(columns=[col])
            y = df[col]
            
            # Determinar si es un problema de clasificación o regresión
            es_clasificacion = (y.nunique() <= 10) and (y.dtype in ['int64', 'object'])  # Heurística para clasificación
            
            if es_clasificacion:
                score = mutual_info_classif(X, y, discrete_features='auto')
            else:
                score = mutual_info_regression(X, y, discrete_features='auto')
            
            score_total = score.mean()  # Promedio de scores
            scores[col] = score_total
            self.metricas[col] = score_total  # Almacena el score en metricas

        # Seleccionar las N variables más importantes
        if cantidad is None:
            cantidad = 1

        top_vars = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:cantidad]
        columnas, valores = zip(*top_vars)
        return list(columnas), list(valores)

    def obtener_metricas(self):
        """Devuelve todas las métricas calculadas"""
        if not self.metricas:
            raise ValueError("Primero debes ejecutar determinar_variables_dependientes()")
        return self.metricas
