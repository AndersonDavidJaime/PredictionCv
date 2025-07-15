import pandas as pd
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

class FaseSeleccion:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df_preparado = None
        self.metricas = {}

    def _preparar_datos(self):
        """Prepara los datos normalizando y codificando variables"""
        df = self.df.copy()
        
        # Normalización numérica (evitando columna objetivo)
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

            # Verificar que y sea numérica y no contenga nulos
            if not pd.api.types.is_numeric_dtype(y):
                raise ValueError(f"La variable dependiente '{col}' no es numérica.")
            if y.isnull().any():
                raise ValueError(f"La variable dependiente '{col}' contiene valores nulos.")

            # Determinar si es un problema de clasificación o regresión
            if y.nunique() <= 10 and pd.api.types.is_integer_dtype(y):
                # Si hay 10 o menos clases únicas y es de tipo entero, es clasificación
                es_clasificacion = True
            elif y.nunique() <= 10 and pd.api.types.is_object_dtype(y):
                # Si hay 10 o menos clases únicas y es de tipo objeto, es clasificación
                es_clasificacion = True
            else:
                # En caso contrario, es regresión
                es_clasificacion = False
            
            if es_clasificacion:
                score = mutual_info_classif(X, y, discrete_features='auto')
            else:
                score = mutual_info_regression(X, y, discrete_features='auto')
            
            score_total = score.mean()  # Promedio de scores
            scores[col] = score_total
            self.metricas[col] = score_total  # Guarda el score en el atributo metricas

        # Seleccionar las N variables más importantes
        if cantidad is None:
            cantidad = 1

        top_vars = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:cantidad]
        columnas, valores = zip(*top_vars)
        return list(columnas), list(valores)

    def obtener_metricas(self):
        """Devuelve el diccionario con todas las métricas calculadas"""
        return self.metricas
