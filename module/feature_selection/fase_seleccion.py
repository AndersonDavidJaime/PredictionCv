import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import StandardScaler

class FaseSeleccionRobusta:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df_preparado = None
        self.metricas = {}

    def _preparar_datos(self):
        df = self.df.copy()
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(num_cols) > 0:
            df[num_cols] = StandardScaler().fit_transform(df[num_cols])
        self.df_preparado = df

    def determinar_variables_dependientes(self, cantidad=1):
        self._preparar_datos()
        df = self.df_preparado
        scores = {}

        for col in df.columns:
            X = df.drop(columns=[col])
            y = df[col]

            # Convertir a numpy para evitar errores de indexación
            X_np = X.to_numpy()
            y_np = y.to_numpy().ravel()

            # Determinar si es clasificación o regresión basado en la variable dependiente
            if y.nunique() <= 10 and pd.api.types.is_integer_dtype(y):
                es_clasificacion = True
            else:
                es_clasificacion = False

            # Información mutua
            if es_clasificacion:
                mi_score = mutual_info_classif(X_np, y_np, discrete_features='auto')
            else:
                mi_score = mutual_info_regression(X_np, y_np, discrete_features='auto')

            # Correlaciones
            try:
                pearson_corr = np.mean([abs(np.corrcoef(X_np[:, i], y_np)[0, 1]) for i in range(X_np.shape[1])])
            except:
                pearson_corr = 0
            try:
                spearman_corr = np.mean([abs(pd.Series(X_np[:, i]).corr(pd.Series(y_np), method='spearman')) for i in range(X_np.shape[1])])
            except:
                spearman_corr = 0

            # Score final combinando información mutua y correlaciones
            final_score = 0.4 * pearson_corr + 0.4 * spearman_corr + 0.2 * np.mean(mi_score)
            scores[col] = final_score
            self.metricas[col] = final_score

        # Orden descendente y selección de la cantidad solicitada
        top_vars = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:cantidad]
        columnas, valores = zip(*top_vars)
        return list(columnas), list(valores)

    def obtener_metricas(self):
        return self.metricas
