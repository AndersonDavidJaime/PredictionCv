import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from multiprocessing import Pool
import streamlit as st

class AlgoritmoEvolutivo:
    def __init__(self, data, target_vars, n_poblacion=5, prob_mut=0.03):
        self.data = data
        self.target = target_vars if isinstance(target_vars, list) else [target_vars]
        self.n_poblacion = max(2, n_poblacion)
        self.prob_mut = prob_mut
        self.independientes = [col for col in data.columns if col not in self.target]
        self.n_vars = len(self.independientes)
        self.historial_fitness = []  # Inicializa el historial de fitness

    def _generar_individuo(self):
        return np.random.randint(0, 2, self.n_vars)
    
    def _evaluar_individuo(self, individuo):
        cols_selec = [self.independientes[i] for i, val in enumerate(individuo) if val == 1]
        
        if not cols_selec:
            return 0.0
        
        X = self.data[cols_selec]
        y = self.data[self.target].values.ravel()  # Asegúrate de que y sea un array 1D
        
        model = RandomForestRegressor(n_estimators=5, max_depth=3, n_jobs=-1)
        try:
            mse = -cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=2).mean()
            return 1 / (1 + mse)
        except Exception as e:
            print(f"Error en la evaluación del individuo: {e}")  # Para depuración
            return 0.0

    def _evaluar_poblacion(self, poblacion):
        with Pool(2) as pool:
            return np.array(pool.map(self._evaluar_individuo, poblacion))
    
    def _cruzar(self, padre1, padre2):
        punto = np.random.randint(1, self.n_vars-1)
        hijo = np.concatenate([padre1[:punto], padre2[punto:]])
        return hijo
    
    def _mutar(self, individuo):
        return np.where(np.random.random(self.n_vars) < self.prob_mut, 1 - individuo, individuo)
    
    def ejecutar(self, n_generaciones=3):
        poblacion = [self._generar_individuo() for _ in range(self.n_poblacion)]
        mejor_fitness = -np.inf
        mejor_individuo = None
        
        for _ in range(n_generaciones):
            fitness = self._evaluar_poblacion(poblacion)
            self.historial_fitness.append(fitness.max())  # Guarda el mejor fitness de esta generación
            
            idx_mejor = np.argmax(fitness)
            if fitness[idx_mejor] > mejor_fitness:
                mejor_fitness = fitness[idx_mejor]
                mejor_individuo = poblacion[idx_mejor]
        
        vars_seleccionadas = []
        if mejor_individuo is not None:
            vars_seleccionadas = [self.independientes[i] for i, val in enumerate(mejor_individuo) if val == 1]
        
        # Asegúrate de que siempre se devuelva un diccionario válido
        return {
            'variables': vars_seleccionadas,
            'fitness': mejor_fitness if mejor_fitness != -np.inf else 0.0,
            'total_vars': len(vars_seleccionadas),
            'historial_fitness': self.historial_fitness  # Devuelve el historial de fitness
        }
