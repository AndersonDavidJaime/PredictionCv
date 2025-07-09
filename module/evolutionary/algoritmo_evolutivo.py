from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd

class AlgoritmoEvolutivo:
    def __init__(self, data, target, n_poblacion=30, prob_mut=0.05):
        self.data = data
        self.target = target
        self.n_poblacion = n_poblacion
        self.prob_mut = prob_mut
        self.variables = [col for col in data.columns if col != target]
        self.n_vars = len(self.variables)
        self.historial_fitness = []
        self.historial_seleccion = []
        self.historial_cruce = []
        self.historial_mutacion = []

    def _inicializar_poblacion(self):
        return np.random.randint(0, 2, (self.n_poblacion, self.n_vars))

    def _evaluar_individuo(self, individuo):
        vars_seleccionadas = [self.variables[i] for i, val in enumerate(individuo) if val == 1]
        if not vars_seleccionadas:
            return 0.0, 0.0  # Penalizar individuos vacíos
        X = self.data[vars_seleccionadas]
        y = self.data[self.target]
        modelo = RandomForestRegressor()
        mse = -cross_val_score(modelo, X, y, scoring='neg_mean_squared_error').mean()
        r2 = cross_val_score(modelo, X, y, scoring='r2').mean()
        return 1 / (1 + mse), r2  # Fitness inverso al MSE

    def _seleccion_ruleta(self, poblacion, fitness):
        prob = fitness / fitness.sum()
        indices = np.random.choice(range(len(poblacion)), size=len(poblacion), p=prob)
        return poblacion[indices]

    def _cruce(self, padre1, padre2):
        punto_cruce = np.random.randint(1, self.n_vars - 1)
        hijo1 = np.concatenate((padre1[:punto_cruce], padre2[punto_cruce:]))
        hijo2 = np.concatenate((padre2[:punto_cruce], padre1[punto_cruce:]))
        return hijo1, hijo2

    def _mutacion(self, individuo):
        for i in range(self.n_vars):
            if np.random.rand() < self.prob_mut:
                individuo[i] = 1 - individuo[i]  # Flip bit
        return individuo

    def ejecutar(self, n_generaciones):
        poblacion = self._inicializar_poblacion()
        for gen in range(n_generaciones):
            fitness = np.array([self._evaluar_individuo(ind)[0] for ind in poblacion])
            r2_scores = np.array([self._evaluar_individuo(ind)[1] for ind in poblacion])
            self.historial_fitness.append(fitness.mean())
            self.historial_seleccion.append(fitness)
            
            # Selección
            poblacion_seleccionada = self._seleccion_ruleta(poblacion, fitness)
            self.historial_cruce.append(poblacion_seleccionada)

            # Cruce y mutación
            nueva_poblacion = []
            for i in range(0, self.n_poblacion, 2):
                padre1, padre2 = poblacion_seleccionada[i], poblacion_seleccionada[i + 1]
                hijo1, hijo2 = self._cruce(padre1, padre2)
                nueva_poblacion.append(self._mutacion(hijo1))
                nueva_poblacion.append(self._mutacion(hijo2))
            poblacion = np.array(nueva_poblacion)

        # Evaluar la mejor solución
        mejor_indice = np.argmax(fitness)
        mejor_fitness = fitness[mejor_indice]
        mejor_r2 = r2_scores[mejor_indice]
        return poblacion[mejor_indice], mejor_fitness, mejor_r2, self.historial_fitness
