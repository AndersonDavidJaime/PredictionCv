import numpy as np
import pandas as pd
from deap import base, creator, tools
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
import streamlit as st


class AlgoritmoEvolutivo:
    def __init__(self, data, target_vars, modelo_seleccionado=None,
                 n_poblacion=5, prob_mut=0.03, prob_cruce=0.5,
                 n_generaciones=10, lambda_penal=0.03, min_vars=1, cv=3,
                 enforce_strict=False, max_attempts_strict=30):
        self.data = data.copy()
        self.target = target_vars if isinstance(target_vars, list) else [target_vars]

        if len(self.target) != 1:
            raise ValueError("AlgoritmoEvolutivo soporta una única variable dependiente por ejecución.")
        self.target = self.target[0]

        self.n_poblacion = max(2, n_poblacion)
        self.prob_mut = prob_mut
        self.prob_cruce = prob_cruce
        self.n_generaciones = n_generaciones
        self.lambda_penal = lambda_penal
        self.min_vars = min_vars
        self.cv = cv

        self.independientes = [c for c in self.data.columns if c != self.target]
        self.n_vars = len(self.independientes)

        self.enforce_strict = bool(enforce_strict)
        self.max_attempts_strict = int(max_attempts_strict)

        self.task_type = self._detect_task_type()

        # Modelos iniciales (serán reinicializados en ejecutar)
        self.modelos_reg = {}
        self.modelos_clf = {}
        self.modelos = {}

        # Modelo activo
        self.modelo_activo = {}
        if modelo_seleccionado:
            self.modelo_seleccionado = modelo_seleccionado
        else:
            self.modelo_seleccionado = None

        self.historial_por_modelo = {}

        # DEAP setup
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_bool", np.random.randint, 0, 2)
        self.toolbox.register("individual", self._init_individual, creator.Individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._evaluar_individuo)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=self.prob_mut)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

        self.eval_cache = {}

    def _detect_task_type(self):
        try:
            preprocess_log = st.session_state.get("preprocessing_log") \
                or st.session_state.get("preprocessingLog")
            if isinstance(preprocess_log, dict):
                cat_cols = preprocess_log.get("deteccion", {}).get("columnas_categoricas", [])
                if self.target in cat_cols:
                    return "classification"
        except Exception:
            pass

        series = self.data[self.target]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
            return "classification"
        if pd.api.types.is_integer_dtype(series) and series.nunique() <= 10:
            return "classification"
        return "regression"

    def _init_individual(self, icls):
        ind = [0] * self.n_vars
        k = np.random.randint(self.min_vars, self.n_vars + 1)
        seleccionadas = np.random.choice(range(self.n_vars), size=k, replace=False)
        for idx in seleccionadas:
            ind[idx] = 1
        return icls(ind)

    def _evaluar_individuo(self, individuo, mejor_hasta_ahora=float("inf")):
        cols_seleccionadas = [self.independientes[i] for i, v in enumerate(individuo) if v == 1]
        if len(cols_seleccionadas) < self.min_vars:
            return (10.0 + self.lambda_penal * (self.min_vars - len(cols_seleccionadas)),)

        key = tuple(sorted(cols_seleccionadas))
        if key in self.eval_cache:
            best_fitness, resultados_modelos, best_model = self.eval_cache[key]
            individuo.resultados_modelos = resultados_modelos
            individuo.mejor_modelo = best_model
            return best_fitness,

        X = self.data[cols_seleccionadas]
        y = self.data[self.target].values.ravel()

        resultados_modelos = {}
        mejor_modelo = None
        mejor_fitness = float("inf")

        for nombre, modelo in self.modelo_activo.items():
            try:
                if self.task_type == "regression":
                    cv_strategy = KFold(n_splits=min(self.cv, len(y)), shuffle=True, random_state=np.random.randint(0, 10000))
                    MSE = cross_val_score(modelo, X, y, scoring="neg_mean_squared_error", cv=cv_strategy, n_jobs=-1)
                    fitness = -MSE.mean() + self.lambda_penal * len(cols_seleccionadas)
                else:
                    cv_strategy = StratifiedKFold(n_splits=min(self.cv, len(y)), shuffle=True, random_state=np.random.randint(0, 10000))
                    score = cross_val_score(modelo, X, y, scoring="neg_log_loss", cv=cv_strategy)
                    fitness = -score.mean() + self.lambda_penal * len(cols_seleccionadas)
            except Exception:
                fitness = 10.0 + self.lambda_penal * len(cols_seleccionadas)

            resultados_modelos[nombre] = fitness
            if fitness < mejor_fitness:
                mejor_fitness = fitness
                mejor_modelo = nombre

        self.eval_cache[key] = (mejor_fitness, resultados_modelos, mejor_modelo)
        individuo.resultados_modelos = resultados_modelos
        individuo.mejor_modelo = mejor_modelo
        return mejor_fitness,

    def _init_modelos(self):
        """Reinicia los modelos con seeds aleatorios en cada ejecución"""
        if self.task_type == "regression":
            self.modelos = {
                "RandomForest": RandomForestRegressor(
                    n_estimators=50, max_depth=5, n_jobs=-1, random_state=np.random.randint(0, 10000)),
                "GradientBoosting": GradientBoostingRegressor(random_state=np.random.randint(0, 10000)),
                "LinearRegression": LinearRegression(),
                "Ridge": Ridge(),
                "SVR": SVR(),
                "KNN": KNeighborsRegressor()
            }
        else:
            self.modelos = {
                "RandomForest": RandomForestClassifier(
                    n_estimators=50, n_jobs=-1, random_state=np.random.randint(0, 10000)),
                "GradientBoosting": GradientBoostingClassifier(random_state=np.random.randint(0, 10000)),
                "LogisticRegression": LogisticRegression(max_iter=1000, random_state=np.random.randint(0, 10000)),
                "KNN": KNeighborsClassifier()
            }

        if self.modelo_seleccionado and self.modelo_seleccionado in self.modelos:
            self.modelo_activo = {self.modelo_seleccionado: self.modelos[self.modelo_seleccionado]}
        else:
            nombre_default = list(self.modelos.keys())[0]
            self.modelo_activo = {nombre_default: self.modelos[nombre_default]}

        self.historial_por_modelo = {name: [] for name in self.modelo_activo.keys()}

    def ejecutar(self, random_seed=None):
        if random_seed is None:
            np.random.seed(None)
        else:
            np.random.seed(random_seed)

        self.eval_cache = {}
        self._init_modelos()

        poblacion = self.toolbox.population(n=self.n_poblacion)
        for ind in poblacion:
            ind.fitness.values = self.toolbox.evaluate(ind)

        #  Guardar mejor individuo global
        mejor_global = min(poblacion, key=lambda ind: ind.fitness.values[0])
        mejor_fitness_global = mejor_global.fitness.values[0]

        sin_mejora = 0  # Contador de generaciones sin mejora

        # Inicializar historial
        for nombre in self.modelo_activo.keys():
            vals = [ind.resultados_modelos.get(nombre, float("inf")) for ind in poblacion]
            self.historial_por_modelo[nombre].append(min(vals) if vals else float("inf"))

        for gen in range(self.n_generaciones):
            #  Selección por torneo
            offspring = self.toolbox.select(poblacion, len(poblacion))
            offspring = list(map(self.toolbox.clone, offspring))

            #  Elitismo: conservar el mejor individuo global
            offspring[0] = self.toolbox.clone(mejor_global)

            #  Cruce
            for c1, c2 in zip(offspring[1::2], offspring[2::2]):
                if np.random.rand() < self.prob_cruce:
                    self.toolbox.mate(c1, c2)
                    del c1.fitness.values
                    del c2.fitness.values

            # Mutación adaptativa
            mut_prob = self.prob_mut
            if sin_mejora >= 5:  # aumenta mutación si no mejora en 5 generaciones
                mut_prob = min(0.5, self.prob_mut * 2)

            for ind in offspring[1:]:
                if np.random.rand() < mut_prob:
                    self.toolbox.mutate(ind)
                    del ind.fitness.values

            #  Evaluación y reintento si no mejora
            for ind in offspring:
                if not ind.fitness.valid:
                    ind.fitness.values = self.toolbox.evaluate(ind)

                # Intentos extra para escapar de estancamiento
                attempts = 0
                while ind.fitness.values[0] >= mejor_fitness_global and attempts < 5:
                    self.toolbox.mutate(ind)
                    ind.fitness.values = self.toolbox.evaluate(ind)
                    attempts += 1

            #  Reintroducción de diversidad si no mejora en 10 generaciones
            if sin_mejora >= 10:
                n_reemplazos = max(1, self.n_poblacion // 4)
                nuevos = self.toolbox.population(n=n_reemplazos)
                for ind in nuevos:
                    ind.fitness.values = self.toolbox.evaluate(ind)
                offspring[-n_reemplazos:] = nuevos
                self.eval_cache.clear()  # limpiar cache para explorar nuevas soluciones

            # Actualizar población
            poblacion = offspring

            # Actualizar mejor global
            mejor_actual = min(poblacion, key=lambda ind: ind.fitness.values[0])
            if mejor_actual.fitness.values[0] < mejor_fitness_global:
                mejor_global = self.toolbox.clone(mejor_actual)
                mejor_fitness_global = mejor_global.fitness.values[0]
                sin_mejora = 0
            else:
                sin_mejora += 1

            # Actualizar historial
            for nombre in self.modelo_activo.keys():
                vals = [ind.resultados_modelos.get(nombre, float("inf")) for ind in poblacion]
                self.historial_por_modelo[nombre].append(min(vals) if vals else float("inf"))

        #  Extraer resultados finales
        vars_seleccionadas = [
            self.independientes[i] for i, val in enumerate(mejor_global) if val == 1
        ]

        return {
            "variables": vars_seleccionadas,
            "fitness": mejor_fitness_global,
            "modelo": mejor_global.mejor_modelo,
            "resultados_modelos": mejor_global.resultados_modelos,
            "total_vars": len(vars_seleccionadas),
            "historial_por_modelo": self.historial_por_modelo,
            "task_type": self.task_type
        }
