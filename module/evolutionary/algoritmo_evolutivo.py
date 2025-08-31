import numpy as np
import pandas as pd
from deap import base, creator, tools
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

class AlgoritmoEvolutivo:
    def __init__(self, data, target_vars, n_poblacion=5, prob_mut=0.03, prob_cruce=0.5, n_generaciones=10, lambda_penal=0.01, min_vars=1):
        self.data = data
        self.target = target_vars if isinstance(target_vars, list) else [target_vars]
        self.n_poblacion = max(2, n_poblacion)
        self.prob_mut = prob_mut
        self.prob_cruce = prob_cruce
        self.n_generaciones = n_generaciones
        self.lambda_penal = lambda_penal  
        self.independientes = [col for col in data.columns if col not in self.target]
        self.min_vars = min_vars
        self.n_vars = len(self.independientes)

        # Crear tipos de fitness y individuo para MINIMIZACIÓN
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimizacion
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

        self.historial_fitness = []  # Guardar la evolución del fitness

    def _init_individual(self, icls):
        #Genera un individuo que respete el mínimo de variables requeridas
        ind = [0] * self.n_vars
        seleccionadas = np.random.choice(range(self.n_vars), 
                                         size=np.random.randint(self.min_vars, self.n_vars+1), 
                                         replace=False)
        for idx in seleccionadas:
            ind[idx] = 1
        return icls(ind)

    def _evaluar_individuo(self, individuo):
        cols_selec = [self.independientes[i] for i, val in enumerate(individuo) if val == 1]

        # Penalización si no cumple mínimo
        if len(cols_selec) < self.min_vars:
            return float('inf'),  
    
        X = self.data[cols_selec]
        y = self.data[self.target].values.ravel()

        model = RandomForestRegressor(n_estimators=20, max_depth=5, n_jobs=-1, random_state=42)
        try:
            # Calcular el MSE con validación cruzada
            mse = -cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=3).mean()

            # Penalización por número de variables
            penalizacion = self.lambda_penal * len(cols_selec)

            # Fitness = MSE + penalización que busca MINIMIZAR
            fitness = mse + penalizacion
            return fitness,

        except Exception as e:
            print(f"Error en la evaluación del individuo: {e}")
            return float('inf'),

    def ejecutar(self, tol=1e-8, max_mutaciones=50):
        # Crear población inicial
        poblacion = self.toolbox.population(n=self.n_poblacion)

        # Evaluar la población inicial
        fitnesses = list(map(self.toolbox.evaluate, poblacion))
        for ind, fit in zip(poblacion, fitnesses):
            ind.fitness.values = fit

        mejor_hasta_ahora = min(fitnesses)[0]

        # Evolución
        for gen in range(self.n_generaciones):
            offspring = self.toolbox.select(poblacion, len(poblacion))
            offspring = list(map(self.toolbox.clone, offspring))

            # Cruce
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.rand() < self.prob_cruce:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Mutación
            for mutant in offspring:
                if np.random.rand() < self.prob_mut:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluar individuos inválidos
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid_ind:
                ind.fitness.values = self.toolbox.evaluate(ind)

            # Garantizar que el fitness sea decreciente
            fits = [ind.fitness.values[0] for ind in offspring]
            mejor_idx = np.argmin(fits)
            mejor_gen = fits[mejor_idx]

            intentos = 0
            while mejor_gen >= mejor_hasta_ahora - tol and intentos < max_mutaciones:
                # Aplicar mutación adicional al mejor individuo
                self.toolbox.mutate(offspring[mejor_idx])
                offspring[mejor_idx].fitness.values = self.toolbox.evaluate(offspring[mejor_idx])
                mejor_gen = offspring[mejor_idx].fitness.values[0]
                intentos += 1

            # Actualizar mejor fitness
            if mejor_gen < mejor_hasta_ahora - tol:
                mejor_hasta_ahora = mejor_gen

            self.historial_fitness.append(mejor_hasta_ahora)
            poblacion[:] = offspring

        # Mejor individuo final
        mejor_idx = np.argmin([ind.fitness.values[0] for ind in poblacion])
        mejor_individuo = poblacion[mejor_idx]
        vars_seleccionadas = [self.independientes[i] for i, val in enumerate(mejor_individuo) if val == 1]
        mejor_fitness = mejor_individuo.fitness.values[0]

        return {
            'variables': vars_seleccionadas,
            'fitness': mejor_fitness,
            'total_vars': len(vars_seleccionadas),
            'historial_fitness': self.historial_fitness
        }

