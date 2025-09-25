from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import sys
import os
import time  
import plotly.express as px
# Agregar el directorio raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module.preprocessing.preprocessing import Preprocessing
from module.feature_selection.fase_seleccion import FaseSeleccionRobusta
from module.evolutionary.algoritmo_evolutivo import AlgoritmoEvolutivo
import streamlit as st
import numpy as np



# Configuración de página
st.set_page_config(page_title="PredictionCV", layout="wide")
st.title("📊 Predicción CV - Carga y preprocesamiento de datos")
tabs = st.tabs(["            📁 Cargar datos           ", "            ⚙️ Preprocesamiento            ", "            📉 Análisis            ", "            🧠 Algoritmo Evolutivo            ", "          📊 Resultados             "])

# Inicialización de variables de sesión
default_session_state = {
    "df_original": None,
    "df_preprocessed": None,
    "preprocessing_report": None,
    "tipo_conversion_log": [],
    "preprocessing_log": []  
}


for key, default_value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# ===================  Cargar Datos ===================
with tabs[0]:
    st.header("📁 Cargar datos")
    uploaded_file = st.file_uploader("Carga tu archivo (.csv o .xlsx)", type=["csv", "xlsx"])

    # Cargar datos
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
            st.session_state.df_original = df
            st.session_state.tipo_conversion_log = []
            st.success("✅ Archivo cargado correctamente")
        except Exception as e:
            st.error(f"❌ Error al cargar el archivo: {e}")

    # Usar siempre el DataFrame actualizado de la sesión
    if "df_original" in st.session_state and st.session_state.df_original is not None:
        df = st.session_state.df_original

        st.subheader("👁️ Vista previa del dataset")
        st.dataframe(df.head(50), use_container_width=True)

        # Resumen
        resumen_actualizado = {
            "Filas": df.shape[0],
            "Columnas": df.shape[1],
            "Numéricas": df.select_dtypes(include=['int64', 'float64']).shape[1],
            "Categóricas": df.select_dtypes(include=['object', 'category']).shape[1],
            "Fechas": df.select_dtypes(include=['datetime64[ns]']).shape[1]
        }

        st.subheader("📌 Resumen del conjunto de datos")
        st.dataframe(pd.DataFrame(resumen_actualizado.items(), columns=["Descripción", "Valor"]), use_container_width=True)

        # Tipos de datos
        st.subheader("🧬 Tipos de datos por columna")
        st.dataframe(pd.DataFrame({
            "Columna": df.columns,
            "Tipo detectado": df.dtypes.astype(str).values
        }), use_container_width=True)

        # Conversión manual
        with st.expander("✏️ Cambiar tipos de datos manualmente"):
            st.markdown("Selecciona el nuevo tipo de dato para cada columna:")
            tipo_modificado = {}

            for col in df.columns:
                tipo_actual = str(df[col].dtype)
                nuevo_tipo = st.selectbox(
                    f"Tipo actual de '{col}' → {tipo_actual}",
                    ['No cambiar', 'int64', 'float64', 'object', 'category', 'bool', 'datetime64[ns]'],
                    key=f"tipo_{col}"
                )
                tipo_modificado[col] = nuevo_tipo

            aplicar_cambios = st.button("Aplicar cambios de tipo")

            if aplicar_cambios:
                conversion_log = []
                nuevo_df = df.copy()

                for col, nuevo_tipo in tipo_modificado.items():
                    if nuevo_tipo != 'No cambiar':
                        try:
                            if nuevo_tipo == 'datetime64[ns]':
                                nuevo_df[col] = pd.to_datetime(nuevo_df[col], errors='coerce')
                            else:
                                nuevo_df[col] = nuevo_df[col].astype(nuevo_tipo)
                            conversion_log.append(f"✔️ '{col}' convertido a `{nuevo_tipo}` correctamente.")
                        except Exception as e:
                            conversion_log.append(f"⚠️ Error al convertir '{col}' a `{nuevo_tipo}`: {e}")

                # Guardar cambios y forzar recarga visual
                st.session_state.df_original = nuevo_df
                st.session_state.tipo_conversion_log = conversion_log
                st.experimental_rerun()

        # Mostrar log
        if st.session_state.tipo_conversion_log:
            st.subheader("📋 Historial de conversiones recientes")
            for log in st.session_state.tipo_conversion_log:
                if log.startswith("✔️"):
                    st.success(log)
                else:
                    st.error(log)


        # ===== Análisis de valores faltantes =====
        st.subheader("🚫 Análisis de valores faltantes")
        nulos_df = df.isnull().mean().reset_index()
        nulos_df.columns = ['Columna', 'Porcentaje Nulos']
        nulos_df['Porcentaje Nulos'] = (nulos_df['Porcentaje Nulos'] * 100).round(2)
        st.dataframe(nulos_df, use_container_width=True)
        st.bar_chart(data=nulos_df.set_index('Columna'))

        col_nulas_altas = nulos_df[nulos_df['Porcentaje Nulos'] > 50]['Columna'].tolist()
        if col_nulas_altas:
            st.warning(f"Columnas con más de 50% de valores nulos: {', '.join(col_nulas_altas)}")
            if st.checkbox("🗑️ Eliminar columnas con >50% de nulos"):
                df = df.drop(columns=col_nulas_altas)
                st.session_state.df_original = df
                st.success("Columnas eliminadas correctamente.")
                st.rerun()

        # ===== Duplicados =====
        st.subheader("📎 Detección de registros duplicados")
        duplicados = df.duplicated().sum()
        if duplicados > 0:
            st.warning(f"Se encontraron {duplicados} registros duplicados.")
            if st.checkbox("Eliminar registros duplicados"):
                df = df.drop_duplicates()
                st.session_state.df_original = df
                st.success("✔️ Duplicados eliminados correctamente.")
                st.rerun()
        else:
            st.success("✅ No se encontraron registros duplicados.")

        # ===== Estadísticas =====
        st.subheader("📈 Estadísticas descriptivas")
        st.dataframe(df.describe().T, use_container_width=True)


# ===================  Preprocesamiento ===================
with tabs[1]:
    st.header("⚙️ Preprocesamiento")
    if st.session_state.df_original is not None:
        if st.button("⚙️ Ejecutar preprocesamiento completo"):
            with st.spinner("Procesando datos..."):
                try:
                    df_processed, log = Preprocessing.ejecutar_preprocesamiento_completo(st.session_state.df_original)
                    st.session_state.df_preprocessed = df_processed
                    st.session_state.preprocessing_log = log
                    st.success("✅ Preprocesamiento completado con éxito")
                except Exception as e:
                    st.error(f"❌ Error en el preprocesamiento: {str(e)}")
        if st.session_state.df_preprocessed is not None:
            st.subheader("✅ Datos preprocesados")
            st.dataframe(st.session_state.df_preprocessed.head(50), use_container_width=True)
        if st.session_state.preprocessing_log:
            with st.expander("📝 Log del preprocesamiento"):
                st.json(st.session_state.preprocessing_log)

#--------------bien hasta aqui------------------
# =================== TAB 3: Análisis ===================
with tabs[2]:
    st.header("📉 Análisis")

    if st.session_state.df_preprocessed is not None:
        from module.feature_selection.fase_seleccion import FaseSeleccionRobusta

        df_analysis = st.session_state.df_preprocessed
        selector = FaseSeleccionRobusta(df_analysis)

        analisis_modo = st.radio("¿Cómo deseas seleccionar la variable dependiente?", (
            "Seleccionar manualmente",
            "Detectar automáticamente una cantidad específica",
            "Detectar automáticamente todas las dependientes más fuertes"
        ))

        if "variables_detectadas" not in st.session_state:
            st.session_state.variables_detectadas = []
        if "variable_seleccionada" not in st.session_state:
            st.session_state.variable_seleccionada = None

        # --- SELECCIÓN MANUAL ---
        if analisis_modo == "Seleccionar manualmente":
            target_column = st.selectbox("Selecciona la variable objetivo:", df_analysis.columns)
            if st.button("💾 Guardar selección manual"):
                try:
                    selector.target_column = target_column
                    top_vars, _ = selector.determinar_variables_dependientes(cantidad=1)
                    metricas = selector.obtener_metricas()
                    score = metricas.get(target_column, 1.0)

                    # Guardar selección
                    st.session_state.variables_dependientes = [target_column]
                    st.session_state.metricas_analisis = {target_column: score}
                    st.session_state.target_column = target_column

                    # 🚨 DETECTAR AUTOMÁTICAMENTE EL TIPO DE PROBLEMA USANDO EL LOG ORIGINAL
                    if 'preprocessing_log' in st.session_state:
                        categorical_cols = st.session_state.preprocessing_log.get('deteccion', {}).get('columnas_categoricas', [])
                    else:
                        categorical_cols = []

                    if target_column in categorical_cols:
                        st.session_state.tipo_problema = "clasificacion"
                        st.info("🔎 Se detectó un **problema de Clasificación** — Se usará **logarithmic loss** como función objetivo.")
                    else:
                        st.session_state.tipo_problema = "Regresion"
                        st.info("🔎 Se detectó un **problema de Regresión** — Se usará **MSE** como función objetivo.")

                    st.success(f"✓ Selección manual guardada: `{target_column}` — Score: {score:.4f}")

                    # Gráfico de importancia
                    columnas, scores_plot = zip(*sorted(metricas.items(), key=lambda x: x[1], reverse=True))
                    import plotly.express as px
                    fig = px.bar(
                        x=columnas,
                        y=scores_plot,
                        labels={'x': 'Variable', 'y': 'Score'},
                        title="Importancia según Información Mutua y Correlaciones"
                    )
                    fig.update_traces(marker_color=['red' if col == target_column else 'blue' for col in columnas])
                    st.plotly_chart(fig, use_container_width=True)

                except ValueError as e:
                    st.error(f"❌ Error: {str(e)}")

        # --- DETECCIÓN AUTOMÁTICA ---
        else:
            cantidad = None
            threshold = None
            if analisis_modo == "Detectar automáticamente una cantidad específica":
                cantidad = st.number_input(
                    "¿Cuántas variables dependientes deseas determinar?", 
                    min_value=1, max_value=len(df_analysis.columns)-1, value=1, step=1
                )
            elif analisis_modo == "Detectar automáticamente todas las dependientes más fuertes":
                threshold = st.slider(
                    "Umbral mínimo de importancia (score):", 
                    min_value=0.0, max_value=1.0, value=0.1, step=0.01
                )

            if st.button("🔍 Detectar variables dependientes"):
                try:
                    if cantidad:
                        top_vars, valores = selector.determinar_variables_dependientes(cantidad=cantidad)
                        seleccionadas = list(zip(top_vars, valores))
                    elif threshold is not None:
                        _ = selector.determinar_variables_dependientes(cantidad=len(df_analysis.columns))
                        metricas = selector.obtener_metricas()
                        seleccionadas = [(k, v) for k, v in metricas.items() if v >= threshold]

                    if not seleccionadas:
                        st.warning("⚠️ No se encontraron variables con el score suficiente.")
                    else:
                        # Guardar lista de variables detectadas en session_state
                        st.session_state.variables_detectadas = seleccionadas

                        # 🚨 DETECTAR TIPO DE PROBLEMA PARA LA PRIMERA VARIABLE DETECTADA
                        if 'preprocessing_log' in st.session_state:
                            categorical_cols = st.session_state.preprocessing_log.get('deteccion', {}).get('columnas_categoricas', [])
                        else:
                            categorical_cols = []

                        primera_var = seleccionadas[0][0]
                        if primera_var in categorical_cols:
                            st.session_state.tipo_problema = "clasificacion"
                            st.info("🔎 Se detectó un **problema de Clasificación** — Se usará **Accuracy** como función objetivo.")
                        else:
                            st.session_state.tipo_problema = "estimacion"
                            st.info("🔎 Se detectó un **problema de Regresión** — Se usará **MSE** como función objetivo.")

                except ValueError as e:
                    st.error(f"❌ Error: {str(e)}")

            # Mostrar tabla de variables detectadas
            if st.session_state.variables_detectadas:
                st.markdown("### 🔬 Variables detectadas:")
                for idx, (var, score) in enumerate(st.session_state.variables_detectadas):
                    cols = st.columns([0.1, 0.6, 0.3])
                    cols[0].markdown(f"{idx+1}")
                    cols[1].markdown(f"**{var}** — Score: `{score:.4f}`")
                    selected = cols[2].checkbox(
                        "Seleccionar",
                        value=(st.session_state.variable_seleccionada == var),
                        key=f"chk_{var}",
                        on_change=lambda v=var: st.session_state.update({"variable_seleccionada": v})
                    )

                # Botón guardar selección
                if st.button("💾 Guardar variable seleccionada"):
                    if st.session_state.variable_seleccionada:
                        seleccionada = st.session_state.variable_seleccionada
                        st.session_state.variables_dependientes = [seleccionada]
                        st.session_state.metricas_analisis = {seleccionada: dict(st.session_state.variables_detectadas)[seleccionada]}
                        st.session_state.target_column = seleccionada
                        st.success(f"✓ Variable dependiente guardada: `{seleccionada}`")

                        # Gráfico de importancia
                        columnas, scores_plot = zip(*sorted(dict(st.session_state.variables_detectadas).items(), key=lambda x: x[1], reverse=True))
                        import plotly.express as px
                        fig = px.bar(
                            x=columnas,
                            y=scores_plot,
                            labels={'x': 'Variable', 'y': 'Score'},
                            title="Importancia según Información Mutua y Correlaciones"
                        )
                        fig.update_traces(marker_color=['red' if col == seleccionada else 'blue' for col in columnas])
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("⚠️ Debes seleccionar una variable antes de guardar.")

    else:
        st.warning("🔄 Primero debes realizar el preprocesamiento de datos para usar esta sección.")

# =================== algoritmo evolutivo ===================
with tabs[3]:
    st.header("🧠 Algoritmo Evolutivo")

    if 'df_preprocessed' in st.session_state and 'variables_dependientes' in st.session_state:
        st.subheader("Variables objetivo seleccionadas:")
        st.table(pd.DataFrame({
            'Variable': st.session_state.variables_dependientes,
            'Score': [st.session_state.metricas_analisis[v] 
                      for v in st.session_state.variables_dependientes]
        }))

        # 🔽 Selección del modelo
        if 'tipo_problema' in st.session_state and st.session_state.tipo_problema.lower().startswith("clas"):
            modelos_disponibles = ["RandomForest", "GradientBoosting", "LogisticRegression", "KNN"]
        else:
            modelos_disponibles = ["RandomForest", "GradientBoosting", "LinearRegression", "Ridge", "SVR", "KNN"]

        modelo_seleccionado = st.selectbox("Selecciona el modelo a usar:", modelos_disponibles)


        st.markdown("### ⚙️ Configuración Rápida")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            n_poblacion = st.number_input("Tamaño población", min_value=2, max_value=1000, value=10)
        with col2:
            n_generaciones = st.number_input("Generaciones", min_value=1, max_value=1000, value=10)
        with col3:
            prob_cruce = st.number_input("Probabilidad de cruce (%)", min_value=1, max_value=100, value=5) / 100
        with col4:
            prob_mut = st.number_input("Probabilidad de mutación (%)", min_value=0, max_value=100, value=3) / 100
        with col5:
            min_vars = st.number_input("Mínimo de variables combinadas", 
                                       min_value=1, 
                                       max_value=len(st.session_state.df_preprocessed.columns)-1, 
                                       value=2)

        if st.button("⚡ Ejecutar algoritmo evolutivo"):
            try:
                if not isinstance(st.session_state.df_preprocessed, pd.DataFrame) or st.session_state.df_preprocessed.empty:
                    raise ValueError("Los datos preprocesados no son válidos.")
                if not st.session_state.variables_dependientes:
                    raise ValueError("No se han seleccionado variables dependientes.")
                for var in st.session_state.variables_dependientes:
                    if var not in st.session_state.df_preprocessed.columns:
                        raise ValueError(f"La variable dependiente '{var}' no existe en los datos preprocesados.")
                
                categorical_cols = []
                if 'preprocessing_log' in st.session_state:
                    categorical_cols = st.session_state.preprocessing_log.get('deteccion', {}).get('columnas_categoricas', [])

                with st.spinner(f"Ejecutando versión con {n_poblacion} individuos y {n_generaciones} generaciones..."):
                    start_time = time.time()
                    fast_evo = AlgoritmoEvolutivo(
                        data=st.session_state.df_preprocessed,
                        target_vars=st.session_state.variables_dependientes,
                        n_poblacion=n_poblacion,
                        n_generaciones=n_generaciones,
                        prob_mut=prob_mut,
                        prob_cruce=prob_cruce,
                        min_vars=min_vars,
                        modelo_seleccionado=modelo_seleccionado
                    )
                    resultados = fast_evo.ejecutar()
                    end_time = time.time()
                    execution_time = end_time - start_time

                    if 'historial_ejecuciones' not in st.session_state:
                        st.session_state.historial_ejecuciones = []
                    st.session_state.historial_ejecuciones.append({
                        "poblacion": n_poblacion,
                        "generaciones": n_generaciones,
                        "prob_cruce": prob_cruce,
                        "prob_mut": prob_mut,
                        "min_vars": min_vars,
                        "fitness": resultados['fitness'],
                        "variables": resultados['variables'],
                        "tiempo_ejecucion": execution_time
                    })

                    st.session_state.resultados_evolutivos = resultados
                    st.session_state.optimizacion_completa = True
                    st.success("¡Optimización completada!")

                    st.subheader("Variables seleccionadas:")
                    st.write(resultados['variables'])
                    st.metric("Fitness obtenido", value=f"{resultados['fitness']:.20f}")
                    st.write(f"Tiempo de ejecución: {execution_time:.4f} segundos")
                    st.write(f"Total de variables seleccionadas: {resultados['total_vars']}")

            except Exception as e:
                st.error(f"Error en la optimización: {str(e)}")
    else:
        st.warning("Complete primero el preprocesamiento y selección de variables")
# =================== resultados ===================
with tabs[4]:
    st.header("📊 Resultados y Historial")

    if 'resultados_evolutivos' in st.session_state and st.session_state.optimizacion_completa:
        resultados = st.session_state.resultados_evolutivos

        # Mostrar tipo de problema detectado
        st.subheader("🔍 Tipo de problema detectado")
        if resultados['task_type'] == "classification":
            st.write("**Clasificación** (se usa logarithmic loss como métrica)")
        else:
            st.write("**Regresión** (se usa MSE como métrica)")

        # Mejor modelo final (valores desde resultados, pero priorizamos el historial cuando sea posible)
        mejor_modelo = resultados['modelo']
        resultados_modelos = resultados.get('resultados_modelos', {})

        # Intentar obtener el mejor fitness real desde el historial del modelo (si existe)
        historial_modelo_raw = resultados.get('historial_por_modelo', {}).get(mejor_modelo, [])
        # filtrar valores válidos
        valid_hist_vals = [v for v in historial_modelo_raw if v is not None and np.isfinite(v)]
        if valid_hist_vals:
            mejor_fitness_modelo = float(np.min(valid_hist_vals))
        else:
            # fallback a lo que devuelva el algoritmo para el individuo ganador
            if mejor_modelo in resultados_modelos and resultados_modelos[mejor_modelo] is not None:
                mejor_fitness_modelo = resultados_modelos[mejor_modelo]
            else:
                mejor_fitness_modelo = resultados.get('fitness', float("nan"))

        st.subheader("📌 Resumen de la última ejecución")
        st.write("Variables seleccionadas:", resultados['variables'])
        # Mostrar el fitness (si es NaN, Streamlit mostrará 'nan' pero es improbable)
        st.metric("Fitness obtenido", f"{mejor_fitness_modelo:.15f}")
        st.write(f"Total de variables seleccionadas: **{resultados['total_vars']}**")
        st.write(f"Mejor modelo final: **{mejor_modelo}**")

        # Evolución del fitness
        historial_modelo = historial_modelo_raw  # ya obtenido arriba

        if not historial_modelo:
            st.warning("No hay historial disponible para el modelo seleccionado.")
            historial_ganador = []
        else:
            # recortar al número de generaciones mostrado (si existe n_generaciones en scope)
            historial_raw = historial_modelo[:n_generaciones]

            # Construir mínimo acumulado (cumulative minimum). Siempre minimizamos: menor = mejor.
            historial_ganador = []
            mejor_actual = float("inf")
            for f in historial_raw:
                # si valor inválido, dejamos el mejor_actual (si aún inf -> mantendremos Inf/nan)
                if f is None or (isinstance(f, float) and not np.isfinite(f)):
                    # si aún no tenemos un mejor_actual válido, añadimos NaN para que la gráfica ignore
                    historial_ganador.append(mejor_actual if np.isfinite(mejor_actual) else np.nan)
                    continue
                # actualizar mínimo acumulado
                if f < mejor_actual:
                    mejor_actual = f
                historial_ganador.append(mejor_actual)

        # Crear DataFrame solo si hay datos efectivos
        if historial_ganador and any([not (isinstance(v, float) and np.isnan(v)) for v in historial_ganador]):
            df_historial = pd.DataFrame({
                "Generación": list(range(1, len(historial_ganador) + 1)),
                "Fitness": historial_ganador
            })

            st.subheader(f"📈 Evolución del fitness del modelo ({mejor_modelo})")
            st.dataframe(df_historial.style.format({"Fitness": "{:.15f}"}))

            # Para la línea roja usamos el mínimo real del historial mostrado (si existe)
            valid_plot_vals = [v for v in df_historial['Fitness'].tolist() if np.isfinite(v)]
            if valid_plot_vals:
                mejor_fitness_plotted = float(np.min(valid_plot_vals))
            else:
                mejor_fitness_plotted = mejor_fitness_modelo  # fallback

            plt.figure(figsize=(10,5))
            plt.plot(df_historial['Generación'], df_historial['Fitness'], marker='o', color='blue', label="mínimo acumulado por generación")
            # dibujar la línea roja en el mínimo real encontrado
            if np.isfinite(mejor_fitness_plotted):
                plt.axhline(y=mejor_fitness_plotted, color='red', linestyle='--',
                            label=f"Mejor fitness final ({mejor_fitness_plotted:.15f})")
            plt.xlabel("Generación")
            plt.ylabel("Fitness (menor es mejor)")
            plt.title(f"Evolución del fitness - Modelo ({mejor_modelo})")
            plt.grid(True)
            plt.legend()
            st.pyplot(plt)
        else:
            st.info("No hay datos de historial válidos para mostrar la evolución del fitness.")

        # Comparativa entre modelos: calcular el mejor fitness válido por modelo (seguro contra None/inf)
        rows = []
        for k, v in resultados.get('historial_por_modelo', {}).items():
            vals = [x for x in v if x is not None and np.isfinite(x)]
            best_val = float(np.min(vals)) if vals else None
            rows.append((k, best_val))

        df_modelos = pd.DataFrame(rows, columns=["Modelo", "Mejor Fitness"])
        if df_modelos["Mejor Fitness"].notna().any():
            df_modelos = df_modelos.sort_values("Mejor Fitness", na_position="last")

        st.subheader("🏆 Modelo utilizado")
        st.dataframe(df_modelos.style.format({"Mejor Fitness": "{:.15f}"}))


    # ================== Historial de ejecuciones ==================
    if 'historial_ejecuciones' in st.session_state and st.session_state.historial_ejecuciones:
        st.subheader("📋 Historial de ejecuciones")
        df_historial_total = pd.DataFrame(st.session_state.historial_ejecuciones)
        st.dataframe(df_historial_total)
