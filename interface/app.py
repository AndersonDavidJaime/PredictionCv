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


# Configuración de página
st.set_page_config(page_title="PredictionCV", layout="wide")
st.title("📊 Predicción CV - Carga y preprocesamiento de datos")
tabs = st.tabs(["               📁 Cargar datos              ", "⚙️ Preprocesamiento", "📉 Análisis", "🧠 Algoritmo Evolutivo", "📊 Resultados"])

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

        # Inicializamos variables de sesión
        if "variables_detectadas" not in st.session_state:
            st.session_state.variables_detectadas = []
        if "variable_seleccionada" not in st.session_state:
            st.session_state.variable_seleccionada = None

        if analisis_modo == "Seleccionar manualmente":
            target_column = st.selectbox("Selecciona la variable objetivo:", df_analysis.columns)
            if st.button("💾 Guardar selección manual"):
                try:
                    selector.target_column = target_column
                    top_vars, _ = selector.determinar_variables_dependientes(cantidad=1)
                    metricas = selector.obtener_metricas()
                    score = metricas.get(target_column, 1.0)

                    st.session_state.variables_dependientes = [target_column]
                    st.session_state.metricas_analisis = {target_column: score}
                    st.session_state.target_column = target_column

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

        else:
            # Detectar automáticamente
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

                except ValueError as e:
                    st.error(f"❌ Error: {str(e)}")

            # Mostrar tabla de variables detectadas si existe
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
                
                with st.spinner(f"Ejecutando versión con {n_poblacion} individuos y {n_generaciones} generaciones..."):
                    start_time = time.time()
                    fast_evo = AlgoritmoEvolutivo(
                        data=st.session_state.df_preprocessed,
                        target_vars=st.session_state.variables_dependientes,
                        n_poblacion=n_poblacion,
                        n_generaciones=n_generaciones,
                        prob_mut=prob_mut,
                        prob_cruce=prob_cruce,
                        min_vars=min_vars
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
                    st.metric("Fitness obtenido (MSE)", value=f"{resultados['fitness']:.20f}")
                    st.write(f"Tiempo de ejecución: {execution_time:.4f} segundos")
                    st.write(f"Total de variables seleccionadas: {resultados['total_vars']}")

            except Exception as e:
                st.error(f"Error en la optimización: {str(e)}")
    else:
        st.warning("Complete primero el preprocesamiento y selección de variables")

# =================== resultados ===================
with tabs[4]:
    st.header("📊 Resultados y Historial")

    # Mostrar resultados de la última ejecución
    if 'resultados_evolutivos' in st.session_state and st.session_state.optimizacion_completa:
        resultados = st.session_state.resultados_evolutivos
        
        st.subheader("Variables seleccionadas (última ejecución):")
        st.write(resultados['variables'])
        
        st.metric("Fitness obtenido (MSE)", value=f"{resultados['fitness']:.15f}")
        st.write(f"Total de variables seleccionadas: {resultados['total_vars']}")
        
        # Mostrar tabla de fitness por generación
        if 'historial_fitness' in resultados and resultados['historial_fitness']:
            import matplotlib.pyplot as plt
            
            df_fitness = pd.DataFrame({
                'Generación': list(range(1, len(resultados['historial_fitness']) + 1)),
                'Fitness Óptimo': resultados['historial_fitness']
            })

            st.subheader("Fitness Óptimo por Generación")
            st.dataframe(df_fitness.style.format({'Fitness Óptimo': '{:.15f}'}))
            
            # Gráfica de convergencia
            plt.figure(figsize=(10, 5))
            plt.plot(df_fitness['Generación'], df_fitness['Fitness Óptimo'], marker='o', color='blue', label="Fitness óptimo")
            mejor_fitness_final = resultados['historial_fitness'][-1]
            plt.axhline(y=mejor_fitness_final, color='red', linestyle='--', label=f"Mejor fitness final ({mejor_fitness_final:.15f})")
            plt.title("Evolución del Fitness a lo largo de las Generaciones")
            plt.xlabel("Generación")
            plt.ylabel("Fitness (menor es mejor)")
            plt.xticks(df_fitness['Generación'])
            plt.grid(True)
            plt.legend()
            st.pyplot(plt)
    
    # Mostrar historial de ejecuciones
    if 'historial_ejecuciones' in st.session_state and st.session_state.historial_ejecuciones:
        st.subheader("📋 Historial de ejecuciones")
        df_historial = pd.DataFrame(st.session_state.historial_ejecuciones)
        st.dataframe(df_historial)
