from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import sys
import os
import plotly.express as px



# Importar el preprocesamiento desde el módulo externo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module.preprocessing.preprocessing import ejecutar_preprocesamiento_completo 
from module.feature_selection.fase_seleccion import FaseSeleccion
from module.evolutionary.algoritmo_evolutivo import AlgoritmoEvolutivo


# Configuración de página
st.set_page_config(page_title="PredictionCV", layout="wide")
st.title("📊 Predicción CV - Carga y preprocesamiento de datos")
tabs = st.tabs(["📁 Cargar datos", "⚙️ Preprocesamiento", "📉 Análisis", "🧠 Algoritmo Evolutivo", "📊 Resultados"])

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
import pandas as pd
import numpy as np
from category_encoders import TargetEncoder, LeaveOneOutEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import f_oneway
import streamlit as st

# ===================  Preprocesamiento ===================
def detectar_columnas_categoricas(df):
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

def codificar_columnas_categoricas(df, columnas_cat, log):
    df_encoded = df.copy()
    encoder = TargetEncoder()

    for col in columnas_cat:
        otras_columnas = df_encoded.drop(columns=[col])
        numericas = otras_columnas.select_dtypes(include=[np.number])

        if not numericas.empty:
            f_scores = []
            for num_col in numericas.columns:
                grupos = [df_encoded[df_encoded[col] == val][num_col].dropna() for val in df_encoded[col].unique()]
                if len(grupos) > 1:
                    try:
                        _, p_valor = f_oneway(*grupos)
                        f_scores.append(p_valor)
                    except:
                        continue

            if f_scores and min(f_scores) < 0.05:
                df_encoded[col] = encoder.fit_transform(df_encoded[col], df_encoded[numericas.columns[0]])
                log.append(f"✔️ Codificación Target aplicada a '{col}' por su relación estadística con variables numéricas.")
            else:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                log.append(f"✔️ Codificación Label aplicada a '{col}' por falta de correlación.")
        else:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            log.append(f"✔️ Codificación Label aplicada a '{col}' por ausencia de numéricas.")

    return df_encoded, log

def ejecutar_preprocesamiento_completo(df_original):
    df = df_original.copy()
    log_conversiones = []
    columnas_removidas = []

    # 1. Detección del tipo de dataset
    columnas_categoricas = detectar_columnas_categoricas(df)
    es_dataset_numerico = len(columnas_categoricas) == 0

    # 2. Limpieza de columnas duplicadas y filas duplicadas
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.drop_duplicates()

    # 3. Eliminación de columnas con demasiados nulos
    umbral_nulos = 0.8
    cols_nulas = df.columns[df.isnull().mean() > umbral_nulos].tolist()
    if cols_nulas:
        df.drop(columns=cols_nulas, inplace=True)
        columnas_removidas.extend(cols_nulas)
        log_conversiones.append(f"✔️ Columnas eliminadas por alto porcentaje de nulos: {cols_nulas}")

    # 4. Tratamiento de nulos restantes
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].median(), inplace=True)
                log_conversiones.append(f"✔️ Nulos en '{col}' rellenados con la mediana.")
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
                log_conversiones.append(f"✔️ Nulos en '{col}' rellenados con la moda.")

    # 5. Outliers por IQR
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower, lower, np.where(df[col] > upper, upper, df[col]))
        log_conversiones.append(f"✔️ Outliers tratados en '{col}' con método IQR.")

    # 6. Eliminación de columnas de varianza baja
    selector = VarianceThreshold(threshold=1e-6)
    try:
        selector.fit(df.select_dtypes(include=[np.number]))
        columnas_filtradas = df.select_dtypes(include=[np.number]).columns[~selector.get_support()].tolist()
        if columnas_filtradas:
            df.drop(columns=columnas_filtradas, inplace=True)
            columnas_removidas.extend(columnas_filtradas)
            log_conversiones.append(f"✔️ Columnas eliminadas por baja varianza: {columnas_filtradas}")
    except Exception:
        pass

    # 7. Codificación de columnas categóricas (solo si hay)
    if not es_dataset_numerico and columnas_categoricas:
        df, log_conversiones = codificar_columnas_categoricas(df, columnas_categoricas, log_conversiones)

    # Resultado final
    columnas_finales = df.columns.tolist()
    log = {
        "estado": "éxito",
        "columnas_finales": columnas_finales,
        "columnas_removidas": columnas_removidas,
        "conversiones": log_conversiones
    }

    return df, log

# Integración con Streamlit
with tabs[1]:
    st.header("⚙️ Preprocesamiento")
    if st.session_state.df_original is not None:
        if st.button("⚙️ Ejecutar preprocesamiento completo"):
            with st.spinner("Procesando datos..."):
                try:
                    df_processed, log = ejecutar_preprocesamiento_completo(st.session_state.df_original)
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
                log = st.session_state.preprocessing_log
                st.markdown("**Estado:**")
                st.write("-", log.get("estado", ""))

                st.markdown("**Columnas finales:**")
                st.write(log.get("columnas_finales", []))

                st.markdown("**Columnas removidas:**")
                st.write(log.get("columnas_removidas", []))

                st.markdown("**Conversiones aplicadas:**")
                for line in log.get("conversiones", []):
                    st.write("-", line)



# =================== TAB 3: Análisis ===================
with tabs[2]:
    st.header("📉 Análisis")

    if st.session_state.df_preprocessed is not None:
        from module.feature_selection.fase_seleccion import FaseSeleccion

        df_analysis = st.session_state.df_preprocessed
        selector = FaseSeleccion(df_analysis)

        analisis_modo = st.radio("¿Cómo deseas seleccionar la variable dependiente?", (
            "Seleccionar manualmente",
            "Detectar automáticamente una cantidad específica",
            "Detectar automáticamente todas las dependientes más fuertes"
        ))

        if analisis_modo == "Seleccionar manualmente":
            target_column = st.selectbox("Selecciona la variable objetivo:", df_analysis.columns)
            st.session_state.target_column = target_column  # Guardar en session_state
            if st.button("💾 Guardar selección manual"):
                st.session_state.variables_dependientes = [target_column]
                st.session_state.metricas_analisis = {target_column: 1.0}  # Valor dummy para el score
                st.success("✓ Selección manual guardada")

        elif analisis_modo == "Detectar automáticamente una cantidad específica":
            cantidad = st.number_input("¿Cuántas variables dependientes deseas determinar?", 
                                       min_value=1, max_value=len(df_analysis.columns) - 1, value=1, step=1)

            if st.button("🔍 Detectar variables dependientes"):
                try:
                    top_vars, scores = selector.determinar_variables_dependientes(cantidad=cantidad)
                    for i, (var, score) in enumerate(zip(top_vars, scores)):
                        st.markdown(f"**{i+1}. `{var}`** — Score de dependencia: `{score:.4f}`")
                    st.session_state.variables_dependientes = top_vars  # Guardar todas las variables detectadas
                    st.session_state.metricas_analisis = dict(zip(top_vars, scores))  # Guardar scores
                    st.session_state.target_column = top_vars[0]  # Guardar la primera variable detectada

                    if st.button("💾 Guardar selección automática"):
                        st.success("✓ Selección automática guardada")
                except ValueError as e:
                    st.error(f"❌ Error: {str(e)}")

        elif analisis_modo == "Detectar automáticamente todas las dependientes más fuertes":
            threshold = st.slider("Umbral mínimo de importancia (score):", 
                                  min_value=0.0, max_value=1.0, value=0.1, step=0.01)

            if st.button("🔍 Detectar automáticamente"):
                try:
                    top_vars, scores = selector.determinar_variables_dependientes(cantidad=None)  
                    metricas = selector.obtener_metricas()  

                    seleccionadas = [(k, v) for k, v in metricas.items() if v >= threshold]

                    if not seleccionadas:
                        st.warning("⚠️ No se encontraron variables con un score mayor al umbral seleccionado.")
                    else:
                        seleccionadas.sort(key=lambda x: x[1], reverse=True)
                        columnas, scores = zip(*seleccionadas)

                        st.markdown("### 🔬 Variables seleccionadas automáticamente:")
                        for i, (col, score) in enumerate(seleccionadas):
                            st.markdown(f"**{i+1}. `{col}`** — Score: `{score:.4f}`")

                        st.subheader("📈 Gráfico de importancia")
                        import plotly.express as px
                        fig = px.bar(x=columnas, y=scores, labels={'x': 'Variable', 'y': 'Score'}, title="Importancia según Información Mutua")
                        st.plotly_chart(fig, use_container_width=True)

                        st.session_state.variables_dependientes = columnas  # Guardar todas las variables seleccionadas
                        st.session_state.metricas_analisis = {col: score for col, score in seleccionadas}  # Guardar scores
                        st.session_state.target_column = columnas[0]  # Guardar la primera variable seleccionada

                        if st.button("💾 Guardar selección por umbral"):
                            st.success("✓ Selección por umbral guardada")
                except ValueError as e:
                    st.error(f"❌ Error: {str(e)}")

    else:
        st.warning("🔄 Primero debes realizar el preprocesamiento de datos para usar esta sección.")



# =================== algoritmo evolutivo ===================
with tabs[3]:
    st.header("🧠 Algoritmo Evolutivo")

    if 'df_preprocessed' in st.session_state and 'variables_dependientes' in st.session_state:
        # Mostrar variables objetivo
        st.subheader("Variables objetivo seleccionadas:")
        st.table(pd.DataFrame({
            'Variable': st.session_state.variables_dependientes,
            'Score': [st.session_state.metricas_analisis[v] 
                    for v in st.session_state.variables_dependientes]
        }))
        
        # Configuración rápida
        st.markdown("### ⚙️ Configuración Rápida")
        
        # Configuración de parámetros
        col1, col2 = st.columns(2)
        with col1:
            n_poblacion = st.number_input("Tamaño población", min_value=2, max_value=30, value=4)
        with col2:
            n_generaciones = st.number_input("Generaciones", min_value=1, max_value=30, value=3)

        # Justo antes de ejecutar el algoritmo evolutivo
        print("Datos de entrada:")
        print(st.session_state.df_preprocessed.head())
        print("Variables dependientes:", st.session_state.variables_dependientes)

        if st.button("⚡ Ejecutar algoritmo evolutivo"):
            try:
                # Verificación de datos de entrada
                if not isinstance(st.session_state.df_preprocessed, pd.DataFrame) or st.session_state.df_preprocessed.empty:
                    raise ValueError("Los datos preprocesados no son válidos.")
                
                if not st.session_state.variables_dependientes:
                    raise ValueError("No se han seleccionado variables dependientes.")
                
                # Verificar que las variables dependientes existan en el DataFrame
                for var in st.session_state.variables_dependientes:
                    if var not in st.session_state.df_preprocessed.columns:
                        raise ValueError(f"La variable dependiente '{var}' no existe en los datos preprocesados.")
                
                with st.spinner(f"Ejecutando versión rápida con {n_poblacion} individuos y {n_generaciones} generaciones..."):
                    fast_evo = AlgoritmoEvolutivo(
                        st.session_state.df_preprocessed,
                        st.session_state.variables_dependientes,
                        n_poblacion=n_poblacion,
                        prob_mut=0.03
                    )
                    
                    resultados = fast_evo.ejecutar(n_generaciones=n_generaciones)
                    
                    # Verificación de resultados
                    if not isinstance(resultados, dict) or 'variables' not in resultados:
                        raise ValueError("Los resultados no son válidos.")
                    
                    st.session_state.resultados_evolutivos = resultados
                    st.success("¡Optimización rápida completada!")
                    
                    st.subheader("Variables seleccionadas:")
                    st.write(resultados['variables'])
                    st.metric("Fitness obtenido", value=f"{resultados['fitness']:.4f}")
                    
                    st.session_state.optimizacion_completa = True
            
            except Exception as e:
                st.error(f"Error en la optimización: {str(e)}")
                st.session_state.optimizacion_completa = False
    else:
        st.warning("Complete primero el preprocesamiento y selección de variables")

        
# =================== resultados ===================
with tabs[4]:
    st.header("📊 Resultados")
    
    # Verifica si se han completado los resultados de la optimización
    if 'resultados_evolutivos' in st.session_state and st.session_state.optimizacion_completa:
        resultados = st.session_state.resultados_evolutivos
        
        # Mostrar las variables seleccionadas
        st.subheader("Variables seleccionadas:")
        st.write(resultados['variables'])
        
        # Mostrar el fitness obtenido
        st.metric("Fitness obtenido", value=f"{resultados['fitness']:.4f}")
        
        # Mostrar el total de variables seleccionadas
        st.write(f"Total de variables seleccionadas: {resultados['total_vars']}")
        
        # Gráfico de la evolución del fitness
        if 'historial_fitness' in resultados:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 5))
            plt.plot(resultados['historial_fitness'], marker='o')
            plt.title("Evolución del Fitness a lo largo de las Generaciones")
            plt.xlabel("Generación")
            plt.ylabel("Fitness")
            plt.grid()
            st.pyplot(plt)
        else:
            st.warning("No se dispone de datos de evolución del fitness.")
    
    else:
        st.warning("No se han completado los resultados de la optimización. Asegúrate de ejecutar el algoritmo evolutivo primero.")
