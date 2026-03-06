import streamlit as st
import pandas as pd
import joblib
from sklearn.datasets import load_breast_cancer

# Configuración de la página
st.set_page_config(page_title="Detector de Tumores AI", page_icon="🩺", layout="centered")

st.title("🩺 Asistente de Diagnóstico de Cáncer de Mama (Motor: XGBoost)")
st.write("Ingresa los parámetros del estudio para obtener una predicción del modelo.")

# 1. Cargar el modelo XGBoost y obtener columnas automáticamente
@st.cache_resource 
def cargar_recursos():
    # 🌟 CAMBIO CLAVE: Ahora cargamos el modelo XGBoost
    modelo = joblib.load('modelo_cancer_xgb.pkl')
    
    # Rescatamos los nombres de las 30 columnas directamente de sklearn
    datos = load_breast_cancer()
    columnas = datos.feature_names.tolist() 
    
    return modelo, columnas

modelo, columnas = cargar_recursos()

# 2. Crear el formulario de entrada en la barra lateral
with st.sidebar:
    st.header("Parámetros del Tumor")
    
    # --- SECCIÓN DE CONTEXTO MÉDICO ---
    with st.expander("ℹ️ ¿De dónde vienen estos datos?", expanded=False):
        st.write("""
        Los parámetros de este modelo **no** son las medidas del tumor completo (como las que da un ultrasonido o mastografía). 
        
        Provienen de un procedimiento llamado **Biopsia por Punción con Aguja Fina (PAAF)**. 
        El tejido extraído se digitaliza y se analiza la imagen a nivel microscópico para medir la geometría de los **núcleos de las células**.
        
        * **Radio/Perímetro/Área:** Indican el tamaño del núcleo celular.
        * **Suavidad/Concavidad:** Evalúan si el borde de la célula es regular o tiene deformaciones atípicas.
        """)
    st.divider()
    
    # Menú para elegir el tipo de entrada de datos
    tipo_caso = st.radio(
        "Cargar datos automáticos:",
        ("Escribir a mano", "Caso Maligno", "Caso Benigno")
    )
    
    # Datos reales extraídos del dataset para pruebas
    valores_malignos = [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
    valores_benignos = [13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766, 0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023, 15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259]
    
    input_data = {}
    
    # Generamos los campos de entrada dinámicamente
    with st.expander("Ver y editar los 30 parámetros", expanded=False):
        for i, col in enumerate(columnas):
            if tipo_caso == "Caso Maligno":
                valor_defecto = float(valores_malignos[i])
            elif tipo_caso == "Caso Benigno":
                valor_defecto = float(valores_benignos[i])
            else:
                valor_defecto = 0.0 
                
            input_data[col] = st.number_input(f"{col}", value=valor_defecto, format="%.4f")

# 3. Lógica de predicción y validación
if st.button("Realizar Diagnóstico", type="primary"):
    
    df_input = pd.DataFrame([input_data])
    
    # Validación de seguridad
    if df_input['mean radius'][0] == 0.0 or df_input['mean area'][0] == 0.0:
        st.warning("⚠️ Error de captura: Las medidas principales del tumor (como el radio o el área) no pueden ser cero. Por favor ingresa valores biológicamente posibles o selecciona un caso de prueba.")
    
    else:
        # Hacemos la predicción con XGBoost
        prediccion = modelo.predict(df_input)
        probabilidades = modelo.predict_proba(df_input)
        
        st.divider()
        
        # XGBoost suele devolver las probabilidades en el mismo formato, pero es mucho más asertivo.
        if prediccion[0] == 0:
            st.error(f"### 🚨 Resultado: Probable Maligno")
            st.write(f"Nivel de confianza del modelo: **{probabilidades[0][0]*100:.2f}%**")
        else:
            st.success(f"### ✅ Resultado: Probable Benigno")
            st.write(f"Nivel de confianza del modelo: **{probabilidades[0][1]*100:.2f}%**")
            
        st.info("Nota legal: Esta herramienta es un ejercicio de Machine Learning con fines educativos y no sustituye de ninguna manera un diagnóstico médico profesional.")