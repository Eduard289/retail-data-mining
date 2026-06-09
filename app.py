import streamlit as st
import pandas as pd

# Configuración de la página
st.set_page_config(page_title="Analítica Retail", layout="wide")

# Cabecera principal
st.title("🌍 Analítica Retail: Segmentación Internacional")
st.markdown("---")

# 1. Motor de Ingesta en RAM (Caché para no saturar la red)
@st.cache_data(show_spinner="Conectando con la base de datos de la UCI... (Esto puede tardar 1 minuto la primera vez)")
def cargar_datos():
    # Petición HTTP al dataset oficial
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    df = pd.read_excel(url)
    
    # Limpieza en RAM: Quitamos tickets nulos y devoluciones (que empiezan por 'C')
    df = df.dropna(subset=['InvoiceNo'])
    df = df[~df['InvoiceNo'].astype(str).str.contains('C')]
    return df

try:
    df_retail = cargar_datos()
    
    # 2. Sidebar para la segmentación geográfica
    st.sidebar.header("⚙️ Configuración del Análisis")
    st.sidebar.markdown("Filtra el dataset por país para auditar el comportamiento local.")
    
    paises = sorted(df_retail['Country'].unique())
    # Por defecto, buscamos España para que arranque ahí
    indice_defecto = paises.index('Spain') if 'Spain' in paises else 0
    pais_seleccionado = st.sidebar.selectbox("Selecciona el Mercado:", paises, index=indice_defecto)

    # 3. Filtrado reactivo del dataframe
    df_filtrado = df_retail[df_retail['Country'] == pais_seleccionado]

    # 4. Visualización del Dashboard Base
    st.subheader(f"📊 Mercado en auditoría: {pais_seleccionado}")
    
    # Métricas clave en columnas
    col1, col2, col3 = st.columns(3)
    col1.metric("Tickets Únicos", f"{df_filtrado['InvoiceNo'].nunique():,}")
    col2.metric("Líneas de Artículo", f"{len(df_filtrado):,}")
    col3.metric("Clientes Identificados", f"{df_filtrado['CustomerID'].nunique():,}")

    st.write("Vista previa del ADN transaccional (Primeras 10 líneas):")
    st.dataframe(df_filtrado.head(10))

except Exception as e:
    st.error(f"Error de conexión o lectura: {e}")
