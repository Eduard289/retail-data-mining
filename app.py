import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import altair as alt

# Configuración de la página
st.set_page_config(page_title="Analítica Retail", layout="wide")

# Cabecera principal
st.title("🌍 Analítica Retail: Segmentación Internacional")
st.markdown("---")

# 1. Motor de Ingesta en RAM
@st.cache_data(show_spinner="Conectando con la base de datos de la UCI...")
def cargar_datos():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    df = pd.read_excel(url)
    df = df.dropna(subset=['InvoiceNo'])
    df = df[~df['InvoiceNo'].astype(str).str.contains('C')]
    return df

try:
    df_retail = cargar_datos()
    
    # 2. Sidebar para la segmentación geográfica y parámetros del algoritmo
    st.sidebar.header("⚙️ Configuración del Análisis")
    
    paises = sorted(df_retail['Country'].unique())
    indice_defecto = paises.index('Spain') if 'Spain' in paises else 0
    pais_seleccionado = st.sidebar.selectbox("Selecciona el Mercado:", paises, index=indice_defecto)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎛️ Parámetros del Algoritmo")
    soporte_minimo = st.sidebar.slider("Soporte Mínimo (%)", min_value=1, max_value=20, value=5, step=1) / 100.0
    lift_minimo = st.sidebar.slider("Lift Mínimo", min_value=1.0, max_value=10.0, value=1.0, step=0.5)

    # 3. Filtrado reactivo del dataframe
    df_filtrado = df_retail[df_retail['Country'] == pais_seleccionado]

    # Visualización de métricas
    col1, col2, col3 = st.columns(3)
    col1.metric("Tickets Únicos", f"{df_filtrado['InvoiceNo'].nunique():,}")
    col2.metric("Líneas de Artículo", f"{len(df_filtrado):,}")
    col3.metric("Clientes Identificados", f"{df_filtrado['CustomerID'].nunique():,}")

    st.markdown("---")
    st.subheader("🧠 Motor de Minería: Extracción de Reglas de Asociación")

    # 4. Transformación a Matriz Transaccional
    with st.spinner("Construyendo matriz transaccional en RAM..."):
        basket = (df_filtrado.groupby(['InvoiceNo', 'Description'])['Quantity']
                  .sum().unstack().reset_index().fillna(0).set_index('InvoiceNo'))
        basket_sets = basket > 0

    # 5. Ejecución del Algoritmo y Visualización
    with st.spinner("Ejecutando algoritmo de Machine Learning..."):
        itemsets_frecuentes = apriori(basket_sets, min_support=soporte_minimo, use_colnames=True)
        
        if itemsets_frecuentes.empty:
            st.warning("No se encontraron patrones con el Soporte actual. Prueba a bajar el 'Soporte Mínimo'.")
        else:
            reglas = association_rules(itemsets_frecuentes, metric="lift", min_threshold=lift_minimo)
            
            if reglas.empty:
                st.warning("Hay productos frecuentes, pero ninguno genera reglas cruzadas fuertes. Baja el 'Lift Mínimo'.")
            else:
                reglas['antecedents'] = reglas['antecedents'].apply(lambda x: ', '.join(list(x)))
                reglas['consequents'] = reglas['consequents'].apply(lambda x: ', '.join(list(x)))
                
                reglas_mostrar = reglas[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
                reglas_mostrar.columns = ['Si compran esto...', '...También compran esto', 'Soporte', 'Confianza', 'Lift (Fuerza)']
                reglas_mostrar = reglas_mostrar.sort_values('Lift (Fuerza)', ascending=False)
                
                st.success(f"¡Análisis completado! Se han descubierto {len(reglas)} reglas de asociación fuertes en {pais_seleccionado}.")
                
                # Desplegar la tabla
                st.dataframe(reglas_mostrar, use_container_width=True)

                st.markdown("---")
                st.subheader("📈 Mapa Estratégico de Oportunidades de Cross-Selling")
                
                # Gráfico interactivo
                grafico = alt.Chart(reglas_mostrar).mark_circle().encode(
                    x=alt.X('Soporte', title='Soporte (Popularidad)'),
                    y=alt.Y('Confianza', title='Confianza (Probabilidad)'),
                    size=alt.Size('Lift (Fuerza)', title='Lift', scale=alt.Scale(range=[100, 1000])),
                    color=alt.Color('Lift (Fuerza)', scale=alt.Scale(scheme='viridis')),
                    tooltip=['Si compran esto...', '...También compran esto', 'Lift (Fuerza)']
                ).interactive().properties(
                    height=450
                )
                
                st.altair_chart(grafico, use_container_width=True)

except Exception as e:
    st.error(f"Error en la ejecución: {e}")
