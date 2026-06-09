import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import altair as alt
from st_aggrid import AgGrid, GridOptionsBuilder

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
    
    # 2. Sidebar: Configuración y Buscador
    st.sidebar.header("⚙️ Configuración del Análisis")
    
    paises = sorted(df_retail['Country'].unique())
    indice_defecto = paises.index('Spain') if 'Spain' in paises else 0
    pais_seleccionado = st.sidebar.selectbox("Selecciona el Mercado:", paises, index=indice_defecto)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Filtro de Producto")
    termino_busqueda = st.sidebar.text_input("Buscar familia (ej. BAG, HEART, VINTAGE):", "").upper()

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎛️ Parámetros del Algoritmo")
    soporte_minimo = st.sidebar.slider("Soporte Mínimo (%)", min_value=1, max_value=20, value=5, step=1) / 100.0
    lift_minimo = st.sidebar.slider("Lift Mínimo", min_value=1.0, max_value=10.0, value=1.0, step=0.5)

    # 3. Filtrado reactivo
    df_filtrado = df_retail[df_retail['Country'] == pais_seleccionado]
    tickets_totales = df_filtrado['InvoiceNo'].nunique()

    # Extraemos un diccionario de precios medios para calcular el impacto financiero luego
    precios_dict = df_filtrado.groupby('Description')['UnitPrice'].mean().to_dict()

    # Visualización de métricas
    col1, col2, col3 = st.columns(3)
    col1.metric("Tickets Únicos", f"{tickets_totales:,}")
    col2.metric("Líneas de Artículo", f"{len(df_filtrado):,}")
    col3.metric("Clientes Identificados", f"{df_filtrado['CustomerID'].nunique():,}")

    st.markdown("---")
    st.subheader("🧠 Motor de Minería y Oportunidad Financiera")

    # 4. Transformación a Matriz
    with st.spinner("Construyendo matriz transaccional en RAM..."):
        basket = (df_filtrado.groupby(['InvoiceNo', 'Description'])['Quantity']
                  .sum().unstack().reset_index().fillna(0).set_index('InvoiceNo'))
        basket_sets = basket > 0

    # 5. Ejecución del Algoritmo
    with st.spinner("Calculando reglas e impacto de negocio..."):
        itemsets_frecuentes = apriori(basket_sets, min_support=soporte_minimo, use_colnames=True)
        
        if itemsets_frecuentes.empty:
            st.warning("No se encontraron patrones. Prueba a bajar el 'Soporte Mínimo'.")
        else:
            reglas = association_rules(itemsets_frecuentes, metric="lift", min_threshold=lift_minimo)
            
            if reglas.empty:
                st.warning("No hay reglas fuertes. Baja el 'Lift Mínimo'.")
            else:
                reglas['antecedents'] = reglas['antecedents'].apply(lambda x: ', '.join(list(x)))
                reglas['consequents'] = reglas['consequents'].apply(lambda x: ', '.join(list(x)))
                
                reglas_mostrar = reglas[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
                reglas_mostrar.columns = ['Si compran esto...', '...También compran esto', 'Soporte', 'Confianza', 'Lift (Fuerza)']
                
                # --- MEJORA FINANCIERA ---
                # Calculamos el dinero que mueve el producto consecuente cuando se activa la regla
                def estimar_impacto(fila):
                    consecuentes = fila['...También compran esto'].split(', ')
                    precio_total = sum(precios_dict.get(item, 0) for item in consecuentes)
                    # (Probabilidad de que ocurra en el total de tickets) * Precio de los artículos sugeridos
                    return round((fila['Soporte'] * tickets_totales) * precio_total, 2)
                
                reglas_mostrar['Oportunidad (€)'] = reglas_mostrar.apply(estimar_impacto, axis=1)
                
                # Ordenar por el dinero que generan en lugar de solo por el Lift
                reglas_mostrar = reglas_mostrar.sort_values('Oportunidad (€)', ascending=False)

                # --- MEJORA SEMÁNTICA (BUSCADOR) ---
                if termino_busqueda:
                    mask = reglas_mostrar['Si compran esto...'].str.contains(termino_busqueda) | reglas_mostrar['...También compran esto'].str.contains(termino_busqueda)
                    reglas_mostrar = reglas_mostrar[mask]

                if reglas_mostrar.empty:
                    st.info(f"No se han encontrado reglas que contengan la palabra '{termino_busqueda}'.")
                else:
                    st.success(f"¡Análisis completado! Se muestran {len(reglas_mostrar)} reglas de alto impacto.")
                    
                    # --- MEJORA VISUAL (AG-GRID JS) ---
                    gb = GridOptionsBuilder.from_dataframe(reglas_mostrar)
                    gb.configure_pagination(paginationAutoPageSize=True) # Paginación automática
                    gb.configure_side_bar() # Menú lateral de Excel para agrupar/filtrar
                    gridOptions = gb.build()

                    AgGrid(
                        reglas_mostrar,
                        gridOptions=gridOptions,
                        enable_enterprise_modules=True,
                        fit_columns_on_grid_load=True,
                        theme='balham' # Tema profesional y limpio
                    )

                    st.markdown("---")
                    st.subheader("📈 Mapa Estratégico de Oportunidades")
                    
                    # Gráfico interactivo adaptado para mostrar la oportunidad en el Tooltip
                    grafico = alt.Chart(reglas_mostrar).mark_circle().encode(
                        x=alt.X('Soporte', title='Soporte (Popularidad)'),
                        y=alt.Y('Confianza', title='Confianza (Probabilidad)'),
                        size=alt.Size('Lift (Fuerza)', title='Lift', scale=alt.Scale(range=[100, 1000])),
                        color=alt.Color('Oportunidad (€)', scale=alt.Scale(scheme='greens')), # Burbujas más oscuras = Más dinero
                        tooltip=['Si compran esto...', '...También compran esto', 'Lift (Fuerza)', 'Oportunidad (€)']
                    ).interactive().properties(
                        height=450
                    )
                    
                    st.altair_chart(grafico, width='stretch')

except Exception as e:
    st.error(f"Error en la ejecución: {e}")
