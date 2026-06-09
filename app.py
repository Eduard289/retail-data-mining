import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import altair as alt

# Configuración de la página
st.set_page_config(page_title="Analítica Retail", layout="wide")

# Cabecera principal
st.title(" Analítica retail: minería de datos")
st.markdown("---")

# Ventana emergente (Popup) con la Guía Metodológica, Explicación y Utilidad Retail
@st.dialog(" Guía Metodológica y Análisis Estratégico", width="large")
def mostrar_guia():
    st.markdown("""
    ###  Operativa Retail: ¿Para qué sirve implementar esta herramienta?
    Implementar este motor analítico permite a un comercio pasar de la intuición a la precisión basada en datos. Transforma miles de tickets de caja en decisiones operativas de alto impacto:
    * **Visual Merchandising Inteligente:** Diseñar el *layout* de la tienda física colocando productos con alta fuerza de asociación (Lift) juntos o en el camino natural del cliente para forzar la compra por impulso.
    * **Estrategias de Precios (Bundles):** Crear packs promocionales hiperrentables uniendo un producto gancho de alta frecuencia con su consecuente de alto margen.
    * **Personalización E-commerce:** Alimentar directamente los motores de recomendación web y configurar los pop-ups de *"Los clientes que compraron esto, también se llevaron..."*.
    * **Incremento del UPT (Unidades por Ticket):** El resultado inmediato de estas sinergias es una subida drástica de los artículos vendidos por transacción y del ticket medio.

    ---

    ### 1️ La Tabla: El Descubrimiento de Patrones
    Esta tabla refleja el resultado del algoritmo Apriori. Tras procesar miles de transacciones del país seleccionado, el modelo extrae relaciones ocultas entre productos que el ojo humano jamás detectaría a simple vista en un Excel

    Nos fijamos en las líneas que captura del modelo. Es pura lógica de consumo detectada matemáticamente:
    * **Regla 112 & 115 (Ejemplo España):** Si compran la "Cocina de juguete Poppy" (*Poppy's Playhouse Kitchen*), automáticamente compran la "Habitación de juguete Poppy" (*Bedroom*).
    * **Regla 154 & 155:** Quien compra las velas de celebración para niñas, también se lleva el set de 10 velas de lunares rosas.

    ---

    ### 2️ El Gráfico de Dispersión: El Mapa Estratégico
    La tabla está optimizaa para leer datos exactos. El gráfico es el panel de mando que le presentaríamos a un Director General. Aquí cruzaremos las tres métricas matemáticas clave de la minería de datos:

    * **Eje X (Soporte - Popularidad):** Nos dice de izquierda a derecha cuánto se repite esa combinación en la tienda. Cuanto más a la derecha esté un punto, más habitual es esa cesta de la compra.
    * **Eje Y (Confianza - Probabilidad):** Nos dice de abajo a arriba la probabilidad de acierto. Si un punto está en el 0.8, significa que el 80% de las veces que un cliente coge el Producto A, acaba llevándose el Producto B.
    * **El Tamaño y Color de la Burbuja (LIFT - Fuerza y €):** Es la métrica reina. Un Lift alto (burbujas muy grandes y verdes) indica que la venta del Producto B está impulsada fuertemente por la del Producto A. No es casualidad que se vendan juntos; están íntimamente ligados en la mente del consumidor.

     **3 Cómo leer el gráfico para tomar decisiones:**
    * **El "Santo Grial" (Burbujas grandes, arriba y a la derecha):** Son productos súper populares, con alta probabilidad de compra conjunta y una sinergia altísima. Aquí es donde debemos invertir  presupuesto de marketing cruzado.
    * **Burbujas grandes arriba, pero a la izquierda:** Son productos de nicho (se venden poco volumen total), pero cuando un cliente los busca, la venta cruzada es casi segura. Ideal para promociones exclusivas.
    * **Burbujas pequeñas y oscuras abajo:** Son coincidencias. Productos que a veces caen en el mismo ticket por puro azar. No debemos perder tiempo ni espacio en tienda intentando juntarlos.

    **En resumen:** Esta herramienta de mineria  no solo dice "qué se vende", sino que gracias a este gráfico podemos priorizar dónde poner nuestros esfuerzos operativos para maximizar el margen de beneficio.

    ---
    ### 4 ¿Qué significan las columnas de la tabla?
    * **Soporte:** Mide la frecuencia. Es el porcentaje de veces que estos productos aparecen juntos sobre el total de tickets del país. Sirve para filtrar coincidencias anecdóticas y centrarse en lo que realmente se vende a diario.
    * **Confianza:** Mide la probabilidad. Si marca 0.85, significa que el 85% de los clientes que meten el primer producto en la cesta, acaban comprando también el segundo.
    * **Lift (Fuerza):** Mide el "efecto imán" o sinergia real. Un Lift de 1.0 es puro azar. Un Lift de 3.0 significa que es 3 veces más probable que estos artículos se vendan juntos que por separado. Es la métrica definitiva para armar campañas de *cross-selling*.
    * **Oportunidad (€):** Métrica financiera exclusiva de esta herramienta. Calcula el impacto económico latente multiplicando las veces que se espera que ocurra esta regla por el precio de los artículos sugeridos. Permite priorizar acciones por rentabilidad rea

    ### 5 Conceptos Matemáticos Clave
    * **Soporte:** Indica el porcentaje de tickets totales de la tienda que contienen la combinación analizada.
    * **Confianza:** Es la probabilidad de acierto de la regla.
    * **Lift:** Mide el "efecto imán". Un Lift de 3.0 significa que la compra del primer artículo multiplica por tres la probabilidad de que se lleve el segundo.
    * **Oportunidad Financiera (€):** Estima el impacto económico latente multiplicando la frecuencia proyectada por el precio medio.

    ---

    ### 6 Fuente de los Datos y Contenido
    * **Cita Oficial:** *Online Retail Data Set*, proporcionado por el **UCI Machine Learning Repository** (University of California, Irvine).
    * **Contenido:** Histórico real de **541,909 transacciones** comerciales de una empresa minorista internacional. Abarca un ejercicio fiscal completo (01/12/2010 - 09/12/2011), incluyendo campos críticos como SKU, precio, cantidad y país de facturación.
    """)

# 1. Motor de Ingesta en RAM con mensaje personalizado
@st.cache_data(show_spinner="Conectando con la base de datos real de la UCI, debido a la cantidad de datos de carga puede tardar unos minutos...")
def cargar_datos():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    df = pd.read_excel(url)
    df = df.dropna(subset=['InvoiceNo'])
    df = df[~df['InvoiceNo'].astype(str).str.contains('C')]
    return df

try:
    df_retail = cargar_datos()
    
    # 2. Sidebar: Configuración, Buscador y Botón de Información
    st.sidebar.header("⚙️ Configuración del Análisis")
    
    paises = sorted(df_retail['Country'].unique())
    indice_defecto = paises.index('Spain') if 'Spain' in paises else 0
    pais_seleccionado = st.sidebar.selectbox("Selecciona el Mercado:", paises, index=indice_defecto)

    st.sidebar.markdown("---")
    
    # Botón interactivo para desplegar el popup con la mega-guía
    if st.sidebar.button("📚 Ver Conceptos y Utilidad", use_container_width=True):
        mostrar_guia()

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

    # Extraemos diccionario de precios medios
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
                def estimar_impacto(fila):
                    consecuentes = fila['...También compran esto'].split(', ')
                    precio_total = sum(precios_dict.get(item, 0) for item in consecuentes)
                    return round((fila['Soporte'] * tickets_totales) * precio_total, 2)
                
                reglas_mostrar['Oportunidad (€)'] = reglas_mostrar.apply(estimar_impacto, axis=1)
                reglas_mostrar = reglas_mostrar.sort_values('Oportunidad (€)', ascending=False)

                # --- MEJORA SEMÁNTICA (BUSCADOR) ---
                if termino_busqueda:
                    mask = reglas_mostrar['Si compran esto...'].str.contains(termino_busqueda) | reglas_mostrar['...También compran esto'].str.contains(termino_busqueda)
                    reglas_mostrar = reglas_mostrar[mask]

                if reglas_mostrar.empty:
                    st.info(f"No se han encontrado reglas que contengan la palabra '{termino_busqueda}'.")
                else:
                    st.success(f"¡Análisis completado! Se muestran {len(reglas_mostrar)} reglas de alto impacto.")
                    st.dataframe(reglas_mostrar, width='stretch')

                    st.markdown("---")
                    st.subheader("📈 Mapa Estratégico de Oportunidades")
                    
                    grafico = alt.Chart(reglas_mostrar).mark_circle().encode(
                        x=alt.X('Soporte', title='Soporte (Popularidad)'),
                        y=alt.Y('Confianza', title='Confianza (Probabilidad)'),
                        size=alt.Size('Lift (Fuerza)', title='Lift', scale=alt.Scale(range=[100, 1000])),
                        color=alt.Color('Oportunidad (€)', scale=alt.Scale(scheme='greens')),
                        tooltip=['Si compran esto...', '...También compran esto', 'Lift (Fuerza)', 'Oportunidad (€)']
                    ).interactive().properties(
                        height=450
                    )
                    
                    st.altair_chart(grafico, width='stretch')

    # Footer de firma actualizado (Se muestra siempre abajo del todo)
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #7f8c8d; font-size: 14px;'>Diseñado y desarrollado por Jose Luis Asenjo</p>", unsafe_allow_html=True)
            # Footer de firma
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #7f8c8d; font-size: 14px;'>Diseñado y desarrollado por Jose Luis Asenjo</p>", unsafe_allow_html=True)
except Exception as e:
    st.error(f"Error en la ejecución: {e}")
