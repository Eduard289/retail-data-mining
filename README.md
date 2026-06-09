# Analitica Retail: Segmentacion Internacional y Motor de Mineria de Datos

## Descripcion General del Proyecto
Este proyecto implementa una aplicacion SaaS (Software as a Service) orientada a la analitica avanzada en el sector retail. Su nucleo es un motor de mineria de datos basado en el algoritmo Apriori, diseñado para ejecutar analisis de la cesta de la compra (Market Basket Analysis) sobre transacciones internacionales. La herramienta transforma datos transaccionales brutos en decisiones estrategicas de negocio, identificando patrones ocultos de compra conjunta y cuantificando su impacto financiero directo.

## Arquitectura e Infraestructura
El sistema esta diseñado bajo un paradigma de computacion ligera y procesamiento volatil:
* Zero-Disk Storage: La infraestructura prioriza la privacidad y la velocidad procesando el 100% de las operaciones en la memoria RAM del servidor. No se escriben archivos temporales ni persistentes en el disco duro.
* Ingestion Dinamica en Memoria: El framework implementa funciones de cacheo en memoria para descargar, limpiar y mantener el dataset activo unicamente durante el ciclo de vida de la ejecucion, reduciendo la latencia tras el primer arranque.
* Despliegue Cloud: Desarrollado integramente en Python y desplegado mediante la plataforma Streamlit Cloud.

## Motor de Mineria de Datos (Machine Learning)
El nucleo analitico utiliza la libreria `mlxtend` para extraer reglas de asociacion. El proceso consta de dos fases principales:

1. Transformacion Matricial: El script ingiere un log transaccional y lo pivota dinamicamente en la RAM para construir una matriz booleana multidimensional, donde las filas representan identificadores de tickets unicos y las columnas representan el catalogo completo de unidades de mantenimiento de existencias (SKU).
2. Algoritmo Apriori: Se escanea la matriz para identificar itemsets frecuentes y generar reglas de asociacion probabilisticas, limitadas por umbrales configurables por el usuario.

### Metricas Centrales Evaluadas
* Soporte (Support): Frecuencia relativa de una combinacion de productos respecto al total de transacciones. Actua como filtro primario para descartar ruido estadistico.
* Confianza (Confidence): Probabilidad condicional de que un cliente adquiera el consecuente habiendo adquirido el antecedente.
* Lift (Fuerza de Asociacion): Ratio que mide el incremento en la probabilidad de compra conjunta frente a la probabilidad esperada bajo condiciones de independencia. Un Lift > 1.0 confirma una sinergia comercial real.
* Oportunidad Financiera (EUR): Metrica calculada en tiempo real que cruza el soporte proyectado con el diccionario de precios unitarios (UnitPrice) extraido del mercado auditado, traduciendo probabilidades en retorno de inversion (ROI) latente.

## Capacidades Operativas
La herramienta esta concebida para la toma de decisiones ejecutivas en departamentos de Category Management y Visual Merchandising:
* Segmentacion Geografica: Capacidad de aislar patrones de consumo especificos por mercado de facturacion.
* Perfilado Semantico: Motor de busqueda integrado para filtrar reglas de asociacion por familias de productos.
* Visualizacion Estrategica: Implementacion de graficos de dispersion interactivos mediante la libreria `altair`, mapeando simultaneamente Popularidad (Eje X), Probabilidad (Eje Y) y Fuerza Comercial / Impacto Economico (Tamaño y Color).

## Origen de los Datos
El modelo se alimenta del "Online Retail Data Set", un conjunto de datos estandarizado proporcionado por el UCI Machine Learning Repository. 
* Volumen: 541.909 registros transaccionales historicos.
* Periodo: Ejercicio fiscal comprendido entre el 01/12/2010 y el 09/12/2011.
* Entidad: Comercio minorista internacional online especializado en articulos de regalo.

## Requisitos y Dependencias
El entorno virtual requiere la instalacion de los siguientes paquetes definidos en `requirements.txt`:
* streamlit
* pandas
* openpyxl
* mlxtend
* altair

## Instalacion y Ejecucion Local
1. Clonar el repositorio en el entorno de trabajo local.
2. Crear un entorno virtual e instalar las dependencias: `pip install -r requirements.txt`
3. Ejecutar el servidor ligero de Streamlit: `streamlit run app.py`

---
Diseñado y desarrollado por Jose Luis Asenjo.
