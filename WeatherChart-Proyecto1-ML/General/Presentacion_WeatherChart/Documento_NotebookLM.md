# Documento Base para NotebookLM: WeatherChart - Análisis y Re-evaluación de Modelos Predictivos

## 1. Contexto del Proyecto: ¿Qué es WeatherChart?
**WeatherChart** nace de una premisa ambiciosa: ¿están los géneros musicales que consumen las personas relacionados intrínsecamente con las condiciones socioeconómicas, climáticas y geográficas de su entorno? 

Para responder a esta pregunta, el proyecto ensambló de manera pionera un ecosistema de datos cruzando múltiples fuentes de información:
1. **Top Cantantes de Spotify (2023)**: Información sobre las pistas de audio, su "bailabilidad" (danceability), energía, tempo, "acústica" y género musical principal asignado.
2. **Global Weather (1970 - 2021)**: Factores clímaticos estandarizados en base a temperaturas promedio por países y continentes a lo largo de diversas décadas.
3. **Censos Socioeconómicos Mundiales**: Incorporando el Producto Interno Bruto (GDP) per cápita, tasas de desempleo, densidad de población, y latitud/longitud global.

### El Desafío Inicial
Se procesó un volumen masivo de datos mediante un pipeline automatizado de 13 fases estructurales reduciendo la data de casi 18 millones de tuplas a un conjunto consistente de **~1.6 millones de muestras de entrenamiento altamente representativas** divididas en **86 clases de géneros musicales**.

Usando un diseño base de **XGBoost Classifier**, el modelo primario logró predecir exitosamente estas clases a un excepcional **30.0%** de precisión sobre un problema complejo (predecir una entre 86 categorías sin dependencia directa determinista es una tarea extremadamente difícil en Machine Learning).

---

## 2. Definición del Problema Específico: La Hipótesis del Etiquetado Ruidoso
Si bien XGBoost logró modelar no linealmente parte de los datos, surgió una nueva directriz crucial que guiaba este segmento de la investigación: 
> *La posibilidad matemática e inferencial de que el **~30% o más de las etiquetas originales (géneros) en el Dataset estuvieran mal asignadas** o fueran excesivamente ruidosas.* 

La música es inherentemente ambigua, y tracks catalogados bajo "Pop", "Indie", o "Alt-Rock" a menudo comparten la misma arquitectura musical intrínseca. Para demostrar que existe una estructura matemática de agrupamiento "natural", la misión migró netamente hacia la evaluación mediante técnicas de **Aprendizaje No Supervisado**. 

El objetivo primordial a demostrar en la presentación fue:
1. Validar la existencia de cúmulos naturales a través de algoritmos puramente predictivos (Clusters).
2. Usar un algoritmo unificador masivo para reescribir un nuevo sistema de predicción (eliminar las etiquetas ruidosas y reemplazarlas por agrupamientos consolidados según la densidad de sus características base).
3. Contrastar ambas metodologías para ver si los "nuevos" datos filtrados incrementaban considerablemente la certeza algorítmica.

---

## 3. Metodología de Implementación Unificada (Pipeline Unsupervised Clustering)

Se ejecutó un script robusto estructurado (*16-unsupervised_clustering.py*) sobre un muestreo estratificado proporcional para evitar cuellos de botella algorítmicos.

Fueron integrados exhaustivamente cinco métodos de *Clustering* con evaluaciones de calidad computacional:
1. **K-Means ($k=86$)**: Basado fundamentalmente en la distancia intra-varianzas sobre centroides espaciales estándar.
2. **Fuzzy C-Means**: Se desarrolló meticulosamente el algoritmo matemático (Bezdek, 1981) nativo ante la dependencia en Scikit, el cual no asume que un punto sea de "Un Clase C" dura, sino otorgando márgenes de membrecía blanda controlada donde cada track ostenta un porcentaje pertenencia a diferentes agrupamientos (Factor Fuzziness = 2).
3. **Subtractive Clustering**: Desarrollado algorítmicamente desde cero, basado en densidades absolutas (Chiu, 1994) calculando qué puntos poseen el mayor "potencial" atrayente del resto. (Estimó casi $200$ clusters compactos y pequeños).
4. **DBSCAN**: Clusterización por Densidad Automatizada estimando un `eps` con el 90mo percentil de distancias Nearest-Neighbors. Pudo identificar que casi un `6%` de toda nuestra data de entrenamiento eran en realidad elementos anómalos o "Noise", evidenciando parte de las sospechas del proyecto base.
5. **Familia Clustering Jerárquico (Agglomerative)**: Árbol relacional Ward para entender interacciones topológicas.

**Proyecciones de Alta Dimensión**
Las predicciones espaciales fueron reducidas mediante:
*   **PCA 2D y 3D (Análisis de Componentes Principales)** para identificar ejes lineales ortogonales masivos.
*   **UMAP (Uniform Manifold Mapping)**, un método mucho más potente e intuitivo para visualizar cómo ciertas relaciones multi-paramétricas mantienen una estructura global altamente contigua e identificable (los resultados proyectaron claros "nodos y ramificaciones" distintivas ocultas, antes imposibles de ver en métricas limitadas).

---

## 4. Re-evaluación del Conjunto de Entrenamiento (Re-Labeling)

Para corregir la hipótesis inicial, se forzó la creación de un nuevo conjunto de prueba sobre los $1.6$ millones de datos (las restricciones de memoria computacionales en el set completo requirieron adaptar integraciones con `MiniBatchKMeans`).

**Mecanismo de "Mayoría de Votos" (Majority Voting Mechanism):**
Bajo los $86$ clusters creados orgánicamente por proximidad numérica, se cruzaron las etiquetas verdaderas contra los resultados asignados matemáticamente, para descubrir qué "género musical" predominaba estadísticamente en cada uno.

**Resultados del Re-etiquetamiento:**
*   Se halló que el dataset inicial tenía efectivamente muchísimo ruido. En múltiples géneros el volumen estadístico no se asemejaba a un comportamiento lineal predecible. 
*   Erigida esta base, se reasignaron por mayoría de etiqueta al **61.87% del Dataset global** (Modificando **985,977 muestras** a géneros estadísticos verdaderos correspondientes con la característica y perfil original de los agrupamientos).
*   *Nota Relevante:* Géneros ambiguos como "Reggae", "Dance", o "Edm" arrojaron un solapamiento masivo y un cruce tremendo. La predicción de los agrupamientos ajustó su clase predicha a casi el 95% de sus registros. En contrapartida, el Pop demostró estar consistentemente mejor estandarizado y retuvo la base originaria inalterada en casi su plena completitud original (>500.000 records). 

---

## 5. El Contraste Definitivo Supervisado

Al poseer ahora dos arquitecturas de "verdad" operacionales originarias: 
- El conjunto antiguo lleno de ruido (Originales).
- El nuevo conjunto relabelado y densamente puro (Re-evaluado).

Se desplegó la ejecución comparativa en paralelo (*17-supervised_comparison.py*) donde todo debía culminar en una comparación entre Modelos Supervisados lineales base que, anteriormente, tenían una capacidad de retención limitadísima. 

Se entrenaron $3$ Modelos Estadísticos contra ambos formatos bajo la métrica del conjunto de testeo original que XGBoost retuvo de antemano como Benchmark (baseline `30%` Accuracy):
- Modelos testeados: **Decision Tree**, **Logistic Regression**, y un **Ridge Classifier** (regresión lineal adaptada a clasificaciones de contención simple).

---

## 6. Las Conclusiones Fundamentales de la Experimentación

Las conclusiones obtenidas y demostradas afirman con éxito la tesis inicial planeada por los instructores: Limpiar la categorización y buscar la organización latente dispara el Machine Learning en un entorno donde la ambigüedad original predominaba. Todo sin utilizar modelamiento no lineal sofisticado. 

Resultados comparativos notables en base de pruebas:
- **Decision Tree (Original)** ostentaba un bajísimo $3.06\%$ de asertividad global. Con el reetiquetado de las métricas logró predecir exitosamente y subió su potencial escalando drásticamente hacia los márgenes operacionales con un increíble **$24.90\%$ Accuracy absoluto** (Un incremento de **~21.84%** directos para una estructura puramente basada en la ganancia de Información Gini).
- **Ridge Classifier** obtuvo un modesto $2.68\%$ y con limpieza de clases superó el $11\%$.
- **Logistic Regression (Solucionador Lineal Multinomial SAGA)** demostró retención base elevándose de un ya prometedor $18.9\%$ hasta un **$25.2\%$**, posicionado excepcionalmente cerca del complejo entrenamiento paramétrico que XGBoost necesitaba originalmente para inferir su baseline estelar de $30.0\%$.

### Impacto de este Modelo para los Instructores y Auditores:
*   Muestra exitosamente que la calidad de los datos ("Garbage in, Garbage out") es más que fundacional. 
*   Comprueba a través de clustering orgánico y densidades DBSCAN (como UMAP proyectado espacial), un hecho incontestable empírico: Un 61.87% del volumen del Dataset de música (El doble a lo propuesto en el 30% inicial postulado y subvalorado) estaba estadísticamente en contra de sí mismo.
*   Enseña un excelente control inter y trans-dimensional implementado exitosamente. Demuestra cómo algoritmos muy sencillos (Decision Trees o simple Logistic Regression) entrenados con alta certeza consiguen alcanzar los benchmarks estelares de tecnologías computacionalmente más gravosas, sencillamente ajustando las deficiencias estadísticas con Inteligencia No Supervisada primero.
