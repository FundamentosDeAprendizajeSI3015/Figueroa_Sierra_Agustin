# 🤖 Fundamentos de Aprendizaje Automático – SI3015
**Estudiante:** Agustín Figueroa Sierra

---

## 📋 Descripción del Repositorio

Este repositorio contiene las actividades desarrolladas durante el curso **Fundamentos de Aprendizaje Automático**, organizadas por módulos de trabajo práctico.

El enfoque principal del curso fue la **exploración de datos, análisis y preprocesamiento**, entendiendo que la calidad de la representación de datos (X, y) es un paso crítico antes de entrenar modelos de Machine Learning.

### 🎯 Objetivo del Repositorio

Documentar el proceso completo de preparación de datos para Machine Learning, incluyendo:

- ✅ Exploración inicial de datasets (EDA)
- ✅ Evaluación de calidad de datos
- ✅ Estadística descriptiva
- ✅ Detección de anomalías (outliers)
- ✅ Visualización de patrones
- ✅ Transformación y codificación de variables
- ✅ Escalado de características
- ✅ Construcción de datasets listos para modelado

El repositorio refleja el flujo de trabajo real seguido durante el curso, priorizando la comprensión del pipeline de datos antes del desarrollo de modelos.

---

## 📁 Estructura del Repositorio

### 📊 `iris_analisis/`
**Dataset:** Iris Dataset (scikit-learn)

**Objetivo**
Introducir el flujo básico de Machine Learning, desde la recolección de datos hasta la evaluación inicial en un contexto de aprendizaje supervisado.

**Actividades Realizadas**
- Carga de dataset y construcción de DataFrame
- Exploración inicial de datos (EDA)
- Análisis de distribución de clases
- Estadística descriptiva
- Visualización de características y análisis de correlación
- División Train/Test
- Escalado de características usando StandardScaler
- Entrenamiento y evaluación de modelos

**Conceptos Cubiertos**
- Representación X, y
- Escalado de características
- Análisis exploratorio de datos
- Fundamentos de clasificación supervisada

---

### 💰 `Fintech/`
**Dataset:** Fintech dataset

**Objetivo**
Construir un pipeline de exploración y preprocesamiento de datos, enfocándose en entender que el rendimiento del modelo depende fuertemente de la calidad del dataset preparado.

**Actividades Realizadas**
- Carga y exploración de dataset
- Limpieza y transformación de datos
- Análisis y visualización de características
- Construcción del pipeline de preprocesamiento

**Conceptos Cubiertos**
- Evaluación de calidad de datos
- Flujos de trabajo en limpieza de datos
- Transformación de características
- Separación entre preprocesamiento y modelado

---

### 🚢 `Clase4-Titanic/`
**Dataset:** Titanic Dataset

**Objetivo**
Desarrollar un flujo completo de limpieza y transformación de datos utilizando un dataset con problemas reales de calidad de datos.

**Actividades Realizadas**
- Limpieza y normalización de datos
- Normalización de nombres de columnas
- Conversión de tipos de datos
- Manejo de valores faltantes
- Estadística descriptiva (media, mediana, moda, desviación estándar, varianza, IQR)
- Detección de outliers
- Transformación de características (Label Encoding, One-Hot Encoding)
- Análisis de correlación
- Escalado de características (MinMaxScaler, StandardScaler)

**Conceptos Cubiertos**
- Desafíos de calidad en datasets reales
- Ingeniería de características básica
- Reducción de ruido estadístico
- Preparación de datos tabulares para aprendizaje supervisado

---

### 🚢 `Clase-5-Regresion-Titanic/`
**Dataset:** Titanic Dataset

**Objetivo**
Implementar modelos de Regresión Lineal y Logística utilizando Pipelines de Scikit-Learn, enfocándose en la automatización del preprocesamiento y la optimización de hiperparámetros.

**Actividades Realizadas**
- Ingeniería de características y limpieza avanzada de datos.
- Implementación de **Regresión Lineal (Ridge y Lasso)** para predicción de edad.
- Implementación de **Regresión Logística** para clasificación de supervivencia.
- Optimización de modelos mediante `RandomizedSearchCV` y validación cruzada.
- **Análisis de Importancia de Características**: Visualización del impacto de cada variable en el modelo.
- **Curva ROC y AUC**: Evaluación del rendimiento del clasificador a diferentes umbrales.

**Conceptos Cubiertos**
- Automatización mediante `Pipeline` y `ColumnTransformer`.
- Regularización estructural (Ridge vs. Lasso).
- Evaluación avanzada de modelos de clasificación y regresión.
- Explicabilidad del modelo (Feature Importance).

---

### 🌦️ `WeatherChart-Proyecto1-ML/`
**Proyecto:** Análisis y Predicción de Datos Meteorológicos

**Objetivo**
Demostrar el ciclo completo de Machine Learning a través de un caso de uso educativo enfocado en análisis de datos climáticos.

**Características Clave**
- Análisis exploratorio de datos meteorológicos
- Ingeniería de características: patrones climáticos, tendencias temporales
- Selección de candidatos a modelos: Regresión Lineal, Árboles de Decisión, Bosques Aleatorios
- Evaluación: MSE, MAE, R², Validación Cruzada
- Visualización de predicciones y análisis de residuos

**Principio Central**
Los modelos aprenden patrones de datos históricos y no toman decisiones autónomas. Su función es asistir en el proceso de toma de decisiones humanas. La responsabilidad final siempre recae en profesionales capacitados.

---

## 🛠️ Tecnologías Utilizadas

| Tecnología | Descripción |
|-----------|-------------|
| **Python** | Lenguaje de programación principal |
| **NumPy** | Computación numérica y manipulación de arrays |
| **Pandas** | Manipulación y análisis de datos tabulares |
| **Matplotlib** | Visualización estática de datos |
| **Seaborn** | Visualización estadística avanzada |
| **Scikit-learn** | Algoritmos de Machine Learning y preprocesamiento |

---

## 💡 Conclusiones del Curso

1. **Calidad sobre Cantidad** - El rendimiento del modelo depende fuertemente de la calidad del preprocesamiento de datos.

2. **Exploración Visual** - La exploración visual ayuda a entender la estructura geométrica de los datasets.

3. **Transformaciones Esenciales** - El escalado y transformación de características son pasos esenciales en flujos de trabajo de ML.

4. **Separación de Responsabilidades** - Separar preprocesamiento del modelado mejora la reproducibilidad.

5. **Realidad vs. Teoría** - Los datasets reales requieren significativamente más limpieza que los datasets académicos.

6. **Asistencia, No Automatización** - Los modelos de Machine Learning deben asistir en decisiones humanas, no reemplazarlas.

---

**Última actualización:** Febrero 17, 2026
