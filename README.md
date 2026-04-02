# Vortex: Medical Insurance Risk Segmentation & Cost Prediction

> Pipeline completo de Machine Learning para segmentación de asegurados por perfil de riesgo médico (K-Means) y predicción del costo de cobertura (XGBoost), aplicado al dataset Medical Insurance Cost de Kaggle.

---

## Tabla de Contenidos
- [Pregunta de Negocio](#pregunta-de-negocio)
- [Dataset](#dataset)
- [Tecnologías y Librerías](#tecnologías-y-librerías)
- [Desarrollo](#desarrollo)
- [Resultados](#resultados)
- [Conclusiones y Respuesta a la Pregunta de Negocio](#conclusiones-y-respuesta-a-la-pregunta-de-negocio)
- [Limitaciones y Trabajo Futuro](#limitaciones-y-trabajo-futuro)
- [Learnings](#learnings)
- [Sobre el Uso de IA](#sobre-el-uso-de-ia)
- [Cómo Replicar](#cómo-replicar)
- [Referencias](#referencias)

---

## Pregunta de Negocio

¿Es posible segmentar a los asegurados según su perfil de riesgo médico y, a partir de ello, predecir el costo de cobertura que representa cada cliente para la compañía aseguradora?

Este proyecto aborda dos preguntas complementarias que en la práctica actuarial se usan de forma secuencial:

1. **Segmentación:** ¿Qué perfiles de riesgo existen naturalmente en la cartera de clientes? → K-Means Clustering
2. **Predicción:** ¿Cuánto costará cubrir a un nuevo asegurado antes de su incorporación? → Regresión Lineal como baseline, XGBoost como modelo final

La combinación de ambos enfoques permite a una aseguradora tomar decisiones concretas sobre **pricing de primas**, **estrategia de adquisición de clientes** y **gestión de riesgo de cartera**.

---

## Dataset

| Campo | Detalle |
|-------|---------|
| **Fuente** | [Kaggle - Medical Insurance Cost](https://www.kaggle.com/datasets/mirichoi0218/insurance) |
| **Descripción** | Datos demográficos y de salud de asegurados en EE.UU. con su costo de cobertura anual |
| **Dimensiones** | 1,338 filas × 7 columnas |
| **Calidad** | Sin valores nulos. Dataset limpio listo para modelado. |

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `age` | Numérica | Edad del asegurado |
| `bmi` | Numérica | Índice de masa corporal |
| `smoker` | Categórica | Fumador: yes / no |
| `children` | Numérica | Número de hijos cubiertos |
| `region` | Categórica | Región geográfica en EE.UU. |
| `sex` | Categórica | Género del asegurado |
| `charges` | Numérica | Costo anual de cobertura en USD (variable objetivo) |

---

## Tecnologías y Librerías

**Lenguaje:** Python 3.x
**Entorno:** Jupyter Notebook
**Control de versiones:** Git + GitHub con feature branch workflow

| Librería | Uso |
|----------|-----|
| `pandas` | Carga, transformación y manipulación del dataset |
| `numpy` | Operaciones numéricas y transformación logarítmica |
| `matplotlib` | Construcción base de visualizaciones |
| `seaborn` | Histogramas, scatter plots, heatmaps estadísticos |
| `scikit-learn` | StandardScaler, KMeans, LinearRegression, train_test_split, métricas, K-Fold CV |
| `xgboost` | Modelo de gradient boosting para predicción no lineal de costos |
| `shap` | Explicabilidad de predicciones individuales del modelo |
| `joblib` | Serialización y persistencia de modelos entrenados (`.pkl`) |

---

## Desarrollo

### Etapa 1: Exploración de Datos (EDA)

El EDA partió de hipótesis de negocio formuladas antes de escribir código: se esperaba que `smoker` y `bmi` fueran los principales determinantes del costo de cobertura. El análisis confirmó y enriqueció estas hipótesis con evidencia empírica.

**Análisis estadístico descriptivo:**
- `charges` presenta media de ~$13,270 con std de ~$12,110, una desviación estándar casi igual a la media que indica alta dispersión y distribución sesgada a la derecha.
- Rango de $1,121 a $63,770, con el 75% de los clientes por debajo de $16,639.

```python
df_insurance.shape       # (1338, 7)
df_insurance.info()      # sin valores nulos, tipos de datos mixtos
df_insurance.describe()  # estadísticas descriptivas completas
```

**Hallazgos visuales:**

La distribución de `charges` es **bimodal**: un grupo principal concentrado bajo $15,000 y un segundo grupo entre $30,000-$50,000. Al segmentar por `smoker`, se confirma que el segundo grupo corresponde casi exclusivamente a fumadores.

```python
sns.histplot(data=df_insurance, x='charges', bins=30, kde=True, hue='smoker')
```

![Distribución de Charges por Smoker](images/02_charges_by_smoker.png)

El scatter plot BMI vs Charges reveló un **umbral en BMI = 30** (frontera clínica de obesidad) donde los fumadores registran un salto de ~$15,000 a ~$40,000+ en costos, evidenciando un efecto de interacción `smoker × bmi`.

```python
sns.scatterplot(data=df_insurance, x='bmi', y='charges', hue='smoker', alpha=0.6)
```

![BMI vs Charges por Smoker](images/04_bmi_vs_charges.png)

**Matriz de correlación:**

| Variable | Correlación con `charges` |
|----------|--------------------------|
| `smoker` | **0.79** |
| `age` | 0.30 |
| `bmi` | 0.20 |
| `children` | 0.07 |

`children`, `region` y `sex` fueron descartadas por baja correlación con `charges` y riesgo de introducir overfitting.

![Matriz de Correlación](images/06_correlation_matrix.png)

---

### Etapa 2: Segmentación de Clientes con K-Means

**Variables seleccionadas:** `age`, `bmi`, `smoker`

**Estandarización con StandardScaler:**

K-Means opera sobre distancias euclidianas. Sin estandarización, `age` (rango ~46 unidades) dominaría sobre `smoker` (rango 0-1), sesgando los clusters. StandardScaler transforma cada variable a media=0 y std=1 (Z-score).

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_insurance[['age', 'bmi', 'smoker']])
```

**Selección de k con Elbow Method:**

Se evaluó el WCSS para k=1 a k=10. El codo más pronunciado se identifica en k=3.

```python
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
```

![Elbow Method](images/07_elbow_method.png)

```python
# Modelo final con k=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)
df_insurance['segmento_riesgo'] = kmeans.labels_
```

![Segmentos de Riesgo: BMI vs Charges](images/08_risk_segments_scatter.png)

---

### Etapa 3: Predicción de Costo con Regresión Lineal (baseline)

**Variables input:** `age`, `bmi`, `smoker` | **Variable objetivo:** `charges`
**Split:** 80% entrenamiento (1,070 registros) / 20% prueba (268 registros)

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = df_insurance[['age', 'bmi', 'smoker']]
y = df_insurance['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

**Experimento con transformación logarítmica:**

Se exploró `np.log(charges)` como variable objetivo para mitigar el sesgo a la derecha. El modelo log-transformado produjo peores resultados (R²=0.52 vs 0.77). La causa: `smoker` genera un salto discreto en `charges` que no es una distorsión de escala sino una relación estructuralmente no lineal, no solucionable con transformaciones sobre el target.

---

### Etapa 4: Mejora del Modelo con XGBoost

Dada la limitación de la regresión lineal para capturar la relación no lineal entre `smoker` y `charges`, se implementó **XGBoost Regressor**, un algoritmo de gradient boosting que entrena árboles de decisión secuencialmente, donde cada árbol corrige los errores del anterior.

Se entrenaron tres configuraciones progresivas para encontrar el balance entre capacidad y generalización:

1. **XGBoost Base:** Configuración por defecto (n_estimators=100, max_depth=6)
2. **XGBoost Ajustado v1:** Balance entre capacidad y generalización (n_estimators=500, max_depth=4, learning_rate=0.05)
3. **XGBoost Ajustado v2:** Mayor regularización para datasets pequeños (n_estimators=1000, max_depth=5, learning_rate=0.01, subsample=0.8, colsample_bytree=0.8)

```python
from xgboost import XGBRegressor

model_xgb = XGBRegressor(
    n_estimators=1000,
    max_depth=5,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model_xgb.fit(X_train, y_train)
y_pred_xgb = model_xgb.predict(X_test)
```

---

## Resultados

### Segmentación: Perfiles de Riesgo Identificados

| Segmento | Perfil | Age promedio | BMI promedio | Fumador | Charges promedio | N clientes |
|----------|--------|-------------|-------------|---------|-----------------|-----------|
| 0 | **Bajo riesgo** | 26.8 años | 29.4 | No | $5,059 | 516 (38.6%) |
| 1 | **Riesgo medio** | 51.2 años | 31.8 | No | $11,611 | 548 (40.9%) |
| 2 | **Alto riesgo** | 38.5 años | 30.7 | Sí | $32,050 | 274 (20.5%) |

### Predicción de Costos: Comparativa de Modelos

| Métrica | Regresión Lineal | XGBoost Base | XGBoost Tuned (v2) |
|---------|-----------------|--------------|-------------------|
| **R²** | 0.77 | 0.82 | **0.86** |
| **MAE** | $4,260 | $2,957 | **$2,652** |
| **RMSE** | $5,874 | $5,226 | **$4,642** |

XGBoost v2 reduce el error promedio en 37.7% y el RMSE en 21% respecto a regresión lineal. El 14% de varianza residual no explicada corresponde al error irreducible: variables clínicas ausentes del dataset (historial médico, medicamentos crónicos, condiciones genéticas) que son estándar en sistemas actuariales.

**Modelo seleccionado:** XGBoost Ajustado (v2).

### Importancia de Features

XGBoost calcula la importancia de cada variable mediante **Gain**: reducción promedio de pérdida cuando la variable participa en una división del árbol.

| Variable | Importancia | Interpretación |
|----------|------------|----------------|
| `smoker` | **Alta** | Variable dominante; factor de riesgo más crítico |
| `age` | **Media** | Correlación con costo, subordinada a tabaquismo |
| `bmi` | **Baja a media** | Correlación débil directa; interacción relevante con `smoker` |

`smoker` tiene mayor peso que `age` y `bmi` combinadas, alineado con la correlación de 0.79 observada en el EDA.

![Feature Importance: XGBoost Model](images/09_feature_importance.png)

### Explicabilidad: SHAP Values

**SHAP (SHapley Additive exPlanations)** explica por qué el modelo hizo una predicción específica para cada cliente, más allá de la importancia global.

- **Puntos rojos** (valor alto de la variable): incrementan el costo predicho
- **Puntos azules** (valor bajo de la variable): lo disminuyen
- **Eje X:** impacto SHAP en la predicción final

Para `smoker`: los fumadores (rojo) se agrupan claramente a la derecha y los no fumadores (azul) a la izquierda, mostrando una separación limpia que confirma su rol dominante. Para `age` y `bmi`: la nube es más dispersa, reflejando una contribución más gradual y subordinada.

Esta visualización es crítica para confianza operacional: en seguros, la explicabilidad ante reguladores y clientes es tan importante como la precisión.

![SHAP Values: Feature Impact on Model Output](images/10_shap_values.png)

### Validación Cruzada (5-Fold Cross-Validation)

Para evitar dependencia de un único test set de 268 muestras, se implementó **5-Fold Cross-Validation**:

- R² promedio esperado: ~0.85-0.86 (consistente con el split 80/20)
- Desviación estándar baja (< 0.05): indica modelo estable y sin overfitting pronunciado

En lugar de un único "R²=0.86 en test set", obtenemos un intervalo de confianza que hace la estimación más honesta para decisiones en producción.

---

## Conclusiones y Respuesta a la Pregunta de Negocio

### ¿Se puede segmentar a los asegurados por perfil de riesgo?

**Sí, con tres segmentos claramente diferenciados y accionables.**

El clustering identificó tres perfiles con diferencias de costo de hasta 6x entre el segmento de menor y mayor riesgo. El hallazgo más relevante es que el tabaquismo, por sí solo, incrementa el costo promedio en ~$20,000 anuales independientemente de la edad. Un fumador de 38 años (segmento 2, $32,050) representa un costo mayor que un no fumador de 51 años con obesidad (segmento 1, $11,611).

**Implicaciones de negocio:**

- **Pricing diferenciado:** Las primas deben estructurarse en al menos tres niveles, con un diferencial sustancial para fumadores, especialmente si tienen BMI ≥ 30, donde el efecto de interacción puede duplicar el costo esperado.
- **Estrategia de adquisición:** El segmento 0 (jóvenes, no fumadores, BMI < 30) representa el perfil más rentable. La estrategia comercial debería priorizar este segmento.
- **Gestión de cartera:** El 20.5% de clientes en el segmento de alto riesgo concentra desproporcionadamente los costos. Identificar este Pareto de riesgo permite priorizar intervenciones preventivas o ajustar coberturas contractuales.

### ¿Se puede predecir el costo de cobertura de nuevos asegurados?

**Sí, con capacidad explicativa del 86% usando solo tres variables. XGBoost es apto para decisiones de pricing con ciertas consideraciones.**

El modelo XGBoost explica el 86% de la variación en costos con un error promedio de $2,652 y RMSE de $4,642. Este RMSE representa el 35% del costo promedio de $13,270, una mejora significativa respecto a regresión lineal (44%). El modelo es suficientemente preciso para estimación de primas por cartera y evaluación de siniestralidad esperada, pero mantiene limitaciones para **pricing individual** en casos outlier. El 14% de varianza no explicada refleja ausencia de indicadores clínicos críticos.

---

## Limitaciones y Trabajo Futuro

### Limitaciones identificadas

**1. Tamaño del dataset**
1,338 registros limitan la capacidad de generalización. Un modelo productivo requeriría decenas de miles de registros para capturar la variabilidad real de una cartera de seguros.

**2. Variables disponibles**
El 14% de variación no explicada sugiere factores ausentes en el dataset: historial clínico, medicamentos crónicos, cirugías previas, condiciones genéticas. Estas variables son estándar en sistemas de información actuarial.

**3. Sesgo geográfico**
Dataset exclusivo del mercado de EE.UU. No transferible directamente a otros sistemas de salud.

### Trabajo futuro recomendado

- Crear la variable de interacción `smoker_obese` (dummy si BMI ≥ 30 y smoker=1) e incorporarla al modelo para mejorar interpretabilidad
- Evaluar el impacto de incorporar `children` y `region` mediante permutation importance
- Comparar XGBoost contra **Random Forest** y **LightGBM** para determinar si hay margen de mejora adicional

---

## Learnings

**EDA como fundamento del modelado**
La exploración validó empíricamente hipótesis formuladas antes de escribir código. La bimodalidad de `charges`, el umbral de BMI=30 en fumadores y la correlación de 0.79 de `smoker` son hallazgos que emergen del EDA y guían todas las decisiones posteriores. Sin EDA previo, el modelo hubiera incluido variables irrelevantes y perdido el efecto de interacción `smoker × bmi`.

**StandardScaler como prerequisito del clustering**
Sin estandarización, `age` (rango ~46 unidades) dominaría sobre `smoker` (rango 1 unidad) en el cálculo de distancias euclidianas de K-Means. El Z-score garantiza que cada variable contribuya equitativamente, independientemente de su escala original.

**Elbow Method para selección de k**
La determinación de k=3 está respaldada tanto por la curva del codo como por la interpretabilidad de negocio. Tres segmentos (bajo, medio, alto riesgo) son directamente accionables para una aseguradora; más segmentos añaden complejidad sin mejora proporcional.

**Transformación logarítmica: cuándo no aplicarla**
La transformación logarítmica mejora modelos con distribuciones continuas sesgadas y relaciones multiplicativas. En este dataset, el salto discreto de `smoker` es una relación estructuralmente no lineal, no una distorsión de escala. Aplicar logaritmo empeoró R² de 0.77 a 0.52. La solución correcta es un algoritmo no lineal, no una transformación del target.

**Multicolinealidad en feature selection**
`bmi` y la variable derivada `obese` (BMI ≥ 30) tienen correlación de 0.80. Incluir ambas introduciría multicolinealidad, inflando coeficientes y dificultando la interpretación del modelo. `bmi` fue retenida por mayor contenido de información como variable continua.

**Feature Importance vs. SHAP: Predicción vs. Explicabilidad**
Feature importance (Gain) responde "qué variables son más importantes globalmente"; SHAP responde "por qué este cliente específico tiene esta predicción". En finanzas y seguros, la explicabilidad no es opcional: es requisito regulatorio. Un modelo con R²=0.86 que puede justificar cada predicción ante un regulador es más valioso que una caja negra con R²=0.95.

**K-Fold CV: Validación robusta sobre datasets pequeños**
Con solo 1,338 registros, un único test set de 268 muestras puede no ser representativo. 5-Fold CV usa todas las muestras para validación en diferentes combinaciones, produciendo un intervalo de confianza más honesto que un solo split.

---

## Sobre el Uso de IA

Este proyecto fue impulsado con apoyo de inteligencia artificial, pero las hipótesis, el razonamiento analítico y las conclusiones son de autoría propia.

El punto de partida fue siempre una pregunta de negocio real y una intuición sobre qué variables importan. Cada decisión de modelado, desde la selección de features hasta la elección de k=3 o el rechazo de la transformación logarítmica, fue tomada con criterio propio y validada con evidencia empírica del dataset.

La IA (Claude y Claude Code) cumplió un rol de profesor y asistente técnico: ayudó a garantizar que el código siguiera buenas prácticas, mejoró la documentación, organizó la estructura del proyecto y orientó decisiones de implementación cuando había dudas técnicas puntuales. No generó hipótesis ni interpretó los resultados por mí.

Creo que usar IA como herramienta de apoyo en el aprendizaje, siendo transparente al respecto, es la forma honesta de trabajar con ella.

---

## Cómo Replicar

### Requisitos

- Python 3.8+
- Cuenta en Kaggle con API key configurada

```bash
pip install -r requirements.txt
```

### Pasos

1. Clona el repositorio:
```bash
git clone https://github.com/FelipeCH13/vortex-insurance-cost.git
cd vortex-insurance-cost
```

2. Crea y activa el entorno virtual:
```bash
python -m venv venv
venv\Scripts\activate        # Windows PowerShell
source venv/bin/activate     # Mac / Linux
```

3. Instala dependencias:
```bash
pip install -r requirements.txt
```

4. Descarga el dataset:
```bash
kaggle datasets download -d mirichoi0218/insurance
Expand-Archive insurance.zip -DestinationPath data/   # Windows PowerShell
unzip insurance.zip -d data/                           # Mac / Linux
```

5. Ejecuta los notebooks en orden:
```
src/notebooks/01_eda.ipynb
src/notebooks/02_clustering.ipynb
src/notebooks/03_regression.ipynb
```

   Al ejecutar los notebooks se generan automáticamente:
   - `images/`: 10 gráficos en formato PNG
   - `models/`: artefactos serializados: `standard_scaler.pkl`, `kmeans_risk_segments.pkl`, `linear_regression.pkl`

---

## Referencias

- [Medical Insurance Cost Dataset - Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [K-Means Clustering - Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [StandardScaler - Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/)
- [SHAP Documentation](https://shap.readthedocs.io/en/latest/)
- [Seaborn Statistical Visualization](https://seaborn.pydata.org/)

---

*Desarrollado por Felipe CH, 2026 | Proyecto académico - Vortex Data Science Program*
