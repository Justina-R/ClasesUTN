# Predicción de Rendimiento Agrícola con Machine Learning

Este proyecto fue desarrollado como parte del curso de **Análisis de Datos e Inteligencia Artificial** de la **UTN Rosario**. Su objetivo es construir una herramienta interactiva para predecir el rendimiento agrícola a partir de variables relevantes como el tipo de semilla, la cantidad de riegos y el fertilizante aplicado. Para ello, se utilizó un dataset obtenido de la siguiente página https://www.kaggle.com/datasets/blueloki/synthetic-agricultural-yield-prediction-dataset.

## 📊 Descripción del proyecto

El proyecto incluye dos partes principales:

1. **Notebook de análisis exploratorio y entrenamiento del modelo** (`.ipynb`):
   - Se cargan y exploran los datasets `agricultural_yield_train.csv` y `agricultural_yield_test.csv`.
   - Se realiza un análisis visual de correlaciones entre variables.
   - Se entrena un modelo de regresión lineal con las variables seleccionadas:
     - `Seed_Variety` (variedad de semilla)
     - `Irrigation_Schedule` (cantidad de riegos)
     - `Fertilizer_Amount_kg_per_hectare` (fertilizante en kg/ha)
   - El modelo alcanza un R² de **0.85**, mostrando buena capacidad predictiva.

2. **Aplicación interactiva con Streamlit** (`app.py`):
   - Permite al usuario ingresar los valores mencionados y obtener una predicción del rendimiento en kg/ha.
   - También incluye visualizaciones:
     - Matriz de correlación de variables.
     - Gráfico de valores reales vs. predichos.

## 🛠️ Tecnologías utilizadas

- Python
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- Streamlit
- joblib

## 🚀 Cómo ejecutar la app

1. Clonar este repositorio.
2. Crear y activar un entorno virtual (opcional pero recomendado):

```bash
python -m venv venv
venv\Scripts\activate
```

3. Instalar dependencias necesarias con `pip install -r requirements.txt`.
4. Ejecutar la aplicación:

```bash
streamlit run app.py
```

## 📁 Estructura del proyecto

```
├── app.py
├── modelo_entrenado.joblib
├── requirements.txt
├── Dataset/
│   ├── agricultural_yield_train.csv
│   └── agricultural_yield_test.csv
├── eda_modelo.ipynb
└── README.md
```
