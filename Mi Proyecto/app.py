import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos y modelo
modelo = joblib.load("modelo_entrenado.joblib")
df_train = pd.read_csv("Dataset/agricultural_yield_train.csv")
df_test = pd.read_csv("Dataset/agricultural_yield_test.csv")

# Sidebar para navegación
pagina = st.sidebar.selectbox("Selecciona una vista:", ["Predicción", "Análisis visual"])

if pagina == "Predicción":
    st.title("Predicción de Rendimiento Agrícola")

    # Inputs
    irrigation_schedule = st.number_input("Cantidad de riegos durante la temporada de cultivo", min_value=0, step=1)
    fertilizer_amount = st.number_input("Cantidad de fertilizante (kg/ha)", min_value=0.0, step=0.1)
    seed_variety_checkbox = st.checkbox("¿Usa una variedad de semilla de alto rendimiento?")
    seed_variety = 1 if seed_variety_checkbox else 0

    # Botón
    if st.button("Predecir rendimiento"):
        input_data = pd.DataFrame([{
            'Seed_Variety': seed_variety,
            'Irrigation_Schedule': irrigation_schedule,
            'Fertilizer_Amount_kg_per_hectare': fertilizer_amount
        }])
        try:
            prediccion = modelo.predict(input_data)[0]
            st.success(f"Predicción del rendimiento: {prediccion:.2f} kg/ha")
        except ValueError as e:
            st.error(f"Error al hacer la predicción: {e}")

elif pagina == "Análisis visual":
    st.title("Visualización del Análisis de Datos")

    # Matriz de correlación
    st.subheader("Matriz de Correlación")
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_train.corr(), annot=True, cmap='coolwarm')
    st.pyplot(plt.gcf())  # Mostrar la figura actual

    # Gráfico de predicciones vs valores reales
    st.subheader("Valores Reales vs Predichos")
    from sklearn.linear_model import LinearRegression

    X_train = df_train[['Seed_Variety', 'Irrigation_Schedule', 'Fertilizer_Amount_kg_per_hectare']]
    y_train = df_train['Yield_kg_per_hectare']
    modelo.fit(X_train, y_train)

    X_test = df_test[['Seed_Variety', 'Irrigation_Schedule', 'Fertilizer_Amount_kg_per_hectare']]
    y_test = df_test['Yield_kg_per_hectare']
    y_pred = modelo.predict(X_test)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='teal')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Valores reales')
    plt.ylabel('Predicciones del modelo')
    plt.title('Valores reales vs. Predicciones')
    plt.grid(True)
    st.pyplot(plt.gcf())
