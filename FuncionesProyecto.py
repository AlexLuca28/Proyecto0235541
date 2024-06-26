import streamlit as st
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def mostrar_grafica(df, selected_tickers, ticker_names, compare_stock):
    
    st.header('Gráfica de Desviación de Acciones Tecnológicas')
    plt.figure(figsize=(10, 6))
    for ticker in selected_tickers:
        plt.plot(df.index, df[f'Desviacion_{ticker}'], label=f'Desviación {ticker_names[ticker]}')

    plt.xlabel('Fecha')
    plt.ylabel('Desviación de Precio de Cierre Ajustado')
    plt.title(f'Desviación de Acciones Tecnológicas con respecto a {compare_stock}')
    plt.legend()
    st.pyplot(plt)

def mostrar_modelo_regresion(df, selected_tickers, compare_ticker):
    if st.button('Mostrar Modelo de Regresión'):
        st.header('Modelo de Regresión')
        
        # Widget para mostrar u ocultar la explicación
        with st.expander("Explicación del Modelo de Regresión", expanded=False):
            st.markdown("""
            ### Explicación del Modelo de Regresión
            El modelo de regresión lineal ordinaria (OLS) es un método estadístico utilizado para analizar la relación entre una variable dependiente y una o más variables independientes.
            
            En este caso, estamos realizando una regresión lineal múltiple utilizando el método OLS. Esto implica que estamos tratando de predecir la desviación de las acciones tecnológicas seleccionadas con respecto a la acción de comparación basándonos en los precios de cierre ajustados de estas acciones.
            
            ### Componentes del Modelo
            - **Variables Independientes (X)**: Son las variables utilizadas para predecir la variable dependiente. En nuestro caso, estas son los precios de cierre ajustados de las acciones tecnológicas seleccionadas.
            
            - **Variable Dependiente (y)**: Es la variable que estamos tratando de predecir. En nuestro caso, es la desviación de las acciones tecnológicas seleccionadas con respecto a la acción de comparación.
            
            - **Constante**: La constante (intercepto) en el modelo de regresión lineal representa el valor de la variable dependiente cuando todas las variables independientes son cero.
            
            - **Coeficientes**: Los coeficientes en el modelo representan la relación entre cada variable independiente y la variable dependiente. Un coeficiente positivo indica una relación positiva, mientras que un coeficiente negativo indica una relación negativa.
            
            ### Resumen del Modelo
            A continuación, se muestra el resumen del modelo de regresión lineal múltiple:
            - **R-cuadrado (R-squared)**: Es una medida de cuánto la variabilidad de la variable dependiente puede explicarse por las variables independientes en el modelo. Un valor cercano a 1 indica un buen ajuste del modelo.
            
            - **P-valor (P-value)**: Indica la significancia estadística de cada coeficiente en el modelo. Un p-valor bajo (generalmente menos de 0.05) sugiere que el coeficiente es significativamente diferente de cero.
            
            - **Coeficientes**: Los coeficientes representan la magnitud y dirección de la relación entre las variables independientes y la variable dependiente. 
            
            - **Error Estándar (Standard Error)**: Es una medida de la precisión de los coeficientes estimados. Un error estándar bajo indica mayor precisión.
            
            - **Intervalo de Confianza (Confidence Interval)**: Proporciona un rango de valores dentro del cual se espera que esté el verdadero valor del coeficiente con cierto nivel de confianza.
            
            - **Probabilidad del Modelo (F-statistic)**: Evalúa la significancia global del modelo. Un valor bajo sugiere que al menos una variable independiente tiene un efecto significativo en la variable dependiente.
            """)

        # Regresión de múltiples variables
        X = df[selected_tickers]
        X = sm.add_constant(X)  # Agregar una constante para el término independiente
        y = df[compare_ticker]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        model = sm.OLS(y_train, X_train).fit()

        # Mostrar resumen del modelo
        st.text(model.summary())

        # Calcular y mostrar el error cuadrático medio
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        st.write(f'Error Cuadrático Medio: {mse}')
        
        # Análisis del modelo y conclusión
        def analizar_y_concluir(model):
            st.subheader("Análisis del Modelo y Conclusión")
            
            # Obtener el resumen del modelo
            summary = model.summary2().tables[1]
            
            # Análisis R-cuadrado
            r_squared = model.rsquared
            st.write(f'**R-cuadrado:** {r_squared:.4f}')
            if r_squared >= 0.8:
                st.write("El modelo tiene un buen ajuste.")
            elif 0.5 <= r_squared < 0.8:
                st.write("El modelo tiene un ajuste moderado.")
            else:
                st.write("El modelo tiene un ajuste pobre.")
            
            # Análisis P-valores
            significant_vars = summary[summary['P>|t|'] < 0.05].index.tolist()
            st.write(f'**Variables significativas:** {significant_vars}')
            if not significant_vars:
                st.write("No hay variables significativas en el modelo.")
            
            # Análisis de los coeficientes
            st.write(f'**Coeficientes del modelo:**')
            st.dataframe(summary[['Coef.', 'P>|t|', '[0.025', '0.975]']])

            # Conclusión basada en el análisis
            conclusion = "Conclusión del Modelo:\n\n"
            conclusion += f"El R-cuadrado del modelo es {r_squared:.4f}, "
            if r_squared >= 0.8:
                conclusion += "lo que indica que el modelo tiene un buen ajuste a los datos. "
            elif 0.5 <= r_squared < 0.8:
                conclusion += "lo que indica que el modelo tiene un ajuste moderado a los datos. "
            else:
                conclusion += "lo que indica que el modelo tiene un ajuste pobre a los datos. "

            if significant_vars:
                conclusion += f"Las siguientes variables son significativas: {', '.join(significant_vars)}. "
            else:
                conclusion += "No hay variables significativas en el modelo. "

            conclusion += "Recomendamos revisar las variables utilizadas y considerar la inclusión de otras variables que puedan mejorar el modelo."

            st.write(conclusion)
        
        # Llamada a la función de análisis y conclusión
        analizar_y_concluir(model)
