import streamlit as st
import pandas as pd
import numpy as np 
import yfinance as yf
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Diccionario de nombres de acciones por ticker
ticker_names = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc. (Google)',
    'AMZN': 'Amazon.com Inc.',
    'META': 'Meta Platforms Inc. (Facebook)',
    'NVDA': 'NVIDIA Corporation',
    'TSLA': 'Tesla, Inc.',
    'ORCL': 'Oracle Corporation',
    'AMD': 'Advanced Micro Devices, Inc.'
}

# Título de la aplicación
st.title('Análisis de Regresión Múltiple de Acciones Tecnológicas')

# Sidebar para seleccionar las acciones y las fechas
st.sidebar.header('Selecciona las acciones y fechas')
options = list(ticker_names.values())
tech_stocks = st.sidebar.multiselect('Acciones tecnológicas', options)

compare_stock = st.sidebar.selectbox('Acción de comparación', options)

# Obtener los tickers correspondientes a los nombres seleccionados
selected_tickers = [key for key, value in ticker_names.items() if value in tech_stocks]
compare_ticker = [key for key, value in ticker_names.items() if value == compare_stock][0]

start_date = st.sidebar.date_input('Fecha de inicio', value=pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input('Fecha de fin', value=pd.to_datetime('2024-01-01'))

# Obtener datos de Yahoo Finance
data = {ticker: yf.download(ticker, start=start_date, end=end_date)['Adj Close'] for ticker in selected_tickers + [compare_ticker]}
df = pd.DataFrame(data)

# Eliminar filas con valores faltantes o infinitos
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Calcular la desviación de las acciones tecnológicas con respecto a la acción de comparación
for ticker in selected_tickers:
    df[f'Desviacion_{ticker}'] = df[ticker] - df[compare_ticker]

# Análisis descriptivo
with st.expander("Analisis descriptivo", expanded=False):
    st.header('Análisis Descriptivo')
    st.write(df.describe())

# Matriz de correlación
with st.expander("Matriz de Correlacion", expanded=False):
    st.header('Matriz de Correlación')
    correlation_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)


# Mostrar gráfica en una página separada
with st.expander("Graficas", expanded=False):
    from FuncionesProyecto import mostrar_grafica
    mostrar_grafica(df, selected_tickers, ticker_names, compare_stock)
    st.header('Gráfica de Precios Ajustados')
    fig, ax = plt.subplots(figsize=(10, 6))
    for ticker in selected_tickers:
        ax.plot(df[ticker], label=ticker_names[ticker])
    ax.plot(df[compare_ticker], label=ticker_names[compare_ticker], linestyle='--')
    ax.legend()
    st.pyplot(fig)
    

# Regresión de múltiples variables en otra página
from FuncionesProyecto import mostrar_modelo_regresion
mostrar_modelo_regresion(df, selected_tickers, compare_ticker)



