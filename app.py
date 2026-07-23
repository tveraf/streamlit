import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="Dashboard de Apuestas Deportivas", layout="wide", page_icon="⚽")

st.title("📈 Análisis de Apuestas Deportivas")
st.markdown("Este dashboard simula y analiza datos de apuestas deportivas con interacción dinámica.")

# --- BARRA LATERAL (SIMULACIÓN Y FILTROS) ---
st.sidebar.header("⚙️ Parámetros de Simulación")
n_bets = st.sidebar.slider("Número de Apuestas a simular", min_value=50, max_value=2000, value=300, step=50)

@st.cache_data
def simulate_betting_data(n):
    # Asegurar reproducibilidad en la misma sesión
    np.random.seed(42)
    sports = ['Fútbol', 'Baloncesto', 'Tenis', 'Béisbol', 'eSports']
    bet_types = ['Ganador', 'Over/Under', 'Hándicap', 'Parlay']
    results = ['Ganada', 'Perdida']

    data = {
        'Fecha': [datetime.today() - timedelta(days=np.random.randint(0, 365)) for _ in range(n)],
        'Deporte': np.random.choice(sports, n),
        'Tipo de Apuesta': np.random.choice(bet_types, n),
        'Cuota': np.round(np.random.uniform(1.2, 4.5), 2),
        'Stake (Inversión)': np.round(np.random.uniform(5, 100), 2),
        'Resultado': np.random.choice(results, n, p=[0.42, 0.58]) # Ligeramente más probable perder
    }
    df = pd.DataFrame(data)

    def calc_profit(row):
        if row['Resultado'] == 'Ganada':
            return round((row['Stake (Inversión)'] * row['Cuota']) - row['Stake (Inversión)'], 2)
        else:
            return -row['Stake (Inversión)']

    df['Beneficio Neto'] = df.apply(calc_profit, axis=1)
    df = df.sort_values(by='Fecha').reset_index(drop=True)
    return df

# Cargar datos simulados
df = simulate_betting_data(n_bets)

st.sidebar.markdown("---")
st.sidebar.header("🔍 Filtros de Análisis")
selected_sports = st.sidebar.multiselect("Filtrar por Deporte", df['Deporte'].unique(), default=df['Deporte'].unique())
selected_bet_types = st.sidebar.multiselect("Filtrar por Tipo de Apuesta", df['Tipo de Apuesta'].unique(), default=df['Tipo de Apuesta'].unique())

# Aplicar filtros
filtered_df = df[(df['Deporte'].isin(selected_sports)) & (df['Tipo de Apuesta'].isin(selected_bet_types))].copy()
filtered_df = filtered_df.sort_values(by='Fecha').reset_index(drop=True)
filtered_df['Beneficio Acumulado'] = filtered_df['Beneficio Neto'].cumsum()

# --- 1. ANÁLISIS CUANTITATIVO (KPIs) ---
st.header("📊 1. Análisis Cuantitativo")
col1, col2, col3, col4, col5 = st.columns(5)

total_bets = len(filtered_df)
total_stake = filtered_df['Stake (Inversión)'].sum()
total_profit = filtered_df['Beneficio Neto'].sum()
roi = (total_profit / total_stake) * 100 if total_stake > 0 else 0
win_rate = (len(filtered_df[filtered_df['Resultado'] == 'Ganada']) / total_bets) * 100 if total_bets > 0 else 0

col1.metric("Total Apuestas", total_bets)
col2.metric("Inversión Total", f"${total_stake:,.2f}")
col3.metric("Beneficio Neto", f"${total_profit:,.2f}", delta=f"{total_profit:,.2f}")
col4.metric("ROI (Retorno)", f"{roi:,.2f}%", delta=f"{roi:,.2f}%")
col5.metric("Win Rate (Acierto)", f"{win_rate:,.1f}%")

# --- 2. ANÁLISIS CUALITATIVO ---
st.header("🧠 2. Análisis Cualitativo")
if total_bets > 0:
    best_sport = filtered_df.groupby('Deporte')['Beneficio Neto'].sum().idxmax()
    worst_sport = filtered_df.groupby('Deporte')['Beneficio Neto'].sum().idxmin()
    best_bet_type = filtered_df.groupby('Tipo de Apuesta')['Beneficio Neto'].sum().idxmax()
    
    if roi > 0:
        insight = "🟢 **Estrategia Rentable:** Tu ROI es positivo. Estás venciendo el margen de la casa de apuestas."
    elif roi > -5:
        insight = "🟡 **Estrategia Equilibrada:** Estás rondando el punto de equilibrio. Pequeños ajustes en la selección de cuotas pueden llevarte a ganancias."
    else:
        insight = "🔴 **Estrategia en Pérdidas:** Revisa tu gestión de banca (Bankroll Management) o evita mercados de alta varianza."

    st.info(f"{insight}\n\n"
            f"🏆 **Fortalezas:** Tu deporte más rentable es el **{best_sport}** y tu mejor mercado es **{best_bet_type}**.\n\n"
            f"📉 **Debilidades:** El deporte que más pérdidas te genera es el **{worst_sport}**. Considera reducir tu stake en este sector.")
else:
    st.warning("No hay suficientes datos para el análisis cualitativo.")

# --- 3. ANÁLISIS GRÁFICO ---
st.header("📉 3. Análisis Gráfico")
tab1, tab2, tab3 = st.tabs(["Evolución de Bankroll", "Rendimiento por Deporte", "Distribución de Resultados"])

with tab1:
    fig_line = px.line(filtered_df, x='Fecha', y='Beneficio Acumulado', 
                       title='Curva de Beneficio Acumulado (Yield)',
                       markers=True)
    fig_line.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_line, use_container_width=True)

with tab2:
    profit_by_sport = filtered_df.groupby('Deporte')['Beneficio Neto'].sum().reset_index()
    fig_bar = px.bar(profit_by_sport, x='Deporte', y='Beneficio Neto', 
                     title='Beneficio/Pérdida por Deporte',
                     color='Beneficio Neto', color_continuous_scale=px.colors.diverging.RdYlGn)
    st.plotly_chart(fig_bar, use_container_width=True)

with tab3:
    fig_pie = px.pie(filtered_df, names='Resultado', title='Win Rate (Ganadas vs Perdidas)',
                     color='Resultado', color_discrete_map={'Ganada':'#00CC96', 'Perdida':'#EF553B'})
    st.plotly_chart(fig_pie, use_container_width=True)

# --- 4. INTERACCIÓN DINÁMICA CON RESULTADOS ---
st.header("📋 4. Historial de Apuestas Interactivo")
st.markdown("Explora los datos en bruto. Puedes ordenar las columnas haciendo clic en los encabezados.")

def highlight_result(val):
    color = '#d4edda' if val == 'Ganada' else '#f8d7da'
    text_color = '#155724' if val == 'Ganada' else '#721c24'
    return f'background-color: {color}; color: {text_color}'

st.dataframe(filtered_df[['Fecha', 'Deporte', 'Tipo de Apuesta', 'Cuota', 'Stake (Inversión)', 'Resultado', 'Beneficio Neto']]
             .style.map(highlight_result, subset=['Resultado'])
             .format({'Cuota': '{:.2f}', 'Stake (Inversión)': '${:.2f}', 'Beneficio Neto': '${:.2f}'}),
             use_container_width=True)
