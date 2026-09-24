import streamlit as st
import pandas as pd
import tiktoken
from groq import Groq
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ==========================================
# Configuración de la Página
# ==========================================
st.set_page_config(page_title="NLP & LLM Dashboard", layout="wide")
st.title("Plataforma NLP: Generación, Tokens y Similitud")

# ==========================================
# Caché de Modelos Locales (Embeddings)
# ==========================================
@st.cache_resource
def load_embedding_model():
    # Modelo ligero para calcular embeddings reales
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = load_embedding_model()

# ==========================================
# Barra Lateral: Configuración y API Key
# ==========================================
with st.sidebar:
    st.header("Configuración del Modelo")
    api_key = st.text_input("Ingresa tu API Key de GROQ", type="password")
    
    # Modelos disponibles en la API de Groq
    modelo_seleccionado = st.selectbox(
        "Selecciona el LLM",
        ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma2-9b-it"]
    )
    
    temperatura = st.slider("Temperatura", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
    max_tokens = st.slider("Max Tokens", min_value=100, max_value=4096, value=1024, step=100)

# ==========================================
# Pestañas de la Aplicación
# ==========================================
tab1, tab2, tab3 = st.tabs(["Generación de Texto", "Análisis Lexical (Tokens & BoW)", "Embeddings & Similitud"])

# --- TAB 1: Generación de Texto ---
with tab1:
    st.header("Interacción con LLM (Groq)")
    user_prompt = st.text_area("Ingresa tu prompt aquí:", height=150)
    
    if st.button("Generar Respuesta"):
        if not api_key:
            st.error("Por favor, ingresa una API Key válida de Groq en la barra lateral.")
        elif not user_prompt:
            st.warning("El prompt está vacío.")
        else:
            try:
                client = Groq(api_key=api_key)
                with st.spinner("Generando..."):
                    chat_completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": user_prompt}],
                        model=modelo_seleccionado,
                        temperature=temperatura,
                        max_tokens=max_tokens,
                    )
                    respuesta = chat_completion.choices[0].message.content
                    st.write("### Respuesta:")
                    st.info(respuesta)
            except Exception as e:
                st.error(f"Error en la API: {e}")

# --- TAB 2: Análisis Lexical (Tokens & Bag of Words) ---
with tab2:
    st.header("Tokenización y Bag of Words (BoW)")
    texto_analisis = st.text_area("Ingresa un texto para analizar sus componentes:", "La inteligencia artificial transforma el mundo. El mundo cambia rápido.")
    
    if st.button("Analizar Texto"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Tokens y Tokens IDs")
            # Usamos tiktoken (estándar de OpenAI) como aproximación visual para entender la tokenización
            codificador = tiktoken.get_encoding("cl100k_base")
            tokens_ids = codificador.encode(texto_analisis)
            tokens = [codificador.decode([t]) for t in tokens_ids]
            
            df_tokens = pd.DataFrame({"Token": tokens, "Token ID": tokens_ids})
            st.dataframe(df_tokens, use_container_width=True)
            st.metric("Total de Tokens", len(tokens_ids))
            
        with col2:
            st.subheader("Bag of Words (BoW)")
            # Representación BoW usando Scikit-Learn
            vectorizer = CountVectorizer()
            try:
                bow_matrix = vectorizer.fit_transform([texto_analisis])
                df_bow = pd.DataFrame(bow_matrix.toarray(), columns=vectorizer.get_feature_names_out())
                st.dataframe(df_bow.T.rename(columns={0: "Frecuencia"}), use_container_width=True)
            except ValueError:
                st.warning("El texto es demasiado corto para generar un Bag of Words significativo.")

# --- TAB 3: Embeddings & Similitud ---
with tab3:
    st.header("Métricas de Similitud y Espacio Vectorial")
    st.markdown("Compara dos textos para ver qué tan similares son semánticamente usando Embeddings Densos y Similitud del Coseno.")
    
    texto_1 = st.text_input("Texto 1:", "El aprendizaje automático es una rama de la inteligencia artificial.")
    texto_2 = st.text_input("Texto 2:", "Machine learning pertenece al campo de la IA.")
    
    if st.button("Calcular Similitud"):
        with st.spinner("Calculando Embeddings..."):
            # Generar embeddings (vectores numéricos de alta dimensionalidad)
            emb1 = embedder.encode([texto_1])
            emb2 = embedder.encode([texto_2])
            
            # Calcular Similitud del Coseno
            similitud = cosine_similarity(emb1, emb2)[0][0]
            
            st.subheader("Resultados")
            st.metric("Similitud del Coseno", f"{similitud:.4f}")
            st.progress(float(similitud))
            
            with st.expander("Ver Vectores de Embeddings (Primeros 10 valores)"):
                df_embeddings = pd.DataFrame({
                    "Dimensión": range(1, 11),
                    "Vector Texto 1": emb1[0][:10],
                    "Vector Texto 2": emb2[0][:10]
                })
                st.dataframe(df_embeddings, hide_index=True)
