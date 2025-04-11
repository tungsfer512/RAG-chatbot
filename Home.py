import streamlit as st

st.set_page_config(page_title="Langchain Chatbot", page_icon="💬", layout="wide")

st.header("Chatbot Implementations with Langchain")

st.image("./assets/RAG.png", caption="RAG methods")

st.write(
    """
# RAG (RAG)
"""
)
st.image("./assets/rag_indexing.png", caption="RAG indexing")
st.image("./assets/rag_generation.png", caption="RAG generation")

st.write(
    """
# Rank RAG (Rank RAG)
"""
)
st.image("./assets/rank_rag.png", caption="Rank RAG")

st.write(
    """
# RAG Fusion (RAG Fusion)
"""
)
st.image("./assets/rag_fusion.png", caption="RAG Fusion")
