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
# Corrective RAG (CRAG)

Corrective-RAG (CRAG) is a strategy for RAG that incorperates self-reflection / self-grading on retrieved documents. 

In the paper [here](https://arxiv.org/pdf/2401.15884.pdf), a few steps are taken:

* If at least one document exceeds the threshold for relevance, then it proceeds to generation
* Before generation, it performns knowledge refinement
* This paritions the document into "knowledge strips"
* It grades each strip, and filters our irrelevant ones
* If all documents fall below the relevance threshold or if the grader is unsure, then the framework seeks an additional datasource
* It will use web search to supplement retrieval

We will implement some of these ideas from scratch using [LangGraph](https://python.langchain.com/docs/langgraph):

* Let's skip the knowledge refinement phase as a first pass. This can be added back as a node, if desired. 
* If *any* documents are irrelevant, let's opt to supplement retrieval with web search. 
* We'll use [Tavily Search](https://python.langchain.com/docs/integrations/tools/tavily_search) for web search.
* Let's use query re-writing to optimize the query for web search.

"""
)

st.image("./assets/crag.png", caption="RAG CRAG")
