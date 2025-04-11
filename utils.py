import typing

if typing.TYPE_CHECKING:
    from pydantic import *  # noqa: F403
else:
    try:
        from pydantic.v1 import *  # noqa: F403
    except ImportError:
        pass

import streamlit as st
from streamlit.logger import get_logger
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from dotenv import load_dotenv

load_dotenv()
logger = get_logger("Langchain-Chatbot")


# decorator
def enable_chat_history(func):

    def execute(*args, **kwargs):
        func(*args, **kwargs)

    return execute


def display_msg(msg, author):
    """Method to display message on the UI

    Args:
        msg (str): message to display
        author (str): author of the message -user/assistant
    """
    if st.session_state.get("messages", None) is None:
        st.session_state.messages = []
    st.session_state.messages.append({"role": author, "content": msg})
    st.chat_message(author).write(msg)


def configure_llm():
    available_llms = [
        "llama3.2",
        "gemma3:4b",
    ]
    llm_opt = st.sidebar.radio(label="LLM", options=available_llms)
    if llm_opt == "llama3.2":
        llm = ChatOllama(model="llama3.2", base_url=st.secrets["OLLAMA_ENDPOINT"])
        return llm
    elif llm_opt == "gemma3:4b":
        llm = ChatOllama(model="gemma3:4b", base_url=st.secrets["OLLAMA_ENDPOINT"])
        return llm
    return None


def print_qa(cls, question, answer):
    log_str = "\nUsecase: {}\nQuestion: {}\nAnswer: {}\n" + "------" * 10
    logger.info(log_str.format(cls.__name__, question, answer))


@st.cache_resource
def configure_embedding_model():
    embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return embedding_model


def sync_st_session():
    for k, v in st.session_state.items():
        st.session_state[k] = v
