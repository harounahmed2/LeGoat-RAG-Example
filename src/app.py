from typing import List
import streamlit as st
from streamlit_chat import message
from datetime import datetime
from PIL import Image
from utils import LoadConfig
from utils import load_data, RAG, get_Lebron
import subprocess
import os
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
    Document
)
from llama_index.llms import OpenAI
import openai


# ===================================
# Setting page title and header
# ===================================
im = Image.open("images/lebron.png")
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]

st.set_page_config(page_title="LeBron Score Tracking", page_icon=im, layout="wide")
st.markdown(
    "<h1 style='text-align: center;'>LeGoat-Tracker - Using RAG (Retrieval Augmented Generation) to keep up with LeBron James' All Time Achievements</h1>",
    unsafe_allow_html=True,
)
st.divider()

current_date = datetime.now()
date_str = current_date.strftime("%B %d")
day = current_date.day

if 4 <= day <= 20 or 24 <= day <= 30:
    suffix = "th"
else:
    suffix = ["st", "nd", "rd"][day % 10 - 1]

formatted_date = f"{date_str}{suffix}"

st.markdown(
        f"<center><i>[LAST UPDATED TODAY, {formatted_date}] LeGoat Tracker is an LLM assistant designed to give large language models access to static information downloaded from BasketballReference to keep up with LeBron's unprecedented historical longevity. The system does not keep track of your conversation and treats every turn/input independently. If possible, it will try to provide justification for its answer, a source, and a date</center>",
        unsafe_allow_html=True,
    )
st.divider()

# ===================================
# Initialise session state variables
# ===================================
if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

# ==================================
# Sidebar:
# ==================================
counter_placeholder = st.sidebar.empty()
with st.sidebar:
    st.markdown(
        "<h3 style='text-align: center;'>Ask about LeBron's current standing !</h3>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<center><b>Example: </b></center>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<center><i>Where does LeBron rank in assists all time?</i></center>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<center><i>What is LeBron's point Total? Did he pass 40,000 points yet?</i></center>",
        unsafe_allow_html=True,
    )
    st.sidebar.image("images/lebron.png", use_column_width=True)

clear_button = st.sidebar.button("Clear Conversation", key="clear"),
if clear_button:
    st.session_state["generated"].clear()
    st.session_state["past"].clear()
    # Optionally force a rerun to refresh the state immediately
    #st.experimental_rerun()

# ==================================

response_container = st.container()  # container for message display

Retriever = LoadConfig()

if query := st.chat_input(
    "What do you want to know about LeGoat?"
):
    st.session_state["past"].append(query)
    try:

        with st.spinner("Reading through current static file(s)..."):
            get_Lebron()
            data = load_data()
            index = RAG(Retriever, _docs=data)
            query_engine = index.as_query_engine(
                response_mode="tree_summarize",
                verbose=True,
                similarity_top_k=Retriever.similarity_top_k,
            )
        with st.spinner("Thinking..."):
            response = query_engine.query(query + Retriever.llm_format_output)

        st.session_state["generated"].append(response.response)
        del index
        del query_engine

        with response_container:
            for i in range(len(st.session_state["generated"])):
                message(st.session_state["past"][i], is_user=True)

                message(st.session_state["generated"][i], is_user=False)

    except Exception as e:
        print(e)
        st.session_state["generated"].append(
            "An error occured with reading the latest LeBron html download from basketball reference."
        )