"""Interface de discussion Streamlit pour le RAG."""

import streamlit as st

from utils.rag import ask

st.set_page_config(page_title="Événements culturels", page_icon="🎭")
st.title("Trouver un événement culturel")
st.caption("Décris tes envies pour obtenir des recommandations à partir de la base.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("Nouvelle conversation"):
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Quel événement recherches-tu ?"):
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Recherche des événements et préparation de la réponse…"):
            result = ask(question, messages=st.session_state.messages)
        st.markdown(result["answer"])
        if result["events"]:
            with st.expander("Événements utilisés"):
                for event in result["events"]:
                    st.write(event["metadata"])
    st.session_state.messages.extend([
        {"role": "user", "content": question},
        {"role": "assistant", "content": result["answer"]},
    ])
