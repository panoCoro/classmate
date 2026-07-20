import streamlit as st
from src.generate import generate_answer

st.set_page_config(page_title="Classmate Chatbot", page_icon=":robot_face:")

st.title("Classmate Chatbot")
st.caption("Ask questions about the Natural Language Processing module. Answers come only from the module materials with citations")
question = st.text_input("Enter your question:", "")

if st.button("Ask Classmate") and question.strip():
    with st.spinner("Searching the module materials..."):
        answer, sources = generate_answer(question.strip())
    st.subheader("Answer:")
    st.write(answer)
    refused = ("could not find an answer in the materials" in answer.lower() or "i don't know" in answer.lower())
    if sources and not refused:
        st.subheader("Sources:")
        for s in sources:
            st.markdown(f"- {s}")
    elif refused and sources:
        st.caption("The answer could not be found in the materials, but here are the closest sources that were searched: " + ", ".join(sources))