import streamlit as st
from src.generate import generate_answer, REFUSAL_NEEDLE, is_refusal
from src.verify import faithfulness
import time 

st.set_page_config(page_title="Classmate Chatbot", page_icon=":robot_face:")

st.title("Classmate Chatbot")
st.caption("Ask questions about the Natural Language Processing module. Answers come only from the module materials with citations")
question = st.text_input("Enter your question:", "")

if st.button("Ask Classmate") and question.strip():
    t0 = time.perf_counter()
    with st.spinner("Searching the module materials..."):
        answer, sources, results = generate_answer(question.strip())
    st.subheader("Answer:")
    st.write(answer)
    refused = is_refusal(answer)
    if not refused:
        t_v = time.perf_counter()
        with st.spinner("Verifying faithfulness of the answer..."):
            score, details = faithfulness(answer, results)
            print(f"[timing] faithfulness: {time.perf_counter() - t_v:.2f} seconds")
        if score is not None:
            st.caption(f"Faithfulness Score (weakest claim): {score:.2f}")
            with st.expander("Per-sentence support scores"):
                for sent, sup in details:
                    st.markdown(f"- **{sup:.2f}** - {sent}")
    elapsed = time.perf_counter() - t0
    st.caption(f"Answered in: {elapsed:.1f} seconds")
    if sources and not refused:
        st.subheader("Sources:")
        for s in sources:
            st.markdown(f"- {s}")
    elif refused and sources:
        st.caption("The answer could not be found in the materials, but here are the closest sources that were searched: " + ", ".join(sources))
        