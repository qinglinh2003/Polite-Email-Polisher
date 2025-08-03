import streamlit as st
from app.model import load_model, polite_rewrite

@st.cache_resource
def get_model():
    return load_model()

model = get_model()

def main():
    st.title("Polite Email Rewriter ✉️")

    input_text = st.text_area("Enter your email text:", height=200)

    if st.button("Rewrite"): 
        with st.spinner("Rewriting..."):
            result = polite_rewrite(input_text, model)
            st.text_area("Rewritten Email:", value=result, height=200)

if __name__ == "__main__":
    main()
