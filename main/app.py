import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import time
import os

st.title("RAG & Fine-Tuned QA Interface (No Flask)")

# ------------------------
# Load model & tokenizer
# ------------------------
@st.cache_resource
def load_model():
    model_id = "pkwilp/my-safetensors-model"

    # Get token from Streamlit secrets (if private)
    hf_token = st.secrets.get("HUGGINGFACE", {}).get("TOKEN", None)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_auth_token=hf_token
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
    )
    return pipe

pipe = load_model()

# ------------------------
# UI
# ------------------------
mode = st.radio("Select Mode", ["RAG", "Fine-Tuned"], key="mode_selector")
user_query = st.text_area("Enter your question:", key="user_query")

if st.button("Ask", key="ask_button"):
    if not user_query.strip():
        st.error("Please enter a question.")
    else:
        start = time.time()
        # Simple RAG placeholder
        if mode == "RAG":
            context = "This is where retrieved context would go..."
            input_text = f"Context: {context}\n\nQuestion: {user_query}\nAnswer:"
        else:
            input_text = user_query

        outputs = pipe(
            input_text,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
        )

        answer = outputs[0]["generated_text"]
        elapsed = time.time() - start

        st.markdown(f"**Answer:** {answer}")
        st.markdown(f"**Mode:** {mode}")
        st.markdown(f"**Response Time:** {elapsed:.2f} seconds")

