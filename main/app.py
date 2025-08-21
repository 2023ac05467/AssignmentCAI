# app.py -- Consolidated Streamlit app (no Flask)
# Retains all RAG + Fine-Tuned + Groq functionality from Flask.py
# Deploy this on Streamlit Cloud. Use st.secrets for HF and GROQ tokens.

import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import streamlit as st
import json
import time
import warnings
import numpy as np
import joblib
from groq import Groq

# ML libs
import faiss
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    pipeline,
)
import torch
import requests

warnings.filterwarnings("ignore", message="resource_tracker: There appear to be")

# ===========================
# Config / Paths (edit as needed)
# ===========================
# If you moved your finetuned model to Hugging Face, set MODEL_REPO to HF repo id.
# Otherwise set FINETUNED_MODEL_PATH to local path (not recommended for Streamlit Cloud).
MODEL_REPO = "pkwilp/my-safetensors-model"  # preferred: HF repo
FINETUNED_MODEL_PATH = "../ipynb/qa_finetuned_model_saved"  # fallback local path

CHUNKS_PATH = "data/embeddings/chunks.json"     # relative to repo root on Streamlit Cloud
FAISS_INDEX_PATH = "data/embeddings/faiss_index.idx"
TFIDF_VECTORIZER_PATH = "data/embeddings/tfidf_vectorizer.joblib"
MEMORY_BANK_PATH = "data/embeddings/memory_bank.json"

# Retrieval thresholds (tweakable)
RAG_SCOPE_SIM_THRESHOLD = 0.30
FT_SCOPE_SIM_THRESHOLD = 0.01

# Guardrails
BANNED_WORDS = {'hack', 'attack', 'kill', 'illegal', 'violence'}
HALLUCINATION_PHRASES = [
    "as an ai", "i do not know", "i'm just a language model", "cannot answer", "no information"
]

# =====================================
# Helper: Secure tokens from Streamlit secrets
# =====================================
def get_hf_token():
    return st.secrets.get("HUGGINGFACE", {}).get("TOKEN", None)

def get_groq_key():
    return "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"
#    return st.secrets.get("GROQ", {}).get("API_KEY", "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")


# =====================================
# Resource loaders (cached)
# =====================================
@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def load_tfidf_vectorizer(path=TFIDF_VECTORIZER_PATH):
    if os.path.exists(path):
        return joblib.load(path)
    st.warning(f"TF-IDF vectorizer not found at {path}. Sparse retrieval disabled.")
    return None

@st.cache_resource(show_spinner=False)
def load_chunks(path=CHUNKS_PATH):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        # Expect chunks to be a list of dicts {'id':..., 'chunk':..., 'meta':...}
        return chunks
    st.warning(f"Chunks file not found at {path}.")
    return []

@st.cache_resource(show_spinner=False)
def load_faiss_index(path=FAISS_INDEX_PATH):
    if os.path.exists(path):
        try:
            idx = faiss.read_index(path)
            return idx
        except Exception as e:
            st.error(f"Failed to read FAISS index at {path}: {e}")
            return None
    st.warning(f"FAISS index not found at {path}. Dense retrieval disabled.")
    return None

@st.cache_resource(show_spinner=False)
def load_memory_bank(path=MEMORY_BANK_PATH):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

@st.cache_resource(show_spinner=False)
def load_finetuned_model(model_repo=MODEL_REPO, local_path=FINETUNED_MODEL_PATH):
    # Determine whether to use HF repo or local path
    use_hf = True if model_repo and (not local_path or not os.path.exists(local_path)) else False
    hf_token = get_hf_token()

    # Choose repo/source
    source = model_repo if use_hf else local_path

    # Auto-detect config to decide Seq2Seq vs CausalLM
    try:
        cfg = AutoConfig.from_pretrained(source, use_auth_token=hf_token if use_hf else None)
    except Exception as e:
        st.warning(f"Could not load config from {source}: {e}")
        cfg = None

    model = None
    tokenizer = None
    text_pipe = None

    try:
        tokenizer = AutoTokenizer.from_pretrained(source, use_auth_token=hf_token if use_hf else None)
    except Exception as e:
        st.warning(f"Could not load tokenizer from {source}: {e}")

    try:
        if cfg is not None and getattr(cfg, "is_encoder_decoder", False):
            # Seq2Seq (T5, Bart, etc.)
            model = AutoModelForSeq2SeqLM.from_pretrained(source, use_auth_token=hf_token if use_hf else None)
            text_pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)
        else:
            # Try causal LM
            model = AutoModelForCausalLM.from_pretrained(source, use_auth_token=hf_token if use_hf else None)
            text_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
    except Exception as e:
        st.warning(f"Could not load model from {source}: {e}")
        model = None
        text_pipe = None

    return {
        "tokenizer": tokenizer,
        "model": model,
        "pipeline": text_pipe,
        "config": cfg,
        "source": source,
    }

@st.cache_resource(show_spinner=False)
def init_groq_client():
    key = get_groq_key()
    if not key:
        return None
    if Groq is None:
        st.warning("Groq SDK not installed; Groq generation disabled.")
        return None
    try:
        client = Groq(api_key=key)
        return client
    except Exception as e:
        st.warning(f"Failed to init Groq client: {e}")
        return None

# =====================================
# Load everything (lazy)
# =====================================
embedder = load_embedder()
chunks = load_chunks()
chunk_ids = [c.get("id") for c in chunks] if chunks else []
faiss_index = load_faiss_index()
tfidf_vectorizer = load_tfidf_vectorizer()
tfidf_matrix = None
if tfidf_vectorizer and chunks:
    tfidf_matrix = tfidf_vectorizer.transform([c["chunk"] for c in chunks])

memory_bank = load_memory_bank()

finetuned = load_finetuned_model()
hf_pipeline = finetuned.get("pipeline") if finetuned else None
groq_client = init_groq_client()

# Prevent FAISS/OpenMP multithreading conflicts
try:
    faiss.omp_set_num_threads(1)
except Exception:
    pass

# =====================================
# Preprocessing & Memory utilities
# =====================================
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def preprocess(text: str) -> str:
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    tokens = text.lower().split()
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)

def load_memory_bank_local():
    global memory_bank
    try:
        with open(MEMORY_BANK_PATH, "r", encoding="utf-8") as f:
            memory_bank = json.load(f)
    except Exception:
        memory_bank = []
    return memory_bank

def save_to_memory_bank(query, llm_output):
    mem = load_memory_bank_local()
    query_clean = preprocess(query)
    q_emb = embedder.encode([query_clean])[0].tolist()
    mem.append({"query": query_clean, "query_emb": q_emb, "llm_output": llm_output})
    try:
        os.makedirs(os.path.dirname(MEMORY_BANK_PATH), exist_ok=True)
        with open(MEMORY_BANK_PATH, "w", encoding="utf-8") as f:
            json.dump(mem, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"Could not save memory bank: {e}")

def memory_bank_match(query, threshold=0.9):
    mem = load_memory_bank_local()
    if not mem:
        return None
    q_clean = preprocess(query)
    q_emb = embedder.encode([q_clean])[0]
    for item in mem:
        mem_emb = np.array(item["query_emb"])
        sim = float(np.dot(q_emb, mem_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(mem_emb) + 1e-8))
        if sim > threshold:
            return item["llm_output"]
    return None

# =====================================
# Retrieval functions
# =====================================
def dense_retrieve(query, top_n=5):
    """Use FAISS index for dense retrieval. Returns list of (id,score)."""
    if faiss_index is None:
        return []
    q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
    D, I = faiss_index.search(q_emb, top_n)
    results = []
    for rank, idx in enumerate(I[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        results.append((chunks[idx]["id"], float(D[0][rank])))
    return results

def sparse_retrieve(query, top_n=5):
    """Use TF-IDF vectorizer for sparse retrieval. Returns list of (id,score)."""
    if tfidf_vectorizer is None or tfidf_matrix is None:
        return []
    q_vec = tfidf_vectorizer.transform([query])
    scores = (tfidf_matrix @ q_vec.T).toarray().squeeze()
    top_idx = np.argsort(scores)[::-1][:top_n]
    return [(chunks[i]["id"], float(scores[i])) for i in top_idx]

def combine_results(dense, sparse, top_n=5, alpha=0.5):
    """Fuse dense & sparse results. Returns list of chunk dicts (top_n)."""
    score_map = {}
    for cid, score in dense:
        score_map[cid] = score_map.get(cid, 0) + alpha * score
    for cid, score in sparse:
        score_map[cid] = score_map.get(cid, 0) + (1 - alpha) * score
    sorted_ids = sorted(score_map.items(), key=lambda x: -x[1])[:top_n]
    final_chunks = []
    for cid, sc in sorted_ids:
        # find chunk object
        c = next((c for c in chunks if c["id"] == cid), None)
        if c:
            final_chunks.append({"id": cid, "chunk": c["chunk"], "score": sc, "meta": c.get("meta")})
    return final_chunks

# =====================================
# Generation functions
# =====================================
def generate_response_local(user_query, retrieved_chunks, max_new_tokens=128):
    """Use local HF pipeline (seq2seq or causal)"""
    if hf_pipeline is None:
        return "Local model unavailable", False

    context = "\n".join([c["chunk"] for c in retrieved_chunks])
    context = "\n"
    prompt = f"{context}\nUser: {user_query}\nAnswer:"
    
    try:
        # Different pipeline types return different fields; normalize
        start_time = time.time()
        out = hf_pipeline(prompt, max_new_tokens=max_new_tokens)
        elapsed_time = time.time() - start_time
        if isinstance(out, list):
            # text2text-generation returns [{"generated_text": "..."}]
            txt = out[0].get("generated_text") or out[0].get("generated_text") or str(out[0])
        else:
            txt = str(out)
        return txt.strip(), elapsed_time, True
    except Exception as e:
        return f"Local generation error: {e}", 0, False

def generate_response_groq(user_query, retrieved_chunks, model_name="llama-3.3-70b-versatile", max_tokens=256, temperature=0.1):
    """Use Groq client if available."""
    client = groq_client
    if client is None:
        return "Groq client not configured", False
    context = "\n".join([c["chunk"] for c in retrieved_chunks])
    prompt = f"{context}\nUser: {user_query}\nAnswer:"
    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        elapsed_time = time.time() - start_time
        # groq SDK may return choices structure; adapt defensively
        txt = ""
        try:
            txt = response.choices[0].message.content.strip()
        except Exception:
            # Another fallback format:
            txt = str(response)
        return txt, elapsed_time, True
    except Exception as e:
        return f"Groq call failed: {e}", 0, False

# =====================================
# Guardrails
# =====================================
def is_query_valid(query):
    if not query or len(query.strip()) < 3:
        return False, "Query too short or empty."
    lowered = query.lower()
    if any(b in lowered for b in BANNED_WORDS):
        return False, "Query contains banned/harmful content."
    return True, ""

def is_output_factual(output):
    lowered = output.lower()
    if any(p in lowered for p in HALLUCINATION_PHRASES):
        return False
    if len(output.strip()) < 10:
        return False
    return True

# =====================================
# Streamlit UI - Single consolidated interface
# =====================================
st.set_page_config(page_title="RAG + Fine-tuned QA (No Flask/Deploying to Streamlit Cloud)", layout="wide")
st.title("RAG & Fine-Tuned QA Interface")
st.subheader("CAI Assignment 2 - Group 100")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    mode = st.radio("Mode", ["RAG", "Fine-Tuned"], index=0)
    use_groq = st.checkbox("Use Groq for generation (if configured)", value=True, disabled = True)
    top_n = st.slider("Top N retrieved chunks", 1, 10, 5)
    alpha = st.slider("Dense/Sparse fusion alpha (dense weight)", 0.0, 1.0, 0.5)
    max_new_tokens = st.number_input("Max generation tokens", min_value=16, max_value=1024, value=256, step=16)
    show_chunks = st.checkbox("Show retrieved chunks", value=True)
    show_memory = st.checkbox("Use memory bank match (threshold 0.9)", value=True)

# Main UI
query = st.text_area("Enter your question:", height=150)
submit = st.button("Ask")

if submit:
    valid, reason = is_query_valid(query)
    if not valid:
        st.error(f"Invalid query: {reason}")
    else:
        # Check memory bank first (optional)
        mem_ans = None
        if show_memory:
            mem_ans = memory_bank_match(query, threshold=0.9)
        if mem_ans:
            st.success("Answer from memory bank:")
            st.write(mem_ans)
        else:
            

            # Preprocess for retrieval
            q_proc = preprocess(query)

            # Dense + sparse retrieval
            dense = dense_retrieve(q_proc, top_n=top_n)
            sparse = sparse_retrieve(q_proc, top_n=top_n)
            combined = combine_results(dense, sparse, top_n=top_n, alpha=alpha)

            # If combined empty, fallback to top sparse or dense
            if not combined:
                # fallback to sparse results text
                if sparse:
                    combined = []
                    for cid, sc in sparse[:top_n]:
                        c = next((c for c in chunks if c["id"] == cid), None)
                        if c:
                            combined.append({"id": cid, "chunk": c["chunk"], "score": sc})
                elif dense:
                    combined = []
                    for cid, sc in dense[:top_n]:
                        c = next((c for c in chunks if c["id"] == cid), None)
                        if c:
                            combined.append({"id": cid, "chunk": c["chunk"], "score": sc})

            # Generate answer
            start_time = time.time()
            #if use_groq:
            if mode == "RAG":
                out_text, time_server, ok = generate_response_groq(query, combined, max_tokens=max_new_tokens)
            else:
                out_text, time_server, ok = generate_response_local(query, combined, max_new_tokens=max_new_tokens)

            elapsed = time.time() - start_time
            # Post-filtering guardrails
            factual = is_output_factual(out_text) if ok else False
            
            # Compute confidence from retrieval scores (use highest dense score or sparse as proxy)
            max_sim = 0.0
            if dense:
                max_sim = max([s for _, s in dense]) if dense else 0.0
            elif sparse:
                max_sim = max([s for _, s in sparse]) if sparse else 0.0

            # If low similarity or not factual flag, indicate out-of-scope
            if max_sim < (RAG_SCOPE_SIM_THRESHOLD if mode.lower() == "rag" else FT_SCOPE_SIM_THRESHOLD) or not factual:
                st.warning("Answer may be out of scope or non-factual.")
                # still show the model output but flag it
                st.markdown("**Model output (flagged):**")
                st.write(out_text)
                st.write(f"Confidence (retrieval sim proxy): {max_sim:.3f}")
                st.write(f"Elapsed time: {elapsed:.3f}s")
                st.write(f"Inference time: {time_server:.3f}s")
            else:
                st.success("Answer (generated):")
                st.write(out_text)
                st.write(f"Confidence (retrieval sim proxy): {max_sim:.3f}")
                st.write(f"Elapsed time: {elapsed:.3f}s")
                st.write(f"Inference time: {time_server:.3f}s")
            # Show retrieved chunks if requested
            if show_chunks:
                st.subheader("Retrieved Chunks")
                st.json(combined)

            # Save to memory bank (async-like behavior is not possible; we save synchronously)
            try:
                save_to_memory_bank(query, out_text)
            except Exception:
                pass

# Footer: small diagnostics / helpful info
st.sidebar.markdown("---")
st.sidebar.markdown("Model source:")
st.sidebar.markdown(f"`{finetuned.get('source') if finetuned else 'None'}`")
st.sidebar.markdown("FAISS index:")
st.sidebar.markdown(f"`{FAISS_INDEX_PATH}`" if faiss_index is not None else "Not loaded")
st.sidebar.markdown("TF-IDF vectorizer:")
st.sidebar.markdown(f"`{TFIDF_VECTORIZER_PATH}`" if tfidf_vectorizer is not None else "Not loaded")
st.sidebar.markdown("MEMORY_BANK::")
st.sidebar.markdown(f"`{MEMORY_BANK_PATH}`" if memory_bank is not None else "Not loaded")
