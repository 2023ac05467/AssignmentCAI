import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import streamlit as st
import joblib
import faiss
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM,pipeline
import requests
import warnings
import time
from groq import Groq

### ===============================
### 2. Retrieval-Augmented Generation (RAG) System Implementation
# Split the cleaned text into chunks suitable for retrieval with at least two chunk sizes (e.g., 100 and 400 tokens).
# Assign unique IDs and metadata to chunks.
# 2.2 Embedding & Indexing

# Embed chunks using a small open-source sentence embedding model (e.g., all-MiniLM-L6-v2, E5-small-v2).
# Build:
# Dense vector store (e.g., FAISS, ChromaDB).
# Sparse index (BM25 or TF-IDF) for keyword retrieval.
# 2.3 Hybrid Retrieval Pipeline

# For each user query:
# Preprocess (clean, lowercase, stopword removal).
# Generate query embedding.
# Retrieve top-N chunks from:
# Dense retrieval (vector similarity).
# Sparse retrieval (BM25).
# Combine results by union or weighted score fusion.
# 2.4 Advanced RAG Technique (Select One)
    
# Memory-Augmented Retrieval

# Supplement retrieval with a persistent memory bank of frequently asked or important Q&A pairs.
# 2.5 Response Generation

# Use a small, open-source generative model (e.g., DistilGPT2, GPT-2 Small, Llama-2 7B if available).
# Concatenate retrieved passages and user query as input to generate the final answer.
# Limit total input tokens to the model context window.
# 2.6 Guardrail Implementation

# Implement one guardrail:
# Input-side: Validate queries to filter out irrelevant or harmful inputs.
# Output-side: Filter or flag hallucinated or non-factual outputs.

torch.set_num_threads(1)

# Prevent tokenizer parallelism deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be")

# Disable FAISS multithreading conflicts
faiss.omp_set_num_threads(1)
RAG_SCOPE_SIM_THRESHOLD = 0.30  # Threshold for similarity in RAG mode
FT_SCOPE_SIM_THRESHOLD=0.01 # Threshold for similarity in Fine-Tuned mode
# ===============================
###2.6 Guardrail Implementation
# ===============================
BANNED_WORDS = {'hack', 'attack', 'kill', 'illegal', 'violence'}
HALLUCINATION_PHRASES = [
    "as an ai", "i do not know", "i'm just a language model", "cannot answer", "no information"
]

# Paths
MODEL_REPO = "pkwilp/my-safetensors-model"  # preferred: HF repo
FINETUNED_MODEL_PATH = "pkwilp/my-safetensors-model"
CHUNKS_PATH = 'data/embeddings/chunks.json'
FAISS_INDEX_PATH = 'data/embeddings/faiss_index.idx'
TFIDF_VECTORIZER_PATH = 'data/embeddings/tfidf_vectorizer.joblib'
MEMORY_BANK_PATH = 'data/embeddings/memory_bank.json'

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["GROQ_API_KEY"]="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"
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# ===============================
# Globals (lazy loaded later)
embedder = None
chunks = None
chunk_ids = None
faiss_index = None
tfidf_vectorizer = None
tfidf_matrix = None
qa_pipeline_ft = None

# =====================================
# Helper: Secure tokens from Streamlit secrets
# =====================================
def get_hf_token():
    return st.secrets.get("HUGGINGFACE", {}).get("TOKEN", None)

def get_groq_key():
    return "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"
#    return st.secrets.get("GROQ", {}).get("API_KEY", "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")

# ===============================
# 2.1 Data Processing
# ===============================
def preprocess(text):
    import re
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = text.lower().split()
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    return ' '.join(tokens)

# ===============================
# Memory Bank (    Memory-Augmented Retrieval)
# ===============================
def load_memory_bank():
    try:
        with open(MEMORY_BANK_PATH, 'r') as f:
            return json.load(f)
    except:
        return []

def save_to_memory_bank(query, llm_output):
    memory_bank = load_memory_bank()
    query_clean = preprocess(query)
    query_emb = embedder.encode([query_clean])[0].tolist()
    memory_bank.append({'query': query_clean, 'query_emb': query_emb, 'llm_output': llm_output})
    with open(MEMORY_BANK_PATH, 'w') as f:
        json.dump(memory_bank, f, ensure_ascii=False, indent=2)

def memory_bank_match(query, threshold=0.9):
    memory_bank = load_memory_bank()
    query_clean = preprocess(query)
    query_emb = embedder.encode([query_clean])[0]
    for item in memory_bank:
        mem_emb = np.array(item['query_emb'])
        sim = np.dot(query_emb, mem_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(mem_emb) + 1e-8)
        if sim > threshold:
            return item['llm_output']
    return None

# ===============================
# Retrieval
## 2.3 Hybrid Retrieval Pipeline

# For each user query:
# Preprocess (clean, lowercase, stopword removal).
# Generate query embedding.
# Retrieve top-N chunks from:
# Dense retrieval (vector similarity).
# Sparse retrieval (BM25).
# Combine results by union or weighted score fusion.
# ===============================
def dense_retrieve(query, top_n=5):
    q_emb = embedder.encode([query], convert_to_numpy=True).astype('float32')
    D, I = faiss_index.search(q_emb, top_n)
    return [(chunk_ids[i], float(D[0][rank])) for rank, i in enumerate(I[0])]

def sparse_retrieve(query, top_n=5):
    q_vec = tfidf_vectorizer.transform([query])
    scores = (tfidf_matrix @ q_vec.T).toarray().squeeze()
    import heapq
    top_idx = heapq.nlargest(top_n, range(len(scores)), scores.take)
    return [(chunk_ids[i], scores[i]) for i in top_idx]

def combine_results(dense, sparse, top_n=5, alpha=0.5):
    scores = {}
    for cid, score in dense:
        scores[cid] = scores.get(cid, 0) + alpha * score
    for cid, score in sparse:
        scores[cid] = scores.get(cid, 0) + (1 - alpha) * score
    sorted_chunks = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
    return [cid for cid, _ in sorted_chunks]

# ===============================
# 2.5 Response Generation

# Use a small, open-source generative model (e.g., DistilGPT2, GPT-2 Small, Llama-2 7B if available).
# Concatenate retrieved passages and user query as input to generate the final answer.
# Limit total input tokens to the model context window.
## PreRequisites:
# Install transformers, sentence-transformers, and faiss-cpu
## Llama must be instlled and running (llama3.2:latest)
# ===============================
def generate_response(user_query, retrieved_chunks, model_name='llama3.2:latest'):
    context = "\n".join([c['chunk'] for c in retrieved_chunks])
    prompt = f"{context}\nUser: {user_query}\nAnswer:"

    payload = {
        "model": model_name,
        "prompt": prompt,
        "options": {"num_predict": 128}
    }

    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        response.raise_for_status()

        output_text = ""
        for line in response.iter_lines():
            if line:
                data = line.decode("utf-8")
                if '"response"' in data:
                    chunk = data.split('"response":"')[-1].split('"')[0]
                    output_text += chunk

        return output_text.strip()

    except requests.exceptions.RequestException as e:
        return f"Error: {e}"
### ===============================
# 2.5 Response Generation for Groq
# Use Groq API to generate responses
# Ensure Groq API is set up and running
# ===============================
def generate_response_groq(user_query, retrieved_chunks, model_name="llama-3.3-70b-versatile"):
    # Build context from retrieved chunks
    #client = groq_client Check
    context = "\n".join([c['chunk'] for c in retrieved_chunks])
    prompt = f"{context}\nUser: {user_query}\nAnswer:"

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=256,   # limit the answer length
            temperature=0.1   # controls creativity
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f" Groq API call failed: {e}")
        return ""
# ===============================
# Guardrails
# 2.6 Guardrail Implementation

# Implement one guardrail:
# Input-side: Validate queries to filter out irrelevant or harmful inputs.
# Output-side: Filter or flag hallucinated or non-factual outputs.
# =============================
def is_query_valid(query):
    if not query or len(query.strip()) < 5:
        return False, "Query too short or empty."
    lowered = query.lower()
    if any(bad in lowered for bad in BANNED_WORDS):
        return False, "Query contains harmful or banned content."
    return True, ""

def is_output_factual(output):
    lowered = output.lower()
    if any(phrase in lowered for phrase in HALLUCINATION_PHRASES):
        return False
    if len(output.strip()) < 10:
        return False
    return True

# =====================================
# Resource loaders (cached)
# =====================================
@st.cache_resource(show_spinner=False)
def load_models():
    global embedder, chunks, chunk_ids, faiss_index, tfidf_vectorizer, tfidf_matrix, qa_pipeline_ft

    print("Loading models and data...")

    # Embedding model
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # Data
    with open(CHUNKS_PATH, 'r') as f:
        chunks = json.load(f)
    chunk_ids = [c['id'] for c in chunks]
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    tfidf_matrix = tfidf_vectorizer.transform([c['chunk'] for c in chunks])

    # Fine-tuned QA model (seq2seq)
    hf_token = get_hf_token()
    cfg = AutoConfig.from_pretrained(FINETUNED_MODEL_PATH, use_auth_token=hf_token)
    tokenizer_ft = AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH, use_auth_token=hf_token)
    model_ft = AutoModelForSeq2SeqLM.from_pretrained(
        FINETUNED_MODEL_PATH,
        use_auth_token=hf_token,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
        device_map=None
    )
    qa_pipeline_ft = pipeline(
        "text2text-generation",
        model=model_ft,
        tokenizer=tokenizer_ft,
        device=-1
    )
    print("Fine-tuned model loaded successfully.")
    print(qa_pipeline_ft("What was SAP's cloud revenue in 2023?", max_new_tokens=64))
    print("All models loaded successfully.")

def ask(user_query, mode):
    valid, reason = is_query_valid(user_query)
    if not valid:
        return jsonify({'error': f'Invalid query: {reason}'}), 400
    start_time = time.time()
    if mode == "fine-tuned":
        print("Using fine-tuned model for query:", user_query)
        try:
            answer = qa_pipeline_ft(f"{user_query}", max_new_tokens=256)[0]["generated_text"]
            print("Fine-tuned model answer:", answer)
        except Exception as e:
            print(f"Error in fine-tuned model inference: {e}")
            return jsonify({'error': f'Fine-tuned model error: {str(e)}'}), 500
        # Confidence: similarity between query and most relevant FAISS chunk
        query_proc = preprocess(user_query)
        dense = dense_retrieve(query_proc, top_n=1)
        max_sim = dense[0][1] if dense else 0


        factual = is_output_factual(answer)
        elapsed_time = round(time.time() - start_time, 3)
        if max_sim < FT_SCOPE_SIM_THRESHOLD or not factual:
            return {
                'answer': "Data not in scope",
                'confidence_score': round(float(max_sim), 3),
                'retrieved_time': elapsed_time,
                'source': 'fine_tuned_model'
            }

        save_to_memory_bank(user_query, answer)
        return {
            'answer': answer,
            'confidence_score': round(float(max_sim), 3),
            'retrieved_time': elapsed_time,
            'source': 'fine_tuned_model'
        }

    elif mode == "rag":
        query_proc = preprocess(user_query)
        dense = dense_retrieve(query_proc)
        sparse = sparse_retrieve(query_proc)
        combined_ids = combine_results(dense, sparse)

        # Compute max similarity from dense retrieval for confidence
        max_sim = max([score for _, score in dense]) if dense else 0

        retrieved_chunks = [c for cid in combined_ids for c in chunks if c['id'] == cid]

        # llm_output = generate_response(user_query, retrieved_chunks)
        llm_output = generate_response_groq(user_query, retrieved_chunks)
        factual = is_output_factual(llm_output)
        elapsed_time = round(time.time() - start_time, 3)

        if max_sim < RAG_SCOPE_SIM_THRESHOLD or not factual:
            return {
                'answer': "Data not in scope",
                'confidence_score': round(float(max_sim), 3),
                'retrieved_time': elapsed_time,
                'source': 'rag_generated'
            }

        save_to_memory_bank(user_query, llm_output)
        return {
            'answer': llm_output,
            'confidence_score': round(float(max_sim), 3),
            'retrieved_time': elapsed_time,
            'source': 'rag_generated',
            'chunks': retrieved_chunks
        }

# ===============================
# App (Front End UX)
# ===============================
# 2.8 Interface Development

# Build a user interface (Streamlit, Gradio, CLI, or GUI).
## Streamlit is used in main/app.py
# ===============================


st.title("RAG & Fine-Tuned QA Interface")

# Add stable keys for widgets
mode = st.radio("Select Mode", ["RAG", "Fine-Tuned"], key="mode_selector")
user_query = st.text_area("Enter your question:", key="user_query")

if st.button("Ask", key="ask_button"):
    api_url = "http://localhost:5050/ask"
    payload = {"query": user_query, "mode": mode.lower()}

    try:
        start = time.time()
        data = ask(user_query, mode.lower())
        client_elapsed = time.time() - start  # client-side total request time

        st.markdown(f"**Answer:** {data.get('answer', '')}")
        st.markdown(f"**Method:** {data.get('source', '')}")
                        # Show numeric confidence score from API
        if "confidence_score" in data:
            st.markdown(f"**Confidence Score:** {data['confidence_score']:.3f} (0 = low, 1 = high)")

        # Show elapsed time returned by API (server-side)
        if "retrieved_time" in data:
            st.markdown(f"**Server Retrieval Time:** {data['retrieved_time']:.3f} seconds")

        # Show total client request time
        st.markdown(f"**Total Response Time:** {client_elapsed:.3f} seconds")
            
        if "factual" in data:
            st.markdown(f"**Factual:** {data['factual']}")

        # Show retrieved chunks only if in RAG mode
        if "chunks" in data:
            st.markdown("**Retrieved Chunks:**")
            st.json(data['chunks'])



        else:
            st.error(f"Error: error")

    except Exception as e:
        st.error(f"Request failed: {e}")

