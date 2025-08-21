import streamlit as st
import requests
import time

## To run this app, use the command:
### streamlit run main/app.py
st.title("RAG & Fine-Tuned QA Interface")

# Add stable keys for widgets
mode = st.radio("Select Mode", ["RAG", "Fine-Tuned"], key="mode_selector")
user_query = st.text_area("Enter your question:", key="user_query")

if st.button("Ask", key="ask_button"):
    api_url = "http://localhost:5050/ask"
    payload = {"query": user_query, "mode": mode.lower()}

    try:
        start = time.time()
        response = requests.post(api_url, json=payload, timeout=800)
        client_elapsed = time.time() - start  # client-side total request time

        if response.status_code == 200:
            data = response.json()

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
            st.error(f"Error: {response.json().get('error', 'Unknown error')}")

    except Exception as e:
        st.error(f"Request failed: {e}")