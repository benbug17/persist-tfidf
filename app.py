import streamlit as st
import pandas as pd
from nlp_utils import preprocess_text
from db_utils import (
    init_db, insert_document, get_all_documents,
    compute_tfidf, search_documents
)

st.set_page_config(page_title="NLP Document Analyzer", layout="wide")

st.title("📑 NLP Document Analyzer")

conn = init_db()

# Upload document
st.sidebar.header("📤 Upload Document")
uploaded_file = st.sidebar.file_uploader("Choose a .txt file", type="txt")

if uploaded_file:
    raw_text = uploaded_file.read().decode("utf-8")
    clean_text = preprocess_text(raw_text)
    insert_document(conn, uploaded_file.name, clean_text)
    st.sidebar.success(f"Processed: {uploaded_file.name}")

# Process TF-IDF and Search
docs_df = get_all_documents(conn)

if not docs_df.empty:
    vectors, vectorizer = compute_tfidf(docs_df)

    # Show TF-IDF Matrix
    st.subheader("📊 TF-IDF Matrix")
    tfidf_df = pd.DataFrame(vectors.toarray(), columns=vectorizer.get_feature_names_out())
    st.dataframe(tfidf_df)

    # Search Input
    search_input = st.text_input("🔎 Search documents by text")
    if search_input:
        results = search_documents(search_input, docs_df, vectorizer, vectors, preprocess_text)
        if not results.empty:
            st.success("Top matching documents:")
            st.write(results[['name', 'score']])
        else:
            st.warning("No matching documents found.")
else:
    st.warning("Upload documents to start processing.")
