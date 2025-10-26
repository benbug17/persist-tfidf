import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle, os

VECTORIZER_PATH = "data/vectorizer.pkl"
VECTORS_PATH = "data/vectors.npz"  # you can use scipy.sparse.save_npz

def save_index(vectorizer, vectors):
    os.makedirs("data", exist_ok=True)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    from scipy import sparse
    sparse.save_npz(VECTORS_PATH, vectors)

def load_index():
    import pickle
    from scipy import sparse
    if os.path.exists(VECTORIZER_PATH) and os.path.exists(VECTORS_PATH):
        with open(VECTORIZER_PATH, "rb") as f:
            vec = pickle.load(f)
        v = sparse.load_npz(VECTORS_PATH)
        return vec, v
    return None, None

def init_db():
    conn = sqlite3.connect("data/tfidf.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            content TEXT
        )
    ''')
    conn.commit()
    return conn

def insert_document(conn, name, content):
    cursor = conn.cursor()
    cursor.execute("INSERT INTO documents (name, content) VALUES (?, ?)", (name, content))
    conn.commit()

def get_all_documents(conn):
    return pd.read_sql_query("SELECT * FROM documents", conn)

def compute_tfidf(docs_df):
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform(docs_df['content'])
    return vectors, tfidf

def search_documents(query, docs_df, vectorizer, vectors, preprocess_fn):
    query_vec = vectorizer.transform([preprocess_fn(query)])
    sim_scores = cosine_similarity(query_vec, vectors).flatten()
    top_idx = sim_scores.argsort()[::-1]
    results = docs_df.iloc[top_idx].copy()
    results['score'] = sim_scores[top_idx]
    return results[results['score'] > 0]
