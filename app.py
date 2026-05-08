import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

goodreads = './books.csv'
df = pd.read_csv(goodreads, on_bad_lines='skip').head(5000)
df['search_data'] = df['title'].astype(str) + " by " + df['authors'].astype(str)
docs = df['search_data'].tolist()

tfidf_vec = TfidfVectorizer(stop_words='english')
tfidf_mat = tfidf_vec.fit_transform(docs)

cv = CountVectorizer(stop_words='english')
bow_mat = cv.fit_transform(docs)

bin_vec = CountVectorizer(binary=True, stop_words='english')
bin_mat = bin_vec.fit_transform(docs)

def search_engine(user_query):
    query = [user_query.lower()]

    q_tfidf = tfidf_vec.transform(query)
    res1 = cosine_similarity(q_tfidf, tfidf_mat).flatten()

    q_bow = cv.transform(query)
    res2 = cosine_similarity(q_bow, bow_mat).flatten()

    q_bin = bin_vec.transform(query)
    res3 = cosine_similarity(q_bin, bin_mat).flatten()

    res4 = []
    q_set = set(query[0].split())
    for d in docs:
        d_set = set(d.lower().split())
        intersect = len(q_set.intersection(d_set))
        union = len(q_set.union(d_set))
        if union == 0:
            res4.append(0)
        else:
            res4.append(intersect / union)

    best_indexes = np.argsort(res1)[::-1][:5]

    out = f"top 5 results for '{user_query}':\n\n"
    for idx in best_indexes:
        out += f"book: {docs[idx]}\n"
        out += f"  > tf-idf: {res1[idx]:.4f}\n"
        out += f"  > bow: {res2[idx]:.4f}\n"
        out += f"  > binary: {res3[idx]:.4f}\n"
        out += f"  > jaccard: {res4[idx]:.4f}\n"
        out += "-"*30 + "\n"
    return out

ui = gr.Interface(
    fn=search_engine,
    inputs=gr.Textbox(lines=1, placeholder="enter book title or author name..."),
    outputs=gr.Textbox(lines=15, label="search results"),
    title="book search engine v1",
    description="enter a query to find books using 4 different nlp algorithms."
)

if __name__ == "__main__":
    ui.launch(server_name="0.0.0.0", server_port=7860)
