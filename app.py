from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from goose3 import Goose
import sqlite3
import hashlib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import spacy

nlp = spacy.load("en_core_web_trf")
app = FastAPI(title="Article Processing API", description="An API for scraping, processing, and finding similar articles.")
DB_NAME = "articles.db"
SIGMOID_THRESHOLD = 0.6
RELEVANCE_THRESHOLD = 0.6


# Helper Functions
def create_tables():
    """Ensure required database tables are created."""
    conn = sqlite3.connect(DB_NAME)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS articles (
        article_id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT,
        hash TEXT UNIQUE,
        url TEXT
    )
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS article_indices (
        Tag TEXT,
        TFIDF REAL,
        Sigmoid REAL,
        article_id INTEGER,
        FOREIGN KEY(article_id) REFERENCES articles(article_id)
    )
    """)
    conn.close()


def compute_article_hash(article: str) -> str:
    """Generate a unique hash for an article."""
    return hashlib.sha256(article.encode()).hexdigest()

def extract_tfidf_and_sigmoid(article: str) -> dict:
    """Extract TF-IDF values, Sigmoid scores, and mention summary."""
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([article])
    tags = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]

    # Normalize TF-IDF scores and compute sigmoid
    df = pd.DataFrame({"Tag": tags, "TFIDF": tfidf_scores})
    scaler = MinMaxScaler()
    df["TFIDF"] = scaler.fit_transform(df[["TFIDF"]])
    df["Sigmoid"] = 1 / (1 + np.exp(-df["TFIDF"]))

    # Named Entity Recognition (NER) with SpaCy
    doc = nlp(article)
    mentions = {
        "people": [],
        "organizations": [],
        "events": []
    }

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            mentions["people"].append(ent.text)
        elif ent.label_ in ["ORG", "GPE"]:
            mentions["organizations"].append(ent.text)
        elif ent.label_ == "EVENT":
            mentions["events"].append(ent.text)

    return {
        "tags": df.to_dict(orient="records"),
        "mentions": mentions
    }


def scrape_article_content(url: str) -> str:
    """Scrape article content using Goose3."""
    g = Goose()
    article = g.extract(url=url)
    if not article.cleaned_text:
        raise HTTPException(status_code=400, detail="Failed to extract content from the URL.")
    return article.cleaned_text


def store_article_in_db(article: str, tags: list, url: str) -> int:
    """Store article and related data in the database."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    article_hash = compute_article_hash(article)
    cursor.execute("SELECT article_id FROM articles WHERE hash = ?", (article_hash,))
    query_result = cursor.fetchone()

    if query_result:
        conn.close()
        return query_result[0]  # Article already exists

    cursor.execute("INSERT INTO articles (content, hash, url) VALUES (?, ?, ?)", (article, article_hash, url))
    article_id = cursor.lastrowid

    # Insert all tags
    df = pd.DataFrame(tags)
    df["article_id"] = article_id
    df.to_sql("article_indices", conn, if_exists="append", index=False)

    conn.commit()
    conn.close()
    return article_id


def find_similar_articles(tags: list, top_n: int = 5) -> list:
    """Find articles similar to the given tags."""
    conn = sqlite3.connect(DB_NAME)
    placeholders = ",".join(["?" for _ in tags])
    query = f"""
    SELECT a.url, COUNT(DISTINCT ai.Tag) AS tag_count
    FROM article_indices ai
    JOIN articles a ON ai.article_id = a.article_id
    WHERE ai.Tag IN ({placeholders})
    GROUP BY a.url
    HAVING COUNT(DISTINCT ai.Tag) = ?
    ORDER BY tag_count DESC
    LIMIT ?
    """
    params = tags + [len(tags), top_n]
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df["url"].tolist()


# FastAPI Endpoints
@app.on_event("startup")
def startup_event():
    create_tables()


class ArticleRequest(BaseModel):
    url: str


@app.post("/articles/", response_model=dict)
def add_article(request: ArticleRequest):
    """Scrape and process an article from a URL."""
    url = request.url
    article_content = scrape_article_content(url)
    tfidf_result = extract_tfidf_and_sigmoid(article_content)
    article_id = store_article_in_db(article_content, tfidf_result["tags"], url)
    return {"article_id": article_id, "message": "Article stored successfully."}


@app.post("/articles/similar/", response_model=dict)
def find_similars(request: ArticleRequest):
    """Find articles similar to the one at the given URL."""
    url = request.url
    article_content = scrape_article_content(url)
    tfidf_result = extract_tfidf_and_sigmoid(article_content)
    tags = [tag["Tag"] for tag in tfidf_result["tags"]]
    similar_urls = find_similar_articles(tags)
    return {"similar_articles": similar_urls}
