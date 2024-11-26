# Article Processing API

A FastAPI-based API to scrape, process, and find similar articles.

## Features
- Scrape article content from a URL using `Goose3`.
- Extract TF-IDF tags for each article.
- Store articles in a SQLite database.
- Retrieve similar articles based on TF-IDF tags.

## Installation

- python3 -m spacy download en_core_web_trf
- uvicorn app:app --reload
- Swagger UI: http://127.0.0.1:8000/docs

---

## **Endpoints**

### **POST `/articles/`**
**Description**: Scrape and process an article from a URL.

**Input**:
```json
{
  "url": "https://example.com/article"
}
```

**Response**:
```json
{
  "article_id": 1,
  "message": "Article stored successfully."
}
```

---

### **POST `/articles/similar/`**
**Description**: Find articles similar to the one at the given URL.

**Input**:
```json
{
  "url": "https://example.com/article"
}
```

**Response**:
```json
{
  "similar_articles": [
    "https://example.com/similar1",
    "https://example.com/similar2"
  ]
}
```

--- 
