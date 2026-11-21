# Job Recommendation System

![Logo](logo-01.png)

A job search portal built with **Flask** that recommends jobs using **TF-IDF** ranking and an **inverted index**, with fuzzy keyword matching and country filters.

## Features

- Keyword search with fuzzy matching (typos and near-matches still find results)
- Two search modes: **TF-IDF** (cosine similarity ranking) or **inverted index** (fast term lookup)
- Text preprocessing with NLTK stop-word removal and Porter stemming
- Filter results by country
- Paginated results
- Search history saved in the user session

## Tech stack

Python 3 · Flask · pandas · scikit-learn (TF-IDF, cosine similarity) · spaCy · NLTK · fuzzywuzzy · pycountry · HTML/CSS templates

## How it works

1. The job dataset is loaded with pandas and cleaned (irrelevant columns dropped, incomplete rows removed).
2. An **inverted index** is built from stemmed job-title terms (stop words removed) and saved to `inverted_index.txt`.
3. A **TF-IDF matrix** is built from each job's title, skills, qualifications and responsibilities.
4. A user query is tokenized and stemmed, then matched either through the inverted index or by TF-IDF cosine similarity, with optional country filtering.

## Getting started

```bash
git clone https://github.com/KHERBACHEOUSSAMA/Job-Recommendation-System.git
cd Job-Recommendation-System

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Dataset

The app expects a file named `job_descriptions.csv` in the project root (it is not included in the repository because of its size). It must contain at least these columns:
`Job Title`, `Company`, `Country`, `Salary Range`, `skills`, `Qualifications`, `Responsibilities`.

### Run

```bash
python app.py
```

Then open http://127.0.0.1:5000/

## Project structure

```
Job-Recommendation-System/
├─ app.py                               # Flask app: data loading, indexing, search, routes
├─ project.py
├─ templates/
│  └─ index.html                        # Search UI
├─ inverted_index.txt                   # Generated inverted index
├─ job_recommendation_system_report.pdf # Project report
├─ requirements.txt
└─ Procfile                             # Deployment entry point
```

## Author

**Oussama Kherbache**, backend developer
[GitHub](https://github.com/KHERBACHEOUSSAMA) · [LinkedIn](https://www.linkedin.com/in/oussama-kherbache)
