# Job Recommendation System

![Logo](logo-01.png)

A job search portal developed with **Flask**, using **TF-IDF** and an **Inverted Index** to recommend jobs.

## Features

- Keyword search with **fuzzy matching**.
- Search using **TF-IDF** or **Inverted Index**.
- Filter by **country**.
- Pagination of results.
- Saves search history in the user session.

## Project Structure

recomendedsystem
├─ app.py
├─ inverted_index.txt
├─ job_descriptions.csv
├─ job_recommendation_system_report.pdf
├─ logo.png
├─ project.py
├─ templates/
│ └─ index.html
└─ z.py

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/recomendedsystem.git
cd recomendedsystem
Install dependencies:

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Run the application:

```bash
python app.py
```

Open in your browser: http://127.0.0.1:5000/
Technologies Used

Python 3

Flask

Spacy

TF-IDF & Inverted Index

HTML / CSS (templates)

Author

Oussama Kherbache – Lead Developer
