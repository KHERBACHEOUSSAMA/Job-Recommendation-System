import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, session
import pycountry
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os
from fuzzywuzzy import process


# Télécharger les données nécessaires de NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Initialiser le stemmer
stemmer = PorterStemmer()

# Charger le modèle spaCy
nlp = spacy.load("en_core_web_sm")

# Charger le dataset
path = "job_descriptions.csv"
try:
    df = pd.read_csv(path)
    print("Dataset chargé avec succès.")
except FileNotFoundError:
    print(f"Erreur : Le fichier {path} n'existe pas.")
    exit()

# Limiter les lignes pour les tests
df = df[:100000]

# Vérifier que 'Job Title' existe
if 'Job Title' not in df.columns:
    print("Erreur : La colonne 'Job Title' est absente du dataset.")
    exit()

# Supprimer les colonnes non pertinentes
columns_to_drop = ['Job Id', 'latitude', 'longitude', 'Job Portal', 'location', 'Company Profile', 'Job Description']
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Supprimer les lignes avec des valeurs manquantes
df.dropna(subset=['Job Title', 'Company', 'Country', 'Salary Range', 'skills', 'Qualifications', 'Responsibilities'], inplace=True)

# Convertir 'Job Posting Date' au format datetime
if 'Job Posting Date' in df.columns:
    df['Job Posting Date'] = pd.to_datetime(df['Job Posting Date'], errors='coerce')

def build_inverted_index(df):
    inverted_index = defaultdict(list)
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    # Parcourir chaque ligne du DataFrame
    for idx, row in df.iterrows():
        # Tokeniser la colonne 'Job Title'
        terms = str(row['Job Title']).lower().split()
        stemmed_terms = []
        for term in terms:
            if term in stop_words:  # Ignorer les mots vides
                continue

            # Appliquer le stemming
            stemmed_term = stemmer.stem(term)
            stemmed_terms.append(stemmed_term)

        # Ajouter chaque terme stemmé à l'index inversé
        for term in stemmed_terms:
            if idx not in inverted_index[term]:
                inverted_index[term].append(idx)

    return inverted_index

# Build the inverted index
inverted_index = build_inverted_index(df)

print("\nInverted Index Built Successfully!")

def save_inverted_index(inverted_index, filename='inverted_index.txt'):
    with open(filename, 'w', encoding='utf-8') as f:
        for term, postings in inverted_index.items():
            f.write(f"{term}: {postings}\n")
save_inverted_index(inverted_index)

# Vectorisation TF-IDF
def build_tfidf_matrix(df):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    combined_fields = df['Job Title'] + ' ' + df['skills'] + ' ' + df['Qualifications'] + ' ' + df['Responsibilities']
    tfidf_matrix = tfidf_vectorizer.fit_transform(combined_fields)
    return tfidf_vectorizer, tfidf_matrix

tfidf_vectorizer, tfidf_matrix = build_tfidf_matrix(df)

# Fonction de prétraitement avec stemming
def preprocess_query(query):
    # Tokeniser la requête (séparer les mots)
    tokens = nltk.word_tokenize(query.lower())
    
    # Appliquer le stemming sur chaque mot
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    
    # Joindre les tokens transformés en une chaîne de caractères
    return ' '.join(stemmed_tokens)

def fuzzy_search_jobs(query, df, country=None, page=1, results_per_page=10):
    results = []
    # Utilisation de fuzzywuzzy pour effectuer une recherche approximative
    for idx, row in df.iterrows():
        job_title = row['Job Title']
        if country and country.lower() not in row['Country'].lower():
            continue
        # On compare la requête avec le titre du poste
        ratio = process.extractOne(query, [job_title])[1]
        if ratio >= 70:  # seuil de similarité (70% par exemple)
            job = {
                'JobTitle': job_title,
                'Company': row['Company'],
                'Location': row['Country'],
                'SalaryRange': row['Salary Range'],
                'skills': row['skills'],
                'Qualifications': row['Qualifications'],
                'Experience': row['Experience'],
                'Responsibilities': row['Responsibilities'],
                'Benefits': row['Benefits'],
                'Date': row['Job Posting Date'],
                'Contact': row['Contact'],
                'Score': ratio  # On peut aussi afficher le score de la similarité
            }
            results.append(job)

    # Pagination
    total_results = len(results)
    start_idx = (page - 1) * results_per_page
    end_idx = start_idx + results_per_page
    paginated_results = results[start_idx:end_idx]

    return paginated_results, total_results


# Recherche intelligente avec pagination
def search_jobs(query, tfidf_vectorizer, tfidf_matrix, df, country=None, page=1, results_per_page=10):
    # Transformer la requête prétraitée en un vecteur TF-IDF
    query_vector = tfidf_vectorizer.transform([query])
    
    # Calculer les similarités cosinus
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Trier par pertinence décroissante
    top_indices = cosine_similarities.argsort()[::-1]

    results = []
    for idx in top_indices:
        job = df.iloc[idx]
        if country and country.lower() not in job['Country'].lower():
            continue

        results.append({
            'JobTitle': job['Job Title'],
            'Company': job['Company'],
            'Location': job['Country'],
            'SalaryRange': job['Salary Range'],
            'skills': job['skills'],
            'Qualifications': job['Qualifications'],
            'Experience': job['Experience'],
            'Responsibilities': job['Responsibilities'],
            'Benefits': job['Benefits'],
            'Date': job['Job Posting Date'],
            'Contact': job['Contact'],
            'Score': round(cosine_similarities[idx], 2)
        })

    # Pagination
    total_results = len(results)
    start_idx = (page - 1) * results_per_page
    end_idx = start_idx + results_per_page
    paginated_results = results[start_idx:end_idx]
    
    return paginated_results, total_results

def search_with_inverted_index(query, inverted_index, df, country=None, page=1, results_per_page=10):
    query = preprocess_query(query)
    print(query)
    query_terms = query.lower().split()
    matching_docs = set()

    # Collect matching document IDs from the inverted index
    for term in query_terms:
        if term in inverted_index:
            matching_docs.update(inverted_index[term])

    results = []  # Initialisation correcte en dehors de la boucle
    for doc_id in matching_docs:
        job = df.iloc[doc_id]
        if country and country.lower() not in job['Country'].lower():
            continue

        results.append({  # Ajoutez le résultat à la liste
            'JobTitle': job['Job Title'],
            'Company': job['Company'],
            'Location': job['Country'],
            'SalaryRange': job['Salary Range'],
            'skills': job['skills'],
            'Qualifications': job['Qualifications'],
            'Experience': job['Experience'],
            'Responsibilities': job['Responsibilities'],
            'Benefits': job['Benefits'],
            'Contact': job['Contact'],
        })

    # Pagination
    total_results = len(results)
    start_idx = (page - 1) * results_per_page
    end_idx = start_idx + results_per_page
    paginated_results = results[start_idx:end_idx]

    return paginated_results, total_results

# Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Clé secrète générée aléatoirement pour sécuriser la session

@app.route("/", methods=["GET", "POST"])
def index():
    countries = sorted([country.name for country in pycountry.countries])

    # Vérifier et initialiser 'saved_searches' dans la session Flask si ce n'est pas déjà fait
    if 'saved_searches' not in session:
        session['saved_searches'] = []

    if request.method == "POST":
        query = request.form.get("query")
        country = request.form.get("country")
        search = request.form.get("search", "TF-IDF")
        page = int(request.form.get("page", 1))  # Page actuelle
            
        if not query:
            return render_template("index.html", error="Veuillez entrer une requête.", countries=countries)

        # Recherche via TF-IDF ou Inv-ID
        if search == "Inv-ID":
            recommendations, total_results = search_with_inverted_index(query, inverted_index, df, country, page)
            
        if search == "TF-IDF": 
            recommendations, total_results = search_jobs(query, tfidf_vectorizer, tfidf_matrix, df, country, page)
            
        if search == "Fuzzy":
            recommendations, total_results = fuzzy_search_jobs(query, df, country, page)

        # Sauvegarder la recherche dans la session
        session['saved_searches'].append({
            'query': query,
            'country': country,
            'search': search,
        })

        # Debugging: Afficher l'état de la session avant et après l'ajout
        print("Avant ajout à la session:", session.get('saved_searches', []))
        print("Après ajout à la session:", session['saved_searches'])

        # Sauvegarder la session
        session.modified = True  # Indique à Flask que la session a été modifiée

        return render_template(
            "index.html",
            query=query,
            recommendations=recommendations,
            countries=countries,
            page=page,
            search=search,
            total_results=total_results,
            results_per_page=10
        )

    return render_template("index.html", countries=countries, saved_searches=session.get('saved_searches', []))



if __name__ == "__main__":
    app.run(debug=True)
    
#|--------------------------------------------------------------------------|
#| Built by : ZEGHLACHE Mohamed El Amin, KHERBACHE Oussama, MEHDAOUI Yacine |
#|--------------------------------------------------------------------------|;