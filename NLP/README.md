# NLP - Natural Language Processing

Dette er en kompakt teknisk README for repetisjon av NLP. Fokuset er på typiske imports, preprocessing, vectorization, sentimentanalyse, klassifikasjon og vanlige pipeline-oppsett i Python.

---

## 1. Typiske biblioteker

```python
import pandas as pd
import numpy as np
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from textblob import TextBlob

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
```

Installer ved behov:

```bash
pip install pandas numpy nltk textblob scikit-learn
python -m textblob.download_corpora
```

For NLTK:

```python
nltk.download("stopwords")
nltk.download("vader_lexicon")
```

---

## 2. Typisk NLP-pipeline

```text
Raw text
→ cleaning
→ tokenization
→ stop word removal / lemmatization
→ vectorization
→ model training
→ evaluation
```

I praksis er det vanligste oppsettet:

```text
Text column
→ TfidfVectorizer
→ LogisticRegression / NaiveBayes / LinearSVC
→ classification_report
```

---

## 3. Laste inn tekstdata

```python
df = pd.read_csv("data/reviews.csv")

df.head()
df.info()
df.shape
df.isna().sum()
```

Eksempel med tekst og label:

```python
X = df["review_text"]
y = df["label"]
```

Sjekk fordeling av labels:

```python
df["label"].value_counts()
df["label"].value_counts(normalize=True)
```

---

## 4. Enkel tekstvask

```python
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

Bruk:

```python
df["clean_text"] = df["review_text"].apply(clean_text)
```

Viktig: Ikke vask for aggressivt. Ved sentimentanalyse kan ord som `not`, `never` og `no` være viktige.

---

## 5. Stop words

Stop words er vanlige ord som ofte fjernes, for eksempel `the`, `and`, `is`.

```python
stop_words = set(stopwords.words("english"))

def remove_stopwords(text):
    tokens = str(text).split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)
```

Bruk:

```python
df["clean_text"] = df["clean_text"].apply(remove_stopwords)
```

Ofte brukt ved:

```text
word frequency
bag-of-words
TF-IDF
klassiske ML-modeller
```

Vanligvis ikke nødvendig ved transformer-modeller.

---

## 6. Tokenization

Tokenization betyr å splitte tekst i ord eller tokens.

```python
text = "The hotel was clean and modern."

tokens = text.lower().split()
print(tokens)
```

Mer robust med regex:

```python
tokens = re.findall(r"\b\w+\b", text.lower())
print(tokens)
```

---

## 7. Word frequency

Brukes for å se hvilke ord som går igjen i datasettet.

```python
from collections import Counter

all_words = " ".join(df["clean_text"]).split()
word_freq = Counter(all_words)

word_freq.most_common(20)
```

Som DataFrame:

```python
freq_df = pd.DataFrame(
    word_freq.most_common(30),
    columns=["word", "count"]
)

freq_df
```

---

## 8. Bag-of-Words

Bag-of-Words gjør tekst om til ordtellinger.

```python
vectorizer = CountVectorizer(
    max_features=5000,
    stop_words="english"
)

X = vectorizer.fit_transform(df["clean_text"])

print(X.shape)
```

Se ordene/features:

```python
vectorizer.get_feature_names_out()[:20]
```

Bag-of-Words er enkelt, men mister ordrekkefølge og kontekst.

---

## 9. N-grams

N-grams tar med sekvenser av ord.

```python
vectorizer = CountVectorizer(
    ngram_range=(1, 2),
    max_features=5000,
    stop_words="english"
)

X = vectorizer.fit_transform(df["clean_text"])
```

Eksempler:

```text
unigram: "good"
bigram: "not good"
trigram: "not very good"
```

N-grams er nyttig når fraser betyr mer enn enkeltord.

---

## 10. TF-IDF

TF-IDF er ofte standardvalg i klassisk NLP.

```python
tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    stop_words="english",
    min_df=3,
    max_df=0.9
)

X = tfidf.fit_transform(df["clean_text"])
```

Viktige parametere:

```text
max_features  → maks antall ord/features
ngram_range   → unigrams, bigrams osv.
min_df         → ignorer svært sjeldne ord
max_df         → ignorer svært vanlige ord
stop_words     → fjern vanlige ord
```

TF-IDF brukes ofte sammen med:

```text
LogisticRegression
MultinomialNB
LinearSVC
```

---

## 11. Train/test split

```python
X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Bruk `stratify=y` når klassene er ubalanserte.

---

## 12. Standardmodell 1: TF-IDF + Logistic Regression

Dette er ofte en veldig god baseline.

```python
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=3,
        max_df=0.9
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
```

Brukes til:

```text
sentiment classification
spam detection
topic classification
review classification
```

---

## 13. Standardmodell 2: CountVectorizer + Naive Bayes

Naive Bayes fungerer ofte bra på tekstklassifikasjon.

```python
model = Pipeline([
    ("vectorizer", CountVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words="english"
    )),
    ("clf", MultinomialNB())
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
```

Fordeler:

```text
rask
enkel
god baseline
fungerer ofte bra på tekst
```

---

## 14. Standardmodell 3: TF-IDF + Linear SVC

LinearSVC er ofte sterk på høy-dimensjonale tekstdata.

```python
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words="english"
    )),
    ("clf", LinearSVC(class_weight="balanced"))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
```

Ofte et godt alternativ til Logistic Regression.

---

## 15. Evaluering

```python
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

Ikke stol blindt på accuracy hvis klassene er ubalanserte.

Se spesielt på:

```text
precision
recall
f1-score
support
```

---

## 16. Sentimentanalyse med TextBlob

TextBlob gir enkel polarity og subjectivity.

```python
text = "The hotel was clean and the staff were friendly."

blob = TextBlob(text)

print(blob.sentiment)
```

Output:

```text
Sentiment(polarity=0.45, subjectivity=0.65)
```

Tolkning:

```text
polarity:
-1 = negativ
 0 = nøytral
 1 = positiv

subjectivity:
0 = objektiv
1 = subjektiv
```

Bruk på DataFrame:

```python
df["polarity"] = df["review_text"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
df["subjectivity"] = df["review_text"].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
```

---

## 17. Sentimentanalyse med VADER

VADER er vanlig for korte tekster, reviews og sosiale medier.

```python
vader = SentimentIntensityAnalyzer()

text = "The room was amazing, but the breakfast was terrible."

vader.polarity_scores(text)
```

Output:

```python
{
    "neg": 0.269,
    "neu": 0.477,
    "pos": 0.254,
    "compound": -0.0516
}
```

Bruk compound score:

```python
def calc_sentiment(text):
    text = str(text)

    if text in ["No Negative", "No Positive", ""]:
        return 0

    return vader.polarity_scores(text)["compound"]
```

På DataFrame:

```python
df["sentiment"] = df["review_text"].apply(calc_sentiment)
```

Lag sentiment-label:

```python
def sentiment_label(score):
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

df["sentiment_label"] = df["sentiment"].apply(sentiment_label)
```

---

## 18. Kombinere review-kolonner

I review-datasett har man ofte positive og negative tekstfelt.

```python
df["review_text"] = (
    df["Positive_Review"].fillna("") + " " +
    df["Negative_Review"].fillna("")
)
```

Eksempel med hotel reviews:

```python
df["Positive_Review"] = df["Positive_Review"].replace("No Positive", "")
df["Negative_Review"] = df["Negative_Review"].replace("No Negative", "")

df["review_text"] = df["Positive_Review"] + " " + df["Negative_Review"]
```

---

## 19. Lage label fra score

Hvis man har review score, kan man lage labels selv.

```python
def create_label(score):
    if score >= 7:
        return "positive"
    elif score <= 5:
        return "negative"
    else:
        return "neutral"

df["label"] = df["Reviewer_Score"].apply(create_label)
```

Eventuelt binært:

```python
df = df[df["Reviewer_Score"] != 6]

df["label"] = df["Reviewer_Score"].apply(
    lambda score: 1 if score >= 7 else 0
)
```

---

## 20. Feature engineering fra tekst

Tekstlengde:

```python
df["text_length"] = df["review_text"].str.len()
df["word_count"] = df["review_text"].str.split().str.len()
```

Har positiv/negativ review:

```python
df["has_positive_review"] = (df["Positive_Review"] != "").astype(int)
df["has_negative_review"] = (df["Negative_Review"] != "").astype(int)
```

Sentiment-features:

```python
df["positive_sentiment"] = df["Positive_Review"].apply(calc_sentiment)
df["negative_sentiment"] = df["Negative_Review"].apply(calc_sentiment)
```

---

## 21. Noun phrases med TextBlob

Noun phrases brukes for å hente ut viktige temaer/objekter fra tekst.

```python
text = "The hotel room had a beautiful city view and a comfortable bed."

blob = TextBlob(text)

print(blob.noun_phrases)
```

Eksempel:

```text
['hotel room', 'beautiful city view', 'comfortable bed']
```

Brukes til:

```text
temaer i reviews
chatbots
søkeord
feature extraction
```

---

## 22. POS-tagging

Part-of-speech tagging merker ord med grammatisk rolle.

```python
blob = TextBlob("The hotel was clean and modern.")

print(blob.tags)
```

Eksempel:

```text
[('The', 'DT'), ('hotel', 'NN'), ('was', 'VBD'), ('clean', 'JJ')]
```

Dette brukes sjeldnere i enkle ML-pipelines, men er nyttig for mer regelbasert NLP og språkanalyse.

---

## 23. Tags / kategoriske tekstfelt

Noen datasett har tags lagret som tekst.

Eksempel:

```text
[' Leisure trip ', ' Couple ', ' Double Room ', ' Stayed 2 nights ']
```

Rens:

```python
df["Tags"] = df["Tags"].str.strip("[']")
df["Tags"] = df["Tags"].str.replace(" ', '", ",", regex=False)
```

Lag binære kolonner:

```python
df["Leisure_trip"] = df["Tags"].apply(lambda x: 1 if "Leisure trip" in x else 0)
df["Business_trip"] = df["Tags"].apply(lambda x: 1 if "Business trip" in x else 0)
df["Couple"] = df["Tags"].apply(lambda x: 1 if "Couple" in x else 0)
df["Solo_traveler"] = df["Tags"].apply(lambda x: 1 if "Solo traveler" in x else 0)
df["Family"] = df["Tags"].apply(lambda x: 1 if "Family" in x else 0)
```

Dette gjør tekstbaserte tags om til numeriske features.

---

## 24. Cosine similarity

Cosine similarity brukes for å måle likhet mellom tekster etter vectorization.

```python
from sklearn.metrics.pairwise import cosine_similarity

tfidf = TfidfVectorizer(stop_words="english")

X = tfidf.fit_transform(df["clean_text"])

similarity_matrix = cosine_similarity(X[:5], X[:5])

print(similarity_matrix)
```

Brukes til:

```text
semantisk søk
recommendation systems
duplicate detection
dokumentlikhet
```

---

## 25. Clustering av tekst

Tekst kan clusteres etter TF-IDF eller embeddings.

```python
from sklearn.cluster import KMeans

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X = tfidf.fit_transform(df["clean_text"])

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init="auto"
)

df["cluster"] = kmeans.fit_predict(X)
```

Se viktigste ord per cluster:

```python
terms = tfidf.get_feature_names_out()
centers = kmeans.cluster_centers_

for i in range(5):
    top_terms = centers[i].argsort()[-10:][::-1]
    print(f"Cluster {i}:")
    print([terms[j] for j in top_terms])
```

---

## 26. Topic modelling med NMF

```python
from sklearn.decomposition import NMF

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X = tfidf.fit_transform(df["clean_text"])

nmf = NMF(
    n_components=5,
    random_state=42
)

topics = nmf.fit_transform(X)

words = tfidf.get_feature_names_out()

for topic_idx, topic in enumerate(nmf.components_):
    top_words = topic.argsort()[-10:][::-1]
    print(f"Topic {topic_idx}:")
    print([words[i] for i in top_words])
```

Brukes for å finne temaer i store tekstmengder.

---

## 27. Embeddings

Embeddings gjør tekst om til numeriske vektorer der semantisk like tekster ligger nærmere hverandre.

Vanlige typer:

```text
Word2Vec
GloVe
FastText
BERT
Sentence-BERT
OpenAI embeddings
```

Brukes til:

```text
semantic search
RAG
clustering
classification
recommendation
similarity search
```

Klassisk NLP bruker ofte TF-IDF. Moderne NLP bruker ofte embeddings.

---

## 28. Komplett baseline for tekstklassifikasjon

Dette er et godt standardoppsett å starte med:

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df = pd.read_csv("data/dataset.csv")

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000,
        min_df=3,
        max_df=0.9
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
```

---

## 29. Sammenligne flere modeller raskt

```python
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Naive Bayes": MultinomialNB(),
    "Linear SVC": LinearSVC(class_weight="balanced")
}

for name, clf in models.items():
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words="english"
        )),
        ("clf", clf)
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n" + name)
    print(classification_report(y_test, y_pred))
```

Dette er nyttig når man raskt vil teste flere klassiske NLP-modeller.

---

## 30. Vanlige feil

```text
1. For aggressiv tekstvask
Kan fjerne ord som faktisk betyr noe.

2. Fjerne negasjoner
"not good" kan bli "good", som ødelegger sentiment.

3. Data leakage
Ikke fit vectorizer på hele datasettet før train/test split.

4. Bare bruke accuracy
Ved ubalanserte klasser bør man bruke precision, recall og f1-score.

5. Ikke lage baseline
Start med TF-IDF + Logistic Regression før mer avanserte modeller.

6. Stole blindt på sentimentmodeller
Sarkasme, ironi og kontekst kan gi feil sentiment.
```

---

## 31. Når bruker man hva?

```text
Word frequency
→ rask oversikt over hvilke ord som dominerer

N-grams
→ når fraser som "not good" eller "very clean" er viktige

Bag-of-Words
→ enkel baseline med ordtelling

TF-IDF
→ standardvalg for klassisk tekstklassifikasjon

Naive Bayes
→ rask baseline for tekstklassifikasjon

Logistic Regression
→ sterk og tolkbar baseline

LinearSVC
→ ofte sterk på høy-dimensjonale tekstdata

VADER/TextBlob
→ enkel sentimentanalyse uten egen trening

Embeddings
→ når semantisk likhet og kontekst er viktig

Transformers
→ når man trenger høyere ytelse og bedre språkforståelse
```

---

## 32. Kort oppsummert

Den viktigste praktiske NLP-pipelinen å huske er:

```text
Text
→ clean text
→ TF-IDF
→ Logistic Regression / Naive Bayes / LinearSVC
→ classification_report
```

For sentimentanalyse:

```text
Text
→ TextBlob eller VADER
→ polarity / compound score
→ sentiment label
```

For tekstanalyse uten labels:

```text
Text
→ TF-IDF eller embeddings
→ clustering / topic modelling / similarity
```

Som baseline i de fleste NLP-prosjekter:

```python
Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
    ("clf", LogisticRegression(max_iter=1000))
])
```