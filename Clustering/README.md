# Clustering

Dette er et kort teknisk notat om clustering i maskinlæring. Fokus er på hvordan clustering typisk brukes i Python med `pandas`, `matplotlib`, `seaborn` og `scikit-learn`.

Clustering er en metode innen **unsupervised learning**, der modellen forsøker å gruppere datapunkter uten en kjent target-verdi. I stedet for å predikere en fasit, leter modellen etter struktur i dataene.

Typiske bruksområder:

* Kundesegmentering
* Gruppering av produkter, sanger, tekster eller brukere
* Anomalideteksjon
* Utforskende dataanalyse
* Finne mønstre i datasett uten labels

---

## Typiske biblioteker

```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
```

---

## Typisk workflow

```text
1. Last inn data
2. Undersøk data
3. Fjern eller håndter manglende verdier
4. Velg relevante numeriske features
5. Skaler data
6. Test clustering-modell
7. Evaluer med elbow method / silhouette score
8. Visualiser cluster
9. Tolk clusterne med groupby
```

---

## Laste inn og undersøke data

```python
df = pd.read_csv("data.csv")

df.head()
df.info()
df.describe()
df.isnull().sum()
```

Eksempel med musikkdata:

```python
df = pd.read_csv("../data/nigerian-songs.csv")

df.head()
```

---

## Enkel datarensing

Fjern manglende verdier:

```python
df = df.dropna()
```

Filtrer bort rader som ikke er nyttige:

```python
df = df[df["artist_top_genre"] != "Missing"]
df = df[df["popularity"] > 0]
```

Filtrer til relevante grupper:

```python
selected_genres = ["afro dancehall", "afropop", "nigerian pop"]

df = df[df["artist_top_genre"].isin(selected_genres)]
```

---

## Velge features

Clustering fungerer best når man velger relevante numeriske variabler.

```python
features = [
    "popularity",
    "danceability",
    "acousticness",
    "loudness",
    "energy"
]

X = df[features].copy()
```

Unngå å ta med kolonner som:

* ID-er
* Navn
* Tekstkolonner uten encoding
* Irrelevante features
* Features med ekstrem skala uten scaling

---

## Scaling

Scaling er veldig viktig i clustering fordi mange metoder bruker avstand mellom datapunkter.

Eksempel:

```text
danceability: 0 til 1
energy: 0 til 1
length: 90 000 til 500 000
```

Hvis data ikke skaleres, kan store variabler dominere modellen.

```python
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

Eventuelt tilbake til DataFrame:

```python
X_scaled = pd.DataFrame(
    X_scaled,
    columns=features,
    index=X.index
)
```

---

# K-Means

K-Means er den vanligste clustering-metoden.

Den deler dataene inn i `k` cluster ved å plassere datapunkter nærmeste centroid.

Kort forklart:

```text
1. Velg antall cluster k
2. Initialiser centroider
3. Tildel hvert punkt til nærmeste centroid
4. Oppdater centroidene
5. Gjenta til modellen stabiliserer seg
```

---

## Enkel K-Means-modell

```python
from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=3,
    init="k-means++",
    random_state=42,
    n_init=10
)

labels = kmeans.fit_predict(X_scaled)

df["cluster"] = labels
```

Se fordeling av cluster:

```python
df["cluster"].value_counts()
```

---

## Viktige K-Means-parametere

```python
KMeans(
    n_clusters=3,
    init="k-means++",
    random_state=42,
    n_init=10
)
```

Forklaring:

```text
n_clusters     Antall cluster modellen skal lage
init           Hvordan start-centroidene velges
random_state   Gjør resultatet reproduserbart
n_init         Hvor mange initialiseringer modellen tester
```

`k-means++` er vanlig standard fordi den velger bedre startpunkter enn helt tilfeldig initialisering.

---

## Elbow method

Elbow method brukes for å finne et fornuftig antall cluster.

```python
wcss = []

for k in range(1, 11):
    model = KMeans(
        n_clusters=k,
        init="k-means++",
        random_state=42,
        n_init=10
    )
    
    model.fit(X_scaled)
    wcss.append(model.inertia_)
```

Plot:

```python
plt.figure(figsize=(10, 5))

sns.lineplot(
    x=range(1, 11),
    y=wcss,
    marker="o"
)

plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS / Inertia")
plt.show()
```

Tolkning:

```text
Se etter punktet der kurven begynner å flate ut.
Dette punktet kan være et fornuftig valg for k.
```

---

## Silhouette score

Silhouette score måler hvor godt separerte clusterne er.

Verdien går fra `-1` til `1`.

```text
Nær 1   Tydelige og godt separerte cluster
Nær 0   Overlapp mellom cluster
Under 0 Mulig feil clusterplassering
```

```python
score = silhouette_score(X_scaled, labels)

print(score)
```

Teste flere verdier for `k`:

```python
scores = []

for k in range(2, 11):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)

    scores.append({
        "k": k,
        "silhouette": score,
        "inertia": model.inertia_
    })

scores_df = pd.DataFrame(scores)

scores_df
```

Plot:

```python
plt.figure(figsize=(10, 5))

sns.lineplot(
    data=scores_df,
    x="k",
    y="silhouette",
    marker="o"
)

plt.title("Silhouette score per k")
plt.show()
```

---

## Samlet K-Means-oppsett

```python
features = [
    "popularity",
    "danceability",
    "acousticness",
    "loudness",
    "energy"
]

X = df[features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=3,
    init="k-means++",
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans.fit_predict(X_scaled)

score = silhouette_score(X_scaled, df["cluster"])

print(f"Silhouette score: {score:.3f}")
```

---

# Visualisering

## Scatterplot av to features

```python
plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="popularity",
    y="danceability",
    hue="cluster",
    palette="viridis"
)

plt.title("K-Means clusters")
plt.show()
```

Dette fungerer bra hvis man vil se clusterne i to dimensjoner.

Problemet er at modellen ofte er trent på flere features enn bare de to som plottes.

---

## PCA for 2D-visualisering

Hvis modellen er trent på flere features, kan PCA brukes for å visualisere clusterne i to dimensjoner.

```python
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

df["pca_1"] = X_pca[:, 0]
df["pca_2"] = X_pca[:, 1]
```

Plot:

```python
plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="pca_1",
    y="pca_2",
    hue="cluster",
    palette="viridis"
)

plt.title("Clusters visualized with PCA")
plt.show()
```

---

## Tolke cluster

Etter clustering bør man se på gjennomsnittsverdier per cluster.

```python
cluster_summary = df.groupby("cluster")[features].mean()

cluster_summary
```

Eventuelt median:

```python
df.groupby("cluster")[features].median()
```

Antall per cluster:

```python
df["cluster"].value_counts()
```

Eksempel på tolkning:

```text
Cluster 0: høy danceability, høy energy, middels popularity
Cluster 1: lav acousticness, høy loudness, høy popularity
Cluster 2: lav popularity, lav energy, høy acousticness
```

---

## Visualisere cluster-profiler

```python
cluster_summary = df.groupby("cluster")[features].mean()

cluster_summary.T.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Mean feature values per cluster")
plt.xlabel("Feature")
plt.ylabel("Mean value")
plt.xticks(rotation=45)
plt.show()
```

---

# Alternative clustering-metoder

K-Means er ofte første metode man tester, men andre metoder kan være relevante.

## DBSCAN

DBSCAN grupperer punkter basert på tetthet.

Den trenger ikke `n_clusters`.

```python
dbscan = DBSCAN(
    eps=0.8,
    min_samples=5
)

df["dbscan_cluster"] = dbscan.fit_predict(X_scaled)
```

Sjekk cluster:

```python
df["dbscan_cluster"].value_counts()
```

Merk:

```text
Cluster -1 betyr støy/outliers.
```

Visualisering:

```python
sns.scatterplot(
    data=df,
    x="pca_1",
    y="pca_2",
    hue="dbscan_cluster",
    palette="viridis"
)

plt.title("DBSCAN clusters")
plt.show()
```

---

## Agglomerative Clustering

Hierarkisk clustering.

```python
agg = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)

df["agg_cluster"] = agg.fit_predict(X_scaled)
```

Visualisering:

```python
sns.scatterplot(
    data=df,
    x="pca_1",
    y="pca_2",
    hue="agg_cluster",
    palette="viridis"
)

plt.title("Agglomerative Clustering")
plt.show()
```

---

## Gaussian Mixture Model

GMM gir en mer probabilistisk clustering.

```python
gmm = GaussianMixture(
    n_components=3,
    random_state=42
)

df["gmm_cluster"] = gmm.fit_predict(X_scaled)
```

Sannsynlighet for cluster:

```python
probs = gmm.predict_proba(X_scaled)

probs[:5]
```

---

# Sammenligne flere modeller

Flere clustering-modeller har ganske likt oppsett:

```python
models = {
    "kmeans": KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    ),
    "agglomerative": AgglomerativeClustering(
        n_clusters=3
    ),
    "gmm": GaussianMixture(
        n_components=3,
        random_state=42
    )
}

results = []

for name, model in models.items():
    labels = model.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)

    results.append({
        "model": name,
        "silhouette": score,
        "n_clusters": len(set(labels))
    })

results_df = pd.DataFrame(results)

results_df
```

DBSCAN bør ofte håndteres separat fordi den kan lage `-1` for støy.

```python
dbscan = DBSCAN(
    eps=0.8,
    min_samples=5
)

labels = dbscan.fit_predict(X_scaled)

mask = labels != -1

if len(set(labels[mask])) > 1:
    score = silhouette_score(X_scaled[mask], labels[mask])
else:
    score = None

print(score)
```

---

# Pipeline

Man kan samle scaling og modell i en pipeline.

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    ))
])

df["cluster"] = pipeline.fit_predict(X)
```

Dette er ryddigere og reduserer risikoen for å glemme scaling.

---

# Komplett kompakt eksempel

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Load data
df = pd.read_csv("../data/nigerian-songs.csv")

# Basic cleaning
df = df[df["artist_top_genre"] != "Missing"]
df = df[df["popularity"] > 0]

# Select features
features = [
    "popularity",
    "danceability",
    "acousticness",
    "loudness",
    "energy"
]

X = df[features].copy()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method
wcss = []

for k in range(1, 11):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    
    model.fit(X_scaled)
    wcss.append(model.inertia_)

plt.figure(figsize=(10, 5))
sns.lineplot(x=range(1, 11), y=wcss, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS / Inertia")
plt.show()

# Final K-Means model
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans.fit_predict(X_scaled)

# Evaluation
score = silhouette_score(X_scaled, df["cluster"])
print(f"Silhouette score: {score:.3f}")

# Cluster interpretation
cluster_summary = df.groupby("cluster")[features].mean()
print(cluster_summary)

# PCA visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df["pca_1"] = X_pca[:, 0]
df["pca_2"] = X_pca[:, 1]

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="pca_1",
    y="pca_2",
    hue="cluster",
    palette="viridis"
)

plt.title("K-Means Clusters with PCA")
plt.show()
```

---

# Kort oppsummering

Clustering brukes når man vil finne grupper i data uten en kjent target-verdi.

Den vanligste metoden er K-Means, der man velger antall cluster med `n_clusters`.

Typisk teknisk oppsett:

```text
Velg features
Skaler data
Test flere k-verdier
Bruk elbow method og silhouette score
Tren endelig modell
Visualiser med scatterplot eller PCA
Tolk cluster med groupby
```

Det viktigste i clustering er ikke bare hvilken algoritme man bruker, men hvilke features man velger, om dataene er skalert, og om clusterne faktisk gir mening etterpå.