# Clustering i maskinlæring

Dette dokumentet er et teknisk sammendrag av clustering i maskinlæring. Målet er å ha en samlet README som kan brukes til repetisjon senere, spesielt med fokus på hvordan clustering brukes i praksis med Python, pandas, seaborn, matplotlib og scikit-learn.

Clustering er en metode innen **unsupervised learning**. Det betyr at modellen ikke får fasitsvar i form av labels. I stedet forsøker algoritmen å finne struktur i dataene selv, basert på likhet mellom observasjoner.

Typiske bruksområder:

* Kundesegmentering
* Anomalideteksjon
* Gruppering av dokumenter
* Gruppering av sanger, filmer eller produkter
* Utforskende dataanalyse
* Komprimering av store datasett til mer forståelige grupper
* Foranalyse før man bygger andre modeller

Clustering handler ikke om å predikere en kjent målvariabel. Det handler om å undersøke om datapunkter naturlig kan deles inn i grupper.

---

## 1. Hva er clustering?

Clustering betyr at vi grupperer observasjoner som ligner på hverandre.

En observasjon kan for eksempel være:

* En kunde
* En sang
* Et bilde
* En transaksjon
* En tekst
* En bolig
* En medisinsk måling

Hver observasjon beskrives av features.

Eksempel med musikkdata:

```python
features = [
    "popularity",
    "danceability",
    "acousticness",
    "loudness",
    "energy"
]
```

Hvis to sanger har lignende verdier for disse variablene, kan en clustering-algoritme plassere dem i samme cluster.

---

## 2. Clustering vs. supervised learning

I supervised learning har vi vanligvis:

```text
X = inputvariabler
y = fasit / target / label
```

Eksempel:

```text
Input: areal, beliggenhet, byggeår
Target: boligpris
```

Modellen lærer sammenhengen mellom `X` og `y`.

I clustering har vi bare:

```text
X = inputvariabler
```

Det finnes ingen fasit som modellen lærer fra.

Eksempel:

```text
Input: popularity, danceability, acousticness, energy, loudness
Target: ingen target
```

Modellen forsøker selv å finne mønstre i dataene.

---

## 3. Hvorfor bruke clustering?

Clustering brukes ofte når man ikke vet nøyaktig hvilke grupper som finnes i dataene.

Eksempler:

### Kundesegmentering

Man kan gruppere kunder basert på:

* Kjøpshistorikk
* Bruksmønster
* Prisfølsomhet
* Produktinteresse
* Aktivitet over tid

Resultatet kan bli segmenter som:

```text
Cluster 0: prisbevisste kunder
Cluster 1: lojale premium-kunder
Cluster 2: inaktive kunder
```

### Anomalideteksjon

Hvis de fleste datapunktene ligger tett i cluster, kan punkter som ligger langt unna tolkes som mulige avvik.

Eksempel:

```text
Vanlige transaksjoner = tett cluster
Uvanlige transaksjoner = outliers
```

Dette kan brukes innen svindeldeteksjon, cybersikkerhet eller overvåkning av sensordata.

### Utforskende analyse

Clustering kan brukes før man vet hva man leter etter.

Man kan spørre:

* Finnes det naturlige grupper i dataene?
* Er noen datapunkter tydelig annerledes?
* Er labels i datasettet egentlig godt separert?
* Er dataene egnet for videre modellering?

---

## 4. Viktige begreper

### Cluster

Et cluster er en gruppe datapunkter som algoritmen mener ligner på hverandre.

```text
Cluster 0
Cluster 1
Cluster 2
```

Cluster-ID-en er ikke en ekte label. Den er bare et nummer algoritmen gir til gruppen.

---

### Centroid

En centroid er sentrumspunktet til et cluster.

I K-Means beregnes centroiden som gjennomsnittet av punktene i clusteret.

For et todimensjonalt datasett kan en centroid være:

```text
centroid = gjennomsnittlig x-verdi og gjennomsnittlig y-verdi
```

K-Means prøver å plassere centroidene slik at punktene i hvert cluster ligger så nær sin centroid som mulig.

---

### Distance

Clustering er ofte basert på avstand.

Hvis to datapunkter ligger nær hverandre i feature space, regnes de som mer like.

Vanlige avstandsmål:

* Euclidean distance
* Manhattan distance
* Cosine distance
* Minkowski distance

K-Means bruker vanligvis Euclidean distance.

---

### Inertia

Inertia måler hvor kompakte clusterne er.

Mer teknisk er inertia summen av kvadrerte avstander fra hvert punkt til sin nærmeste centroid.

Lav inertia betyr at punktene ligger tett rundt centroidene.

Men lav inertia alene betyr ikke nødvendigvis at modellen er god. Hvis man øker antall cluster, vil inertia nesten alltid bli lavere. Derfor må inertia tolkes sammen med for eksempel elbow method.

---

### WCSS

WCSS står for:

```text
Within-Cluster Sum of Squares
```

Det er i praksis det samme prinsippet som inertia i K-Means.

WCSS måler hvor mye variasjon det er innad i clusterne.

Lav WCSS betyr at punktene innad i hvert cluster ligger tett sammen.

---

### Silhouette score

Silhouette score måler hvor godt datapunktene passer inn i sine egne cluster sammenlignet med andre cluster.

Verdien ligger mellom -1 og 1.

Tolkning:

```text
Nær 1: punktet passer godt i sitt cluster
Nær 0: punktet ligger mellom to cluster
Under 0: punktet kan være plassert i feil cluster
```

En høy gjennomsnittlig silhouette score indikerer tydeligere separerte cluster.

---

### Elbow method

Elbow method brukes ofte til å velge antall cluster i K-Means.

Man trener K-Means med ulike verdier for `k`, for eksempel fra 1 til 10, og plotter inertia/WCSS.

Deretter ser man etter et knekkpunkt i grafen.

Dette punktet kalles “elbow”.

Eksempel:

```text
k = 1: høy WCSS
k = 2: mye lavere WCSS
k = 3: lavere WCSS
k = 4: litt lavere WCSS
k = 5: nesten ingen forbedring
```

Hvis forbedringen flater ut etter `k = 3`, kan 3 være et fornuftig valg.

---

## 5. Vanlige clustering-algoritmer

Det finnes mange clustering-algoritmer. Valget av algoritme avhenger av datastruktur, støy, outliers, dimensjoner og formen på clusterne.

### K-Means

K-Means er en av de mest brukte clustering-metodene.

Passer godt når:

* Clusterne er relativt runde
* Clusterne har omtrent lik størrelse
* Dataene er numeriske
* Man har en idé om antall cluster
* Dataene ikke har altfor mye støy

Ulemper:

* Krever at man velger `k`
* Sensitiv for outliers
* Sensitiv for skalering
* Fungerer dårlig hvis clusterne har komplisert form
* Antar omtrent sfæriske cluster

---

### Agglomerative clustering

Agglomerative clustering er en hierarkisk metode.

Den starter ofte med at hvert datapunkt er sitt eget cluster. Deretter slås de nærmeste clusterne sammen steg for steg.

Passer godt når:

* Man ønsker hierarkisk struktur
* Man ikke nødvendigvis vet antall cluster på forhånd
* Man vil visualisere grupper med dendrogram
* Dataene ikke passer godt til K-Means

---

### DBSCAN

DBSCAN er en density-based clustering-algoritme.

Den grupperer punkter som ligger tett sammen, og markerer punkter langt unna som støy.

Passer godt når:

* Clusterne har ujevn eller ikke-sirkulær form
* Datasettet har outliers
* Man ikke vil spesifisere antall cluster direkte
* Tetthet er viktigere enn centroid-avstand

Ulemper:

* Krever valg av `eps` og `min_samples`
* Kan slite hvis clusterne har veldig ulik tetthet
* Sensitiv for skalering

---

### OPTICS

OPTICS ligner på DBSCAN, men fungerer bedre når clusterne har varierende tetthet.

Passer godt når:

* Dataene har ujevn tetthet
* DBSCAN blir for sensitiv for én bestemt `eps`
* Man vil undersøke tetthetsstruktur mer fleksibelt

---

### Gaussian Mixture Models

Gaussian Mixture Models, ofte forkortet GMM, er en probabilistisk clustering-metode.

I stedet for å gi hvert punkt en hard cluster-label, kan GMM gi sannsynligheter.

Eksempel:

```text
Punkt A:
Cluster 0: 0.80
Cluster 1: 0.15
Cluster 2: 0.05
```

Passer godt når:

* Clusterne overlapper
* Man vil ha sannsynligheter
* Dataene kan antas å komme fra flere normalfordelinger
* Man ønsker mykere clustering enn K-Means

---

### BIRCH

BIRCH brukes ofte på store datasett.

Passer godt når:

* Datasettet er stort
* Man trenger en mer minneeffektiv metode
* Det finnes outliers
* Man ønsker rask clustering

---

## 6. Typisk workflow for clustering

En vanlig clustering-prosess kan se slik ut:

```text
1. Last inn data
2. Undersøk datasettet
3. Rens data
4. Velg relevante features
5. Skaler features
6. Visualiser data
7. Velg clustering-algoritme
8. Finn passende antall cluster
9. Tren modellen
10. Evaluer clusterne
11. Visualiser resultatet
12. Tolk clusterne faglig
```

Clustering er ofte mer iterativt enn supervised learning. Det er vanlig å teste flere features, skaleringer, algoritmer og verdier for antall cluster.

---

## 7. Importere biblioteker

```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
```

---

## 8. Laste inn data

Eksempelet under bruker et datasett med sanger.

```python
df = pd.read_csv("../data/nigerian-songs.csv")

df.head()
```

Typiske kolonner kan være:

```text
name
album
artist
artist_top_genre
release_date
length
popularity
danceability
acousticness
energy
instrumentalness
liveness
loudness
speechiness
tempo
time_signature
```

---

## 9. Første inspeksjon av datasettet

```python
df.info()
```

Dette viser:

* Antall rader
* Antall kolonner
* Datatyper
* Manglende verdier
* Minnebruk

Deretter kan man sjekke manglende verdier:

```python
df.isnull().sum()
```

Og statistisk sammendrag:

```python
df.describe()
```

---

## 10. Fjerne eller håndtere manglende verdier

Hvis datasettet har manglende verdier, må disse håndteres før clustering.

Eksempel:

```python
df = df.dropna()
```

Eventuelt kan man fylle inn verdier:

```python
df["popularity"] = df["popularity"].fillna(df["popularity"].median())
```

For clustering bør man være forsiktig med imputation, fordi kunstig fylte verdier kan påvirke avstandsberegningen.

---

## 11. Filtrering av støy i datasettet

I musikkdatasettet kan noen rader ha genre satt til `"Missing"`.

```python
df = df[df["artist_top_genre"] != "Missing"]
```

Hvis `popularity = 0` betyr at sangen ikke har en reell popularitetsverdi, kan disse fjernes:

```python
df = df[df["popularity"] > 0]
```

Man kan også filtrere til bestemte sjangre:

```python
selected_genres = ["afro dancehall", "afropop", "nigerian pop"]

df = df[df["artist_top_genre"].isin(selected_genres)]
```

---

## 12. Visualisere fordeling av kategorier

Selv om clustering ikke trenger labels, kan eksisterende labels være nyttige i analysefasen.

Eksempel:

```python
top = df["artist_top_genre"].value_counts()

plt.figure(figsize=(10, 6))
sns.barplot(x=top.index, y=top.values)
plt.xticks(rotation=45)
plt.title("Antall sanger per sjanger")
plt.xlabel("Sjanger")
plt.ylabel("Antall")
plt.show()
```

Dette gir en rask oversikt over om datasettet er balansert eller dominert av enkelte grupper.

---

## 13. Hvorfor labels kan være nyttige selv i clustering

Clustering er unsupervised, så modellen bruker egentlig ikke labels.

Likevel kan labels brukes til:

* Exploratory data analysis
* Visualisering
* Sammenligning etter clustering
* Tolking av cluster
* Sjekke om kjente grupper samsvarer med naturlige grupper

Eksempel:

```python
sns.scatterplot(
    data=df,
    x="popularity",
    y="danceability",
    hue="artist_top_genre"
)
plt.show()
```

Dette kan gi en indikasjon på om sjangrene faktisk skiller seg fra hverandre i feature space.

---

## 14. Korrelasjonsanalyse

Før clustering kan det være nyttig å undersøke korrelasjoner mellom numeriske variabler.

```python
corrmat = df.corr(numeric_only=True)

plt.figure(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.8, square=True, cmap="coolwarm")
plt.title("Korrelasjonsmatrise")
plt.show()
```

Dersom to features er sterkt korrelert, kan de representere mye av den samme informasjonen.

Eksempel:

```text
energy og loudness kan ofte være positivt korrelert.
```

Dette gir mening fordi høyere lydstyrke ofte oppleves som mer energisk.

Men korrelasjon betyr ikke kausalitet.

---

## 15. Visualisere datapunkter før clustering

En enkel scatterplot kan vise om det finnes synlige grupper.

```python
sns.scatterplot(
    data=df,
    x="popularity",
    y="danceability",
    hue="artist_top_genre"
)

plt.title("Popularity vs Danceability")
plt.show()
```

Hvis punktene overlapper mye, kan clustering bli vanskelig.

Hvis punktene danner tydelige grupper, kan clustering fungere bedre.

---

## 16. KDE-plot for fordeling

KDE står for Kernel Density Estimate.

Det brukes for å visualisere sannsynlighetstetthet.

```python
sns.set_theme(style="ticks")

sns.jointplot(
    data=df,
    x="popularity",
    y="danceability",
    hue="artist_top_genre",
    kind="kde"
)

plt.show()
```

Dette kan vise om flere grupper har lignende fordeling.

Hvis sjangrene overlapper sterkt, vil clustering sannsynligvis ikke separere dem tydelig.

---

## 17. Boxplot for å se outliers

Outliers kan påvirke clustering kraftig, spesielt K-Means.

```python
numeric_cols = [
    "popularity",
    "danceability",
    "acousticness",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo",
    "time_signature",
    "length",
    "release_date"
]

plt.figure(figsize=(20, 20), dpi=150)

for i, col in enumerate(numeric_cols, 1):
    plt.subplot(4, 3, i)
    sns.boxplot(x=df[col])
    plt.title(col)

plt.tight_layout()
plt.show()
```

Boxplots gjør det lettere å se hvilke variabler som har ekstreme verdier.

---

## 18. Outliers og clustering

Outliers kan skape problemer fordi mange clustering-algoritmer bruker avstand.

Eksempel:

```text
Et ekstremt datapunkt kan dra centroiden bort fra den naturlige gruppen.
```

Dette gjelder spesielt K-Means.

Mulige løsninger:

* Fjerne ekstreme outliers
* Winsorize verdier
* Bruke robust skalering
* Bruke DBSCAN
* Bruke transformasjoner som log-transformasjon
* Velge features med mindre ekstreme verdier

Eksempel på enkel outlier-filtrering med IQR:

```python
def remove_outliers_iqr(data, columns):
    df_clean = data.copy()

    for col in columns:
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        df_clean = df_clean[
            (df_clean[col] >= lower) &
            (df_clean[col] <= upper)
        ]

    return df_clean
```

Bruk:

```python
df_clean = remove_outliers_iqr(
    df,
    ["popularity", "danceability", "energy", "loudness"]
)
```

Man bør ikke fjerne outliers automatisk uten faglig vurdering. I noen problemer er outliers nettopp det mest interessante.

---

## 19. Velge features

Clustering påvirkes sterkt av hvilke features man bruker.

Eksempel:

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

Gode clustering-features bør:

* Være relevante for problemet
* Ha nok variasjon
* Ikke være rent identifiserende
* Ikke dominere kun fordi de har stor skala
* Ikke være ekstremt støyete
* Ikke være duplikater av hverandre

---

## 20. Skalering av data

Skalering er ofte kritisk i clustering.

K-Means bruker avstand. Hvis én feature har mye større tallområde enn andre, vil den dominere avstandsberegningen.

Eksempel:

```text
length: 90 000 til 500 000
danceability: 0 til 1
energy: 0 til 1
loudness: -20 til 1
```

Uten skalering kan `length` dominere modellen.

Standardisering:

```python
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

Konverter tilbake til DataFrame:

```python
X_scaled = pd.DataFrame(
    X_scaled,
    columns=features,
    index=X.index
)
```

StandardScaler gjør at hver feature får:

```text
mean = 0
standard deviation = 1
```

Dette gjør features mer sammenlignbare for avstandsbaserte algoritmer.

---

## 21. Label encoding av kategoriske variabler

Noen ganger finnes det kategoriske variabler i datasettet.

Eksempel:

```python
le = LabelEncoder()

df["genre_encoded"] = le.fit_transform(df["artist_top_genre"])
```

Men man bør være forsiktig.

Label encoding lager tall som:

```text
afro dancehall = 0
afropop = 1
nigerian pop = 2
```

Problemet er at modellen kan tolke tallene som ordinale, altså at `2 > 1 > 0`.

For clustering bør kategoriske variabler vanligvis håndteres med:

* One-hot encoding
* Separate analyser
* Egnede avstandsmål
* Algoritmer som håndterer kategoriske data bedre

I mange tilfeller bør labels som `artist_top_genre` ikke brukes som input til clustering, men heller brukes etterpå for tolkning.

---

## 22. K-Means forklart

K-Means forsøker å dele dataene inn i `k` cluster.

Algoritmen gjør omtrent dette:

```text
1. Velg k tilfeldige start-centroider
2. Tildel hvert punkt til nærmeste centroid
3. Beregn nye centroider basert på gjennomsnittet av punktene i hvert cluster
4. Gjenta til centroidene slutter å flytte seg mye
```

Målet er å minimere avstanden mellom punktene og centroiden i clusteret de tilhører.

---

## 23. Trene en enkel K-Means-modell

```python
from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=3,
    init="k-means++",
    random_state=42,
    n_init=10
)

kmeans.fit(X_scaled)
```

Predikere cluster for hvert datapunkt:

```python
labels = kmeans.predict(X_scaled)
```

Legge labels tilbake i DataFrame:

```python
df["cluster"] = labels
```

Se resultatet:

```python
df[["name", "artist", "artist_top_genre", "cluster"]].head()
```

---

## 24. Hva betyr `n_clusters`?

`n_clusters` er antall grupper modellen skal lage.

Eksempel:

```python
KMeans(n_clusters=3)
```

Dette betyr at modellen tvinges til å dele dataene inn i tre cluster.

Det betyr ikke nødvendigvis at det faktisk finnes tre naturlige grupper.

Derfor bør `k` undersøkes med metoder som:

* Elbow method
* Silhouette score
* Faglig tolkning
* Visualisering
* Stabilitet på tvers av kjøringer

---

## 25. Hva betyr `random_state`?

K-Means starter med initielle centroider.

Siden dette innebærer tilfeldighet, kan resultatet variere mellom kjøringer.

```python
random_state=42
```

gjør at resultatet blir reproduserbart.

Dette er viktig når man skal:

* Dele kode
* Lage rapport
* Sammenligne modeller
* Feilsøke
* Dokumentere analyse

---

## 26. Hva betyr `k-means++`?

`k-means++` er en smartere metode for initialisering av centroider.

I stedet for å velge tilfeldige startpunkter helt tilfeldig, forsøker `k-means++` å plassere start-centroidene mer spredt.

Dette kan gi:

* Raskere konvergens
* Mer stabile resultater
* Bedre clustering enn helt tilfeldig initialisering

Eksempel:

```python
KMeans(init="k-means++")
```

---

## 27. Finne antall cluster med elbow method

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

Plot resultatet:

```python
plt.figure(figsize=(10, 5))
sns.lineplot(x=range(1, 11), y=wcss, marker="o")

plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS / Inertia")
plt.show()
```

Tolkning:

```text
Se etter punktet der kurven begynner å flate ut.
```

Hvis kurven får et tydelig knekk ved `k = 3`, kan tre cluster være et godt valg.

---

## 28. Finne antall cluster med silhouette score

```python
silhouette_scores = []

for k in range(2, 11):
    model = KMeans(
        n_clusters=k,
        init="k-means++",
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
```

Plot:

```python
plt.figure(figsize=(10, 5))
sns.lineplot(x=range(2, 11), y=silhouette_scores, marker="o")

plt.title("Silhouette Score for ulike k")
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette score")
plt.show()
```

Velg gjerne en `k` som gir høy silhouette score, men ikke bruk metrikken blindt.

Et høyere tall er ikke alltid bedre dersom clusterne ikke gir faglig mening.

---

## 29. Samlet funksjon for å teste flere k-verdier

```python
def evaluate_kmeans(X, k_min=2, k_max=10):
    results = []

    for k in range(k_min, k_max + 1):
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(X)

        inertia = model.inertia_
        silhouette = silhouette_score(X, labels)

        results.append({
            "k": k,
            "inertia": inertia,
            "silhouette": silhouette
        })

    return pd.DataFrame(results)
```

Bruk:

```python
results = evaluate_kmeans(X_scaled, 2, 10)

results
```

Plot:

```python
plt.figure(figsize=(10, 5))
sns.lineplot(data=results, x="k", y="inertia", marker="o")
plt.title("Inertia per k")
plt.show()

plt.figure(figsize=(10, 5))
sns.lineplot(data=results, x="k", y="silhouette", marker="o")
plt.title("Silhouette score per k")
plt.show()
```

---

## 30. Visualisere cluster med scatterplot

Hvis man har to features, kan man visualisere cluster direkte.

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

Dette gir en enkel visuell kontroll.

Men hvis modellen er trent på flere features, viser et 2D-plot bare et utsnitt av clusterstrukturen.

---

## 31. Visualisere cluster med PCA

Hvis man har mange features, kan PCA brukes til å redusere dimensjoner for visualisering.

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

plt.title("Cluster visualisert med PCA")
plt.show()
```

PCA brukes her for visualisering, ikke nødvendigvis som en del av selve modellen.

---

## 32. Tolke cluster

Etter clustering bør man undersøke hva som kjennetegner hvert cluster.

```python
cluster_summary = df.groupby("cluster")[features].mean()

cluster_summary
```

Man kan også se median:

```python
df.groupby("cluster")[features].median()
```

Og antall observasjoner:

```python
df["cluster"].value_counts()
```

Dette hjelper med å gi clusterne meningsfulle navn.

Eksempel:

```text
Cluster 0: høy danceability, høy energy, høy popularity
Cluster 1: lav popularity, lav loudness, høy acousticness
Cluster 2: middels popularity, høy loudness, lav acousticness
```

---

## 33. Visualisere cluster-profiler

```python
cluster_summary = df.groupby("cluster")[features].mean()

cluster_summary.T.plot(kind="bar", figsize=(12, 6))

plt.title("Gjennomsnittlige feature-verdier per cluster")
plt.xlabel("Feature")
plt.ylabel("Gjennomsnitt")
plt.xticks(rotation=45)
plt.show()
```

Dette gjør det enklere å sammenligne clusterne.

---

## 34. Sammenligne cluster med eksisterende labels

Selv om clustering ikke bruker labels, kan man sammenligne cluster med labels etterpå.

```python
pd.crosstab(df["cluster"], df["artist_top_genre"])
```

Dette viser om clusterne overlapper med kjente sjangre.

Hvis clusterne ikke samsvarer med sjangrene, betyr ikke det nødvendigvis at modellen er dårlig.

Det kan bety at:

* Sjangrene overlapper musikalsk
* Features ikke skiller sjangrene godt
* Det finnes andre mønstre enn sjanger
* Dataene er for støyete
* Antall cluster er feil
* Algoritmen passer dårlig til datastrukturen

---

## 35. Hvorfor accuracy ofte er feil metrikk i clustering

Accuracy brukes i supervised learning, der man har ekte labels.

I clustering finnes det vanligvis ingen fasit.

Derfor bør man være forsiktig med kode som dette:

```python
correct_labels = sum(y == labels)

accuracy = correct_labels / len(y)
```

Dette kan være misvisende fordi cluster-ID-er er tilfeldige.

Eksempel:

```text
Ekte label:      0 0 1 1 2 2
Cluster-label:   1 1 2 2 0 0
```

Her er clusterne egentlig perfekte, men label-numrene matcher ikke.

Bedre alternativer:

* Silhouette score
* Adjusted Rand Index hvis man har ekte labels
* Normalized Mutual Information hvis man har ekte labels
* Visuell analyse
* Faglig tolkning av cluster-profiler

---

## 36. Adjusted Rand Index

Hvis man faktisk har ekte labels og vil sammenligne clustering med labels, kan man bruke Adjusted Rand Index.

```python
from sklearn.metrics import adjusted_rand_score

ari = adjusted_rand_score(y_true, labels)

print(ari)
```

Tolkning:

```text
1.0 = perfekt samsvar
0.0 = omtrent tilfeldig
negativ = dårligere enn tilfeldig
```

Dette er bedre enn vanlig accuracy fordi den tar hensyn til at cluster-ID-er kan være permutert.

---

## 37. Normalized Mutual Information

Et annet mål er Normalized Mutual Information.

```python
from sklearn.metrics import normalized_mutual_info_score

nmi = normalized_mutual_info_score(y_true, labels)

print(nmi)
```

NMI måler hvor mye informasjon clusterne gir om de ekte labelsene.

---

## 38. Komplett K-Means-eksempel

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# 1. Last inn data
df = pd.read_csv("../data/nigerian-songs.csv")

# 2. Enkel filtrering
df = df[df["artist_top_genre"] != "Missing"]
df = df[df["popularity"] > 0]

selected_genres = ["afro dancehall", "afropop", "nigerian pop"]
df = df[df["artist_top_genre"].isin(selected_genres)]

# 3. Velg features
features = [
    "popularity",
    "danceability",
    "acousticness",
    "loudness",
    "energy"
]

X = df[features].copy()

# 4. Skaler data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Finn mulig k med silhouette score
scores = []

for k in range(2, 11):
    model = KMeans(
        n_clusters=k,
        init="k-means++",
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

print(scores_df)

# 6. Tren endelig modell
kmeans = KMeans(
    n_clusters=3,
    init="k-means++",
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans.fit_predict(X_scaled)

# 7. Cluster summary
cluster_summary = df.groupby("cluster")[features].mean()

print(cluster_summary)

# 8. PCA for visualisering
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df["pca_1"] = X_pca[:, 0]
df["pca_2"] = X_pca[:, 1]

# 9. Plot cluster
plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="pca_1",
    y="pca_2",
    hue="cluster",
    palette="viridis"
)

plt.title("K-Means clustering visualisert med PCA")
plt.show()
```

---

## 39. DBSCAN-eksempel

DBSCAN krever ikke at man bestemmer antall cluster på forhånd.

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(
    eps=0.8,
    min_samples=5
)

df["dbscan_cluster"] = dbscan.fit_predict(X_scaled)
```

DBSCAN bruker label `-1` for støy/outliers.

```python
df["dbscan_cluster"].value_counts()
```

Visualisering:

```python
plt.figure(figsize=(8, 6))

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

## 40. Hvordan velge `eps` i DBSCAN?

`eps` bestemmer hvor nær punkter må være hverandre for å regnes som naboer.

For liten `eps`:

```text
Mange punkter blir støy
```

For stor `eps`:

```text
For mange punkter havner i samme cluster
```

Man kan undersøke dette med nearest neighbors.

```python
from sklearn.neighbors import NearestNeighbors

neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X_scaled)

distances, indices = neighbors_fit.kneighbors(X_scaled)

distances = np.sort(distances[:, 4])

plt.figure(figsize=(8, 5))
plt.plot(distances)
plt.title("k-distance plot")
plt.xlabel("Punkter sortert etter avstand")
plt.ylabel("Avstand til 5. nærmeste nabo")
plt.show()
```

Se etter et knekkpunkt i grafen. Dette kan være et utgangspunkt for `eps`.

---

## 41. Agglomerative clustering-eksempel

```python
from sklearn.cluster import AgglomerativeClustering

agg = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)

df["agg_cluster"] = agg.fit_predict(X_scaled)
```

Visualisering:

```python
plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="pca_1",
    y="pca_2",
    hue="agg_cluster",
    palette="viridis"
)

plt.title("Agglomerative clustering")
plt.show()
```

---

## 42. Gaussian Mixture Model-eksempel

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(
    n_components=3,
    random_state=42
)

df["gmm_cluster"] = gmm.fit_predict(X_scaled)
```

Sannsynlighet for hvert cluster:

```python
cluster_probs = gmm.predict_proba(X_scaled)

cluster_probs[:5]
```

Dette kan være nyttig hvis punktene ikke tilhører clusterne helt tydelig.

---

## 43. Sammenligne flere algoritmer

```python
models = {
    "kmeans": KMeans(n_clusters=3, random_state=42, n_init=10),
    "agglomerative": AgglomerativeClustering(n_clusters=3),
    "gmm": GaussianMixture(n_components=3, random_state=42)
}

results = {}

for name, model in models.items():
    labels = model.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)

    results[name] = {
        "silhouette": score,
        "n_clusters": len(set(labels))
    }

pd.DataFrame(results).T
```

For DBSCAN må man passe på at modellen kan lage `-1` som støy-label.

```python
dbscan = DBSCAN(eps=0.8, min_samples=5)

labels = dbscan.fit_predict(X_scaled)

valid_labels = labels != -1

if len(set(labels[valid_labels])) > 1:
    score = silhouette_score(X_scaled[valid_labels], labels[valid_labels])
else:
    score = None

print(score)
```

---

## 44. Vanlige problemer i clustering

### Problem 1: Dataene overlapper

Hvis gruppene overlapper mye, vil clusterne bli uklare.

Løsninger:

* Test andre features
* Bruk dimensjonsreduksjon
* Test andre algoritmer
* Aksepter at dataene ikke har tydelige cluster

---

### Problem 2: Feil skalering

Hvis features har svært ulik skala, kan én feature dominere.

Løsning:

```python
X_scaled = StandardScaler().fit_transform(X)
```

---

### Problem 3: For mange irrelevante features

Irrelevante features kan gjøre avstandsberegningen dårligere.

Løsninger:

* Feature selection
* PCA
* Faglig vurdering
* Korrelasjonsanalyse

---

### Problem 4: Outliers drar centroidene

K-Means er sensitiv for ekstreme verdier.

Løsninger:

* Fjern ekstreme outliers
* Bruk RobustScaler
* Bruk DBSCAN
* Bruk GMM
* Bruk medianbaserte metoder

---

### Problem 5: Feil antall cluster

Hvis `k` er feil, kan K-Means lage kunstige grupper.

Løsninger:

* Elbow method
* Silhouette score
* Flere visualiseringer
* Faglig tolkning
* Stabilitetstesting

---

## 45. Clusteranalyse med pipeline

For mer ryddig kode kan man bruke pipeline.

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(
        n_clusters=3,
        init="k-means++",
        random_state=42,
        n_init=10
    ))
])

labels = pipeline.fit_predict(X)

df["cluster"] = labels
```

Dette gjør det tydelig at skalering og modell henger sammen.

---

## 46. Lagre clustering-resultater

```python
df.to_csv("clustered_songs.csv", index=False)
```

Lagre bare relevante kolonner:

```python
output_cols = [
    "name",
    "artist",
    "artist_top_genre",
    "popularity",
    "danceability",
    "energy",
    "loudness",
    "cluster"
]

df[output_cols].to_csv("clustered_songs_summary.csv", index=False)
```

---

## 47. Bruke cluster som feature i senere modellering

Etter clustering kan cluster-ID brukes som en ny feature i supervised learning.

Eksempel:

```python
df["cluster"] = labels
```

Deretter kan `cluster` brukes i en senere modell.

Dette kan være nyttig hvis clusterne representerer latent struktur i dataene.

Eksempel:

```text
Kundesegment kan brukes som input i churn prediction.
```

Men man må passe på datalekkasje. Hvis clustering gjøres før train/test split, kan informasjon fra testsettet lekke inn i treningsprosessen.

---

## 48. Clustering og train/test split

I supervised learning er train/test split standard.

I clustering er det litt annerledes, fordi modellen ikke trenes mot en target.

Likevel bør man tenke på generalisering dersom clusterne skal brukes i produksjon.

For K-Means kan man gjøre:

```python
from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(
    X,
    test_size=0.2,
    random_state=42
)

pipeline.fit(X_train)

train_labels = pipeline.predict(X_train)
test_labels = pipeline.predict(X_test)
```

Dette gir mening hvis man vil bruke modellen til å tildele nye observasjoner til eksisterende cluster.

---

## 49. Inductive vs transductive clustering

Noen clustering-metoder kan brukes til å predikere cluster for nye datapunkter.

Eksempel:

```python
kmeans.predict(new_data)
```

Dette er nyttig i produksjon.

Andre metoder lager cluster bare for datasettet de trenes på, uten en naturlig `predict`-metode.

Eksempel:

```python
AgglomerativeClustering
DBSCAN i scikit-learn
```

Dette skillet er viktig hvis modellen skal brukes på nye data.

---

## 50. Når passer K-Means?

K-Means passer ofte bra når:

* Dataene er numeriske
* Clusterne er omtrent sfæriske
* Clusterne har relativt lik størrelse
* Man ønsker en enkel og rask metode
* Man må kunne predikere cluster for nye observasjoner
* Man har mange datapunkter

K-Means passer dårlig når:

* Clusterne har kompleks form
* Det er mange outliers
* Clusterne har ulik tetthet
* Features ikke er skalert
* Antall cluster er uklart
* Dataene er kategoriske

---

## 51. Når passer DBSCAN?

DBSCAN passer ofte bra når:

* Man forventer outliers
* Clusterne har uregelmessig form
* Man ikke vet antall cluster
* Tetthet er sentralt
* Man vil skille støy fra grupper

DBSCAN passer dårlig når:

* Datasettet har svært ulik tetthet
* Dimensjonene er veldig høye
* `eps` er vanskelig å velge
* Man trenger enkel prediksjon på nye data

---

## 52. Når passer GMM?

Gaussian Mixture Models passer ofte bra når:

* Clusterne overlapper
* Man vil ha sannsynligheter
* Dataene kan modelleres som blanding av normalfordelinger
* Man ønsker myk clustering

GMM passer dårlig når:

* Dataene har veldig ikke-gaussisk form
* Det er mange outliers
* Antall komponenter er uklart
* Man trenger veldig robust clustering

---

## 53. Praktisk sjekkliste

Før clustering:

```text
Har jeg valgt relevante features?
Er dataene numeriske?
Har jeg håndtert manglende verdier?
Har jeg vurdert outliers?
Har jeg skalert dataene?
Har jeg visualisert dataene?
```

Under clustering:

```text
Har jeg testet flere k-verdier?
Har jeg brukt elbow method?
Har jeg sjekket silhouette score?
Har jeg testet flere algoritmer?
Har jeg vurdert stabilitet?
```

Etter clustering:

```text
Kan clusterne tolkes faglig?
Er clusterne balanserte?
Gir clusterne mening visuelt?
Stemmer resultatet med domenekunnskap?
Er clusterne nyttige for videre analyse?
```

---

## 54. Viktige takeaways

Clustering er ikke en fasitmodell. Den finner struktur basert på antakelser i algoritmen.

K-Means finner ikke “sanne grupper”. Den finner grupper som minimerer avstand til centroider.

Featurevalg og skalering er ofte viktigere enn selve algoritmen.

Silhouette score og elbow method er nyttige, men må tolkes sammen med visualisering og faglig forståelse.

Dårlige cluster kan skyldes:

* Svake features
* Overlappende grupper
* Feil algoritme
* Manglende skalering
* Outliers
* For høy varians
* Ingen reell clusterstruktur i dataene

I praksis er clustering mest verdifullt når clusterne kan tolkes og brukes til noe konkret.

---

## 55. Kort kodeoppskrift

```python
# 1. Velg features
features = ["popularity", "danceability", "acousticness", "loudness", "energy"]
X = df[features]

# 2. Skaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Finn k
wcss = []

for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    wcss.append(model.inertia_)

plt.plot(range(1, 11), wcss, marker="o")
plt.xlabel("k")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# 4. Tren modell
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

# 5. Evaluer
score = silhouette_score(X_scaled, df["cluster"])
print(score)

# 6. Tolk
print(df.groupby("cluster")[features].mean())
```

---

## 56. Mini-oppsummering

Clustering brukes når vi ønsker å finne grupper i data uten fasit-labels. Den viktigste ideen er at datapunkter som ligner på hverandre, havner i samme cluster. K-Means er den vanligste metoden og bygger på centroider, avstand og minimering av inertia. For å velge antall cluster bruker man ofte elbow method og silhouette score. Resultatet bør alltid tolkes med visualisering, feature-analyse og domeneforståelse.

Clustering handler derfor ikke bare om å kjøre en algoritme. Det handler om å undersøke om datasettet faktisk har en struktur som er verdt å gruppere.
